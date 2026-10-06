"""Small temporal classifiers and input-gradient explanations for sepsis windows.

The models consume ``values`` with shape ``(batch, time, features)``, an
observation mask with the same shape, and a valid-time mask with shape
``(batch, time)``. They return one binary-classification logit per window with
shape ``(batch,)``. Invalid (left-padded) steps are removed before recurrence
or excluded from transformer attention, so adding more left padding does not
change a prediction.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


def _validate_inputs(values: Tensor, observed_mask: Tensor, valid_time_mask: Tensor,
                     feature_dim: int) -> tuple[Tensor, Tensor]:
    if values.ndim != 3 or values.shape[-1] != feature_dim:
        raise ValueError(f"values must have shape (batch, time, {feature_dim})")
    if observed_mask.shape != values.shape:
        raise ValueError("observed_mask must have the same shape as values")
    if valid_time_mask.shape != values.shape[:2]:
        raise ValueError("valid_time_mask must have shape (batch, time)")
    if values.shape[1] == 0:
        raise ValueError("time dimension must be nonempty")
    # Floating observation masks are used by mask integrated gradients. Keep
    # their graph and fractional values; ordinary callers retain boolean masks.
    if not observed_mask.is_floating_point():
        observed_mask = observed_mask.to(dtype=torch.bool)
    return observed_mask, valid_time_mask.to(dtype=torch.bool)


def _model_input(values: Tensor, observed_mask: Tensor, valid_time_mask: Tensor,
                 feature_dim: int, use_observation_mask: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    observed_mask, valid_time_mask = _validate_inputs(
        values, observed_mask, valid_time_mask, feature_dim
    )
    features = (torch.cat((values, observed_mask.to(dtype=values.dtype)), dim=-1)
                if use_observation_mask else values)
    # Do not let arbitrary values or observation flags at padded positions leak
    # into the models. This also makes the empty-window representation zero.
    features = features.masked_fill(~valid_time_mask.unsqueeze(-1), 0)
    return features, observed_mask, valid_time_mask


def _compact_steps(features: Tensor, valid_time_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Move valid steps to the front while retaining their chronological order."""
    batch, time, width = features.shape
    lengths = valid_time_mask.sum(dim=1)
    compact = features.new_zeros((batch, time, width))
    for row in range(batch):
        valid = features[row, valid_time_mask[row]]
        if valid.numel():
            compact[row, :valid.shape[0]] = valid
    return compact, lengths


class _RecurrentClassifier(nn.Module):
    recurrent_type: type[nn.LSTM] | type[nn.GRU]

    def __init__(self, feature_dim: int = 40, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0,
                 use_observation_mask: bool = True) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("feature_dim, hidden_size, and num_layers must be positive")
        self.feature_dim = feature_dim
        self.use_observation_mask = use_observation_mask
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.recurrent = self.recurrent_type(
            input_size=feature_dim * (2 if use_observation_mask else 1), hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=recurrent_dropout,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, values: Tensor, observed_mask: Tensor,
                valid_time_mask: Tensor) -> Tensor:
        features, _, valid_time_mask = _model_input(
            values, observed_mask, valid_time_mask, self.feature_dim, self.use_observation_mask
        )
        compact, lengths = _compact_steps(features, valid_time_mask)
        # pack_padded_sequence rejects zero lengths. For empty windows, process a
        # single zero step, whose hidden state is the recurrent zero-input state.
        packed = pack_padded_sequence(
            compact, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.classifier(hidden[-1]).squeeze(-1)


class LSTMClassifier(_RecurrentClassifier):
    """LSTM classifier over a left-padded window."""
    recurrent_type = nn.LSTM


class GRUClassifier(_RecurrentClassifier):
    """GRU classifier over a left-padded window."""
    recurrent_type = nn.GRU


class TemporalTransformerClassifier(nn.Module):
    """Transformer encoder with valid-step attention masking and sinusoidal time positions."""

    def __init__(self, feature_dim: int = 40, d_model: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.0,
                 use_observation_mask: bool = True) -> None:
        super().__init__()
        if feature_dim <= 0 or d_model <= 0 or num_layers <= 0:
            raise ValueError("feature_dim, d_model, and num_layers must be positive")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by positive num_heads")
        self.feature_dim = feature_dim
        self.use_observation_mask = use_observation_mask
        self.d_model = d_model
        self.input_projection = nn.Linear(feature_dim * (2 if use_observation_mask else 1), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def _positions(self, batch: int, time: int, device: torch.device,
                   dtype: torch.dtype) -> Tensor:
        position = torch.arange(time, device=device, dtype=dtype).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        encoded = torch.zeros((time, self.d_model), device=device, dtype=dtype)
        encoded[:, 0::2] = torch.sin(position * scale)
        if self.d_model > 1:
            encoded[:, 1::2] = torch.cos(position * scale[:encoded[:, 1::2].shape[1]])
        return encoded.unsqueeze(0).expand(batch, -1, -1)

    def forward(self, values: Tensor, observed_mask: Tensor,
                valid_time_mask: Tensor) -> Tensor:
        features, _, valid_time_mask = _model_input(
            values, observed_mask, valid_time_mask, self.feature_dim, self.use_observation_mask
        )
        compact, lengths = _compact_steps(features, valid_time_mask)
        batch, time, _ = compact.shape
        positions = self._positions(batch, time, compact.device, compact.dtype)
        encoded = self.input_projection(compact) + positions
        # Valid steps were compacted to the front, so the original left-padding
        # mask no longer indexes the encoded sequence. Build its compact form.
        compact_valid = (torch.arange(time, device=values.device)[None, :]
                         < lengths[:, None])
        # An all-padding row needs one safe key to avoid all-masked attention.
        attention_valid = compact_valid.clone()
        empty_rows = lengths == 0
        if empty_rows.any():
            attention_valid[empty_rows, 0] = True
        encoded = self.encoder(encoded, src_key_padding_mask=~attention_valid)
        last_indices = lengths.clamp_min(1) - 1
        final = encoded[torch.arange(batch, device=values.device), last_indices]
        return self.classifier(final).squeeze(-1)


def integrated_gradients(
    model: nn.Module,
    values: Tensor,
    observed_mask: Tensor,
    valid_time_mask: Tensor,
    *,
    baseline: Tensor | None = None,
    steps: int = 64,
) -> Tensor:
    """Return feature-time integrated gradients for each window's binary logit.

    By default the baseline is all-zero *input values*; observation and valid-time
    masks stay fixed at their supplied values. Thus the attribution sum estimates
    ``logit(values, masks) - logit(baseline, masks)``. This is a model explanation,
    not a clinical interpretation. The trapezoidal path integral improves
    completeness as ``steps`` increases.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if baseline is None:
        baseline = torch.zeros_like(values)
    elif baseline.shape != values.shape:
        raise ValueError("baseline must have the same shape as values")
    baseline = baseline.to(device=values.device, dtype=values.dtype)
    delta = values - baseline
    was_training = model.training
    model.eval()
    try:
        total_gradients = torch.zeros_like(values)
        for index in range(steps + 1):
            alpha = index / steps
            point = (baseline + alpha * delta).detach().requires_grad_(True)
            logits = model(point, observed_mask, valid_time_mask)
            gradients, = torch.autograd.grad(logits.sum(), point)
            weight = 0.5 if index in (0, steps) else 1.0
            total_gradients = total_gradients + weight * gradients
        return delta * (total_gradients / steps)
    finally:
        model.train(was_training)


def missingness_integrated_gradients(
    model: nn.Module,
    values: Tensor,
    observed_mask: Tensor,
    valid_time_mask: Tensor,
    *,
    steps: int = 64,
) -> Tensor:
    """Attribute the conditional raw-logit change from an all-missing mask.

    Values are fixed at their current input throughout the path, valid-time
    padding remains fixed, and observation masks interpolate from zero to the
    supplied flags. The returned mask IG sums to
    ``logit(values, observed_mask) - logit(values, zero_mask)`` (up to numerical
    quadrature error). This is a separate conditional baseline from value IG.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not getattr(model, "use_observation_mask", False):
        raise ValueError("model does not use observation masks")
    mask = observed_mask.to(device=values.device, dtype=values.dtype)
    valid = valid_time_mask.to(device=values.device, dtype=torch.bool)
    mask = mask * valid.unsqueeze(-1).to(mask.dtype)
    was_training = model.training
    model.eval()
    try:
        total_gradients = torch.zeros_like(mask)
        for index in range(steps + 1):
            alpha = index / steps
            point = (alpha * mask).detach().requires_grad_(True)
            logits = model(values, point, valid)
            gradients, = torch.autograd.grad(logits.sum(), point)
            weight = 0.5 if index in (0, steps) else 1.0
            total_gradients = total_gradients + weight * gradients
        return mask * (total_gradients / steps)
    finally:
        model.train(was_training)

