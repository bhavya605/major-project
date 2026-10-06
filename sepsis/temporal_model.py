"""Persistence and causal hourly inference for trained temporal research models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import Preprocessor
from .schema import FEATURE_COLUMNS
from .temporal import (GRUClassifier, LSTMClassifier,
                       TemporalTransformerClassifier, integrated_gradients,
                       missingness_integrated_gradients)


@dataclass
class TemporalModelBundle:
    """Frozen model, training-only preprocessing, and operating-point metadata."""

    model: nn.Module
    preprocessor: Preprocessor
    feature_names: tuple[str, ...]
    window_hours: int
    horizon_hours: int
    threshold: float
    persistence_hours: int
    source_type: str
    metadata: dict[str, Any]
    architecture: str
    device: str = "cpu"
    calibrator: Any | None = None


def _architecture(name: str, feature_dim: int, params: dict[str, Any]) -> nn.Module:
    if name == "gru":
        return GRUClassifier(feature_dim=feature_dim, **params)
    if name == "lstm":
        return LSTMClassifier(feature_dim=feature_dim, **params)
    if name == "transformer":
        return TemporalTransformerClassifier(feature_dim=feature_dim, **params)
    raise ValueError(f"unknown temporal architecture: {name}")


def save_temporal_model(bundle: TemporalModelBundle, path: str | Path) -> Path:
    """Write a portable state-dict bundle. Checkpoints are trusted local artifacts."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    params = dict(bundle.metadata.get("model_parameters", {}))
    torch.save({
        "format": "sepsis-temporal-v1",
        "architecture": bundle.architecture,
        "feature_names": list(bundle.feature_names),
        "window_hours": int(bundle.window_hours),
        "horizon_hours": int(bundle.horizon_hours),
        "threshold": float(bundle.threshold),
        "persistence_hours": int(bundle.persistence_hours),
        "source_type": bundle.source_type,
        "metadata": bundle.metadata,
        "preprocessor": {
            "medians": np.asarray(bundle.preprocessor.medians),
            "means": np.asarray(bundle.preprocessor.means),
            "scales": np.asarray(bundle.preprocessor.scales),
        },
        "model_parameters": params,
        "state_dict": bundle.model.state_dict(),
        "calibrator": bundle.calibrator,
    }, target)
    return target


def load_temporal_model(path: str | Path) -> TemporalModelBundle:
    """Load a temporal bundle saved by :func:`save_temporal_model`."""
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "sepsis-temporal-v1":
        raise ValueError("File does not contain a supported temporal research model")
    feature_names = tuple(checkpoint["feature_names"])
    if feature_names != tuple(FEATURE_COLUMNS):
        raise ValueError("checkpoint feature order does not match the canonical schema")
    pre = checkpoint["preprocessor"]
    preprocessor = Preprocessor(np.asarray(pre["medians"]), np.asarray(pre["means"]),
                                np.asarray(pre["scales"]))
    model = _architecture(checkpoint["architecture"], len(feature_names),
                          checkpoint.get("model_parameters", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device("cpu"))
    model.eval()
    return TemporalModelBundle(
        model=model, preprocessor=preprocessor, feature_names=feature_names,
        window_hours=int(checkpoint["window_hours"]),
        horizon_hours=int(checkpoint["horizon_hours"]),
        threshold=float(checkpoint["threshold"]),
        persistence_hours=int(checkpoint["persistence_hours"]),
        source_type=str(checkpoint["source_type"]),
        metadata=dict(checkpoint.get("metadata", {})),
        architecture=str(checkpoint["architecture"]), device="cpu",
        calibrator=checkpoint.get("calibrator"),
    )


def _transform_window(raw: np.ndarray, preprocessor: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Causally impute one inference window, matching per-window training."""
    finite = np.isfinite(raw)
    filled = raw.copy()
    for hour in range(len(filled)):
        missing = ~finite[hour]
        if hour:
            filled[hour, missing] = filled[hour - 1, missing]
        leading = ~np.isfinite(filled[hour])
        filled[hour, leading] = preprocessor.medians[leading]
    scaled = (filled - preprocessor.means) / preprocessor.scales
    return scaled.astype(np.float32), finite


def _input_at_hour(bundle: TemporalModelBundle, raw_history: np.ndarray,
                    end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    begin = max(0, end - bundle.window_hours + 1)
    segment = raw_history[begin:end + 1]
    x, observed = _transform_window(segment, bundle.preprocessor)
    pad = bundle.window_hours - len(segment)
    if pad:
        x = np.pad(x, ((pad, 0), (0, 0)), constant_values=0)
        observed = np.pad(observed, ((pad, 0), (0, 0)), constant_values=False)
    valid = np.r_[np.zeros(pad, dtype=bool), np.ones(bundle.window_hours - pad, dtype=bool)]
    values_t = torch.as_tensor(x[None], dtype=torch.float32)
    observed_t = torch.as_tensor(observed[None], dtype=torch.bool)
    valid_t = torch.as_tensor(valid[None], dtype=torch.bool)
    return values_t, observed_t, valid_t, observed


def predict_temporal_hour(bundle: TemporalModelBundle, history: np.ndarray,
                          explain: bool = True) -> dict[str, Any]:
    """Predict current published-label risk from past/current values only.

    ``history`` is an ``(hours, features)`` matrix in canonical PhysioNet order;
    NaN denotes a missing measurement. Explanations use integrated gradients
    from the training-normalized zero baseline while holding the original
    observation mask and valid-time mask fixed. Values-only models ignore the
    observation mask; values-plus-mask models hold it fixed along the path.
    """
    raw = np.asarray(history, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(bundle.feature_names):
        raise ValueError(f"history must have shape (hours, {len(bundle.feature_names)})")
    if not len(raw):
        raise ValueError("history must contain at least one hour")
    if np.isinf(raw).any():
        raise ValueError("history cannot contain infinity; use NaN for missing readings")

    def evaluate(end: int, with_explanation: bool = False):
        x, observed, valid, obs_window = _input_at_hour(bundle, raw, end)
        with torch.no_grad():
            logit = bundle.model(x, observed, valid)
            probability = float(torch.sigmoid(logit)[0].item())
        attrs = None
        mask_attrs = None
        explanation = None
        if with_explanation:
            attrs_t = integrated_gradients(bundle.model, x, observed, valid, steps=32)
            attrs = attrs_t[0].detach().cpu().numpy()
            with torch.no_grad():
                baseline_logit = float(bundle.model(torch.zeros_like(x), observed, valid)[0].item())
            input_logit = float(logit[0].item())
            attributed_difference = float(attrs_t.sum().item())
            explanation = {
                "logit_baseline": baseline_logit,
                "input_logit": input_logit,
                "completeness_residual": attributed_difference - (input_logit - baseline_logit),
            }
            if getattr(bundle.model, "use_observation_mask", False):
                mask_attrs_t = missingness_integrated_gradients(
                    bundle.model, x, observed, valid, steps=32
                )
                mask_attrs = mask_attrs_t[0].detach().cpu().numpy()
                with torch.no_grad():
                    mask_baseline_logit = float(
                        bundle.model(x, torch.zeros_like(observed), valid)[0].item()
                    )
                explanation.update({
                    "missingness_logit_baseline": mask_baseline_logit,
                    "missingness_logit_input": input_logit,
                    "missingness_completeness_residual": float(mask_attrs_t.sum().item()) -
                        (input_logit - mask_baseline_logit),
                })
        return probability, attrs, mask_attrs, obs_window, explanation

    raw_probability, attrs, mask_attrs, obs_window, explanation = evaluate(len(raw) - 1, explain)
    probability = (float(bundle.calibrator.predict([raw_probability])[0])
                   if bundle.calibrator is not None else raw_probability)
    needed = min(bundle.persistence_hours, len(raw))
    prior_scores = [evaluate(hour)[0] for hour in range(len(raw) - needed, len(raw))]
    if bundle.calibrator is not None:
        prior_scores = [float(bundle.calibrator.predict([score])[0]) for score in prior_scores]
    persistent = (needed == bundle.persistence_hours and
                  all(score >= bundle.threshold for score in prior_scores))
    contributions: list[dict[str, Any]] = []
    feature_contributions: list[dict[str, Any]] = []
    missingness_contributions: list[dict[str, Any]] = []
    missingness_feature_contributions: list[dict[str, Any]] = []
    if attrs is not None:
        pad = bundle.window_hours - min(bundle.window_hours, len(raw))
        for local_hour in range(bundle.window_hours):
            if local_hour < pad:
                continue
            absolute_hour = len(raw) - min(bundle.window_hours, len(raw)) + local_hour - pad
            for feature_index, feature in enumerate(bundle.feature_names):
                contributions.append({
                    "feature": feature,
                    "hour_offset": int(absolute_hour - (len(raw) - 1)),
                    "lag": int(absolute_hour - (len(raw) - 1)),
                    "contribution": float(attrs[local_hour, feature_index]),
                    "observed": bool(obs_window[local_hour, feature_index]),
                })
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        signed_totals: dict[str, float] = {feature: 0.0 for feature in bundle.feature_names}
        for item in contributions:
            signed_totals[item["feature"]] += item["contribution"]
        feature_contributions = [
            {"feature": feature, "contribution": contribution}
            for feature, contribution in signed_totals.items()
        ]
        feature_contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        if mask_attrs is not None:
            mask_totals: dict[str, float] = {feature: 0.0 for feature in bundle.feature_names}
            for local_hour in range(pad, bundle.window_hours):
                absolute_hour = len(raw) - min(bundle.window_hours, len(raw)) + local_hour - pad
                for feature_index, feature in enumerate(bundle.feature_names):
                    contribution = float(mask_attrs[local_hour, feature_index])
                    missingness_contributions.append({
                        "feature": feature,
                        "hour_offset": int(absolute_hour - (len(raw) - 1)),
                        "lag": int(absolute_hour - (len(raw) - 1)),
                        "contribution": contribution,
                        "observed": bool(obs_window[local_hour, feature_index]),
                    })
                    mask_totals[feature] += contribution
            missingness_contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
            missingness_feature_contributions = [
                {"feature": feature, "contribution": contribution}
                for feature, contribution in mask_totals.items()
            ]
            missingness_feature_contributions.sort(
                key=lambda item: abs(item["contribution"]), reverse=True
            )
    return {
        "research": True,
        "risk_probability": probability,
        "raw_probability": raw_probability,
        "calibrated_probability": probability,
        "calibrated": bundle.calibrator is not None,
        "above_threshold": probability >= bundle.threshold,
        "alert": persistent,
        "persistence_met": persistent,
        "persistence_hours": bundle.persistence_hours,
        "horizon_hours": bundle.horizon_hours,
        "target_definition": "Published SepsisLabel at current hour plus additional horizon; PhysioNet labels already lead recorded clinical onset by 6 hours.",
        "as_of_hour": len(raw) - 1,
        "threshold": bundle.threshold,
        "feature_time_contributions": contributions,
        "temporal_contributions": contributions,
        "feature_contributions": feature_contributions,
        "missingness_feature_time_contributions": missingness_contributions,
        "missingness_feature_contributions": missingness_feature_contributions,
        "missingness_contribution_method": ("conditional_mask_integrated_gradients"
                                            if mask_attrs is not None else None),
        "missingness_contribution_baseline": (
            "observation mask zero; current values fixed; valid-time padding fixed"
            if mask_attrs is not None else None
        ),
        "missingness_contribution_target": "raw_model_logit",
        "missingness_contribution_scale": "additive_logit_difference",
        "missingness_available": mask_attrs is not None,
        "contribution_method": "integrated_gradients",
        "explanation_method": "integrated_gradients",
        "contribution_scale": "additive_log_odds_difference",
        "contribution_target": "raw_model_logit",
        "probability_explanation": ("risk_probability is calibrated; integrated-gradients contributions explain the raw model logit, not the calibrator."
                                    if bundle.calibrator is not None else
                                    "risk_probability is the raw model probability; integrated-gradients contributions explain the raw model logit."),
        "contribution_baseline": ("zero in training-normalized value space (training feature means); valid-time mask fixed"
                                  if not getattr(bundle.model, "use_observation_mask", True)
                                  else "zero in training-normalized value space (training feature means); observation and valid-time masks fixed"),
        **(explanation or {}),
        "architecture": bundle.architecture,
        "model_metadata": bundle.metadata,
        "source_type": bundle.source_type,
    }
