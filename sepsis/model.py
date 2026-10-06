"""Patient-level logistic baseline and causal hourly inference helpers.

Risk scores predict the published sepsis label at ``hour + horizon``. The
published PhysioNet label is shifted ahead of clinical onset, so this module
does not interpret a score as clinical onset probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _as_3d(values: np.ndarray, observed_mask: np.ndarray | None = None,
           valid_time_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3:
        raise ValueError("values must have shape (hours, features) or (samples, hours, features)")
    obs = np.isfinite(x) if observed_mask is None else np.asarray(observed_mask, dtype=bool).copy()
    if obs.ndim == 2:
        obs = obs[None, ...]
    if obs.shape != x.shape:
        raise ValueError("observed_mask must match values")
    obs &= np.isfinite(x)
    valid = np.ones(x.shape[:2], dtype=bool) if valid_time_mask is None else np.asarray(valid_time_mask, dtype=bool)
    if valid.ndim == 1:
        valid = valid[None, ...]
    if valid.shape != x.shape[:2]:
        raise ValueError("valid_time_mask must match sample and hour dimensions")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), obs, valid


def summarize_windows(values: np.ndarray, observed_mask: np.ndarray | None = None,
                      valid_time_mask: np.ndarray | None = None,
                      feature_names: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Summarize causal windows using observed values only.

    Per channel, return last observation, observed mean, observed linear slope,
    and observed fraction. No values from after the end of a window are used.
    """
    x, obs, valid = _as_3d(values, observed_mask, valid_time_mask)
    obs &= valid[:, :, None]
    n, width, n_features = x.shape
    counts = obs.sum(axis=1)
    denom = np.maximum(counts, 1)
    mean = (x * obs).sum(axis=1) / denom
    times = np.broadcast_to(np.arange(width, dtype=float)[None, :, None], (n, width, n_features))
    tbar = (times * obs).sum(axis=1) / denom
    xbar = mean
    centered_t = (times - tbar[:, None, :]) * obs
    centered_x = (x - xbar[:, None, :]) * obs
    slope_denom = (centered_t ** 2).sum(axis=1)
    slope = (centered_t * centered_x).sum(axis=1) / np.maximum(slope_denom, 1e-12)
    # Locate the last observed value independently for each feature.
    reverse_index = np.argmax(obs[:, ::-1, :], axis=1)
    last_index = width - 1 - reverse_index
    last = np.take_along_axis(x, last_index[:, None, :], axis=1)[:, 0, :]
    last = np.where(counts > 0, last, 0.0)
    valid_count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    frac = counts / valid_count
    variance = ((x - mean[:, None, :]) ** 2 * obs).sum(axis=1) / denom
    std = np.sqrt(variance)
    min_values = np.min(np.where(obs, x, np.inf), axis=1)
    max_values = np.max(np.where(obs, x, -np.inf), axis=1)
    min_values = np.where(counts > 0, min_values, 0.0)
    max_values = np.where(counts > 0, max_values, 0.0)
    out = np.concatenate((last, mean, std, min_values, max_values, slope,
                          counts.astype(float), frac), axis=1)
    base = list(feature_names or [f"feature_{i}" for i in range(n_features)])
    names = ([f"{name}__last" for name in base] + [f"{name}__mean" for name in base]
             + [f"{name}__std" for name in base]
             + [f"{name}__min" for name in base] + [f"{name}__max" for name in base]
             + [f"{name}__slope" for name in base]
             + [f"{name}__observed_count" for name in base]
             + [f"{name}__observed_fraction" for name in base])
    return out, names


@dataclass
class ModelBundle:
    model: Any
    preprocessor: Any
    feature_names: list[str]
    window_hours: int
    horizon_hours: int
    threshold: float
    persistence_hours: int = 2
    source_type: str = "physionet"
    metadata: dict[str, Any] | None = None
    background_values: np.ndarray | None = None
    calibrator: Any = None


def load_model(path: str) -> ModelBundle:
    """Load a persisted training bundle."""
    import joblib
    obj = joblib.load(path)
    if isinstance(obj, ModelBundle):
        return obj
    if isinstance(obj, dict) and {"model", "preprocessor"} <= obj.keys():
        return ModelBundle(**obj)
    raise ValueError("File does not contain a supported sepsis ModelBundle")


def _preprocess_history(bundle: ModelBundle, history: np.ndarray,
                        observed_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(history, dtype=float)
    if raw.ndim != 2:
        raise ValueError("history must have shape (hours, features)")
    mask = np.isfinite(raw) if observed_mask is None else np.asarray(observed_mask, dtype=bool) & np.isfinite(raw)
    if mask.shape != raw.shape:
        raise ValueError("observed_mask must match history")
    # A direct preprocessor hook keeps inference compatible with the training
    # pipeline without requiring labels or a patient identifier.
    if hasattr(bundle.preprocessor, "transform_arrays"):
        transformed, transformed_mask = bundle.preprocessor.transform_arrays(raw, mask)
        return np.asarray(transformed, dtype=float), np.asarray(transformed_mask, dtype=bool)
    from types import SimpleNamespace
    patient = SimpleNamespace(patient_id="inference", values=np.where(mask, raw, np.nan), labels=np.zeros(len(raw)))
    transformed, transformed_mask = bundle.preprocessor.transform(patient)
    return np.asarray(transformed, dtype=float), np.asarray(transformed_mask, dtype=bool)


def predict_hour(bundle: ModelBundle, history: np.ndarray,
                 observed_mask: np.ndarray | None = None, *,
                 include_explanation: bool = False) -> dict[str, Any]:
    """Forecast from history up to the current hour and report model-grounded contributions."""
    raw_history = np.asarray(history, dtype=float)
    if raw_history.ndim != 2 or not len(raw_history):
        raise ValueError("history must contain at least one hour")
    raw_mask = np.isfinite(raw_history) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    if raw_mask.shape != raw_history.shape:
        raise ValueError("observed_mask must match history")
    def score(end: int) -> tuple[float, float, np.ndarray, list[str]]:
        begin = max(0, end - bundle.window_hours + 1)
        # Match training exactly: each anchor window is forward-filled and
        # leading gaps are median-filled within that window.
        xs, ms = _preprocess_history(bundle, raw_history[begin:end + 1], raw_mask[begin:end + 1])
        # Left-pad with invalid time slots, preserving causal aggregation.
        pad = max(0, bundle.window_hours - len(xs))
        xs = np.pad(xs, ((pad, 0), (0, 0)))
        ms = np.pad(ms, ((pad, 0), (0, 0)))
        valid = np.r_[np.zeros(pad, dtype=bool), np.ones(bundle.window_hours - pad, dtype=bool)]
        base_names = getattr(bundle.preprocessor, "feature_names", None)
        if base_names is None and len(bundle.feature_names) == 8 * xs.shape[1]:
            base_names = [name.removesuffix("__last") for name in bundle.feature_names[:xs.shape[1]]]
        x, names = summarize_windows(xs, ms, valid, base_names)
        raw_probability = float(bundle.model.predict_proba(x)[0, 1])
        probability = (float(bundle.calibrator.predict([raw_probability])[0])
                       if bundle.calibrator is not None else raw_probability)
        return probability, raw_probability, x[0], names
    history_length = len(raw_history)
    probability, raw_probability, row, names = score(history_length - 1)
    recent_count = min(bundle.persistence_hours, history_length)
    recent = [score(i)[0] for i in range(history_length - recent_count, history_length)]
    persistence_met = len(recent) == bundle.persistence_hours and all(p >= bundle.threshold for p in recent)
    if hasattr(bundle.model, "coef_") and hasattr(bundle.model, "intercept_"):
        coef = np.asarray(bundle.model.coef_)[0]
        intercept = float(np.asarray(bundle.model.intercept_).reshape(-1)[0])
        contributions = [{"feature": name, "contribution": float(value * weight)}
                         for name, value, weight in zip(names, row, coef)]
        contribution_scale = "additive_log_odds"
    else:
        # A model-agnostic occlusion attribution: compare each feature's actual
        # input with a neutral zero (training-mean) value in summary space.
        neutral = np.zeros_like(row)
        neutral_probability = float(bundle.model.predict_proba(neutral[None, :])[0, 1])
        perturbed = np.tile(row, (len(row), 1))
        np.fill_diagonal(perturbed, 0.0)
        changed = bundle.model.predict_proba(perturbed)[:, 1]
        contributions = [{"feature": name, "contribution": float(raw_probability - change),
                          "method": "single_feature_zero_occlusion"}
                         for name, change in zip(names, changed)]
        intercept = None
        contribution_scale = "probability_change_non_additive"
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    response = {
        "risk_probability": probability,
        "raw_risk_probability": raw_probability,
        "above_threshold": probability >= bundle.threshold,
        "alert": persistence_met,
        "persistence_met": persistence_met,
        "persistence_hours": bundle.persistence_hours,
        "horizon_hours": bundle.horizon_hours,
        "as_of_hour": history_length - 1,
        "threshold": bundle.threshold,
        "feature_contributions": contributions,
        "contribution_scale": contribution_scale,
        "contribution_target": "raw_model_output",
        "calibrated": bundle.calibrator is not None,
        "log_odds_intercept": intercept,
        "neutral_probability": neutral_probability if contribution_scale != "additive_log_odds" else None,
        "source_type": bundle.source_type,
        "research": True,
        "model_version": (bundle.metadata or {}).get("bundle_version"),
    }
    if include_explanation and bundle.background_values is not None:
        from .explain import explain_baseline
        explanation = explain_baseline(bundle.model, row[None, :], bundle.background_values, names)
        response["shap_explanation"] = {
            "method": explanation["method"], "score_space": explanation["score_space"],
            "target": "raw_model_output_before_calibration",
            "expected_value": explanation["expected_value"],
            "additivity_residual": float(explanation["additivity_residual"][0]),
            "contributions": [{"feature": name, "contribution": float(value)}
                              for name, value in zip(names, explanation["values"][0])],
        }
    return response
