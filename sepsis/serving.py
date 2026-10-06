"""Shared serving adapter for trusted baseline and temporal model artifacts."""
from pathlib import Path
import numpy as np
from sepsis.model import load_model, predict_hour


def json_safe(value):
    """Represent unavailable non-finite research statistics as JSON null."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def load_bundle(path):
    if Path(path).suffix == ".pt":
        from sepsis.temporal_model import load_temporal_model
        return load_temporal_model(path)
    return load_model(path)


def predict(bundle, history, *, explain=True):
    raw = np.asarray(history, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 40 or len(raw) == 0:
        raise ValueError("History must contain one or more hourly rows with 40 predictors")
    if np.isinf(raw).any():
        raise ValueError("Observed predictors must be finite; use NaN for missing readings")
    if hasattr(bundle, "architecture"):
        from sepsis.temporal_model import predict_temporal_hour
        output = predict_temporal_hour(bundle, raw, explain=explain)
    else:
        output = predict_hour(bundle, raw, include_explanation=explain)
    if not np.isfinite(output["risk_probability"]):
        raise ValueError("Model returned a non-finite probability")
    return json_safe(output)


def model_metadata(bundle):
    return json_safe({
        "source_type": bundle.source_type,
        "window_hours": bundle.window_hours,
        "horizon_hours": bundle.horizon_hours,
        "threshold": bundle.threshold,
        "persistence_hours": bundle.persistence_hours,
        "model_family": getattr(bundle, "architecture", type(bundle.model).__name__),
        "metadata": bundle.metadata,
    })
