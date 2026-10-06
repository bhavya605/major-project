"""Model-appropriate SHAP explanations for trained classical baselines."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _class_one(values: Any, expected: Any, n_rows: int, n_features: int) -> tuple[np.ndarray, float]:
    """Normalize SHAP's legacy and current binary-class output conventions."""
    if hasattr(values, "values"):
        explanation = values
        values = explanation.values
        if expected is None:
            expected = explanation.base_values
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        # Current SHAP generally uses rows x features x classes.
        if array.shape[:2] == (n_rows, n_features):
            array = array[:, :, 1]
        elif array.shape[1:] == (n_rows, n_features):
            array = array[1]
        else:
            raise ValueError(f"unsupported SHAP output shape {array.shape}")
    if array.shape != (n_rows, n_features):
        raise ValueError(f"SHAP values have shape {array.shape}, expected {(n_rows, n_features)}")
    expected_array = np.asarray(expected, dtype=float)
    if expected_array.ndim == 0:
        base = float(expected_array)
    elif expected_array.size == 1:
        base = float(expected_array.reshape(-1)[0])
    else:
        base = float(expected_array.reshape(-1)[1])
    return array, base


def explain_baseline(model: Any, x: np.ndarray, background: np.ndarray,
                     feature_names: Sequence[str], *, method: str = "auto") -> dict[str, Any]:
    """Explain positive-class outputs with SHAP and a caller-supplied training background.

    Logistic regression uses a linear explainer in log-odds space. Tree estimators
    use TreeExplainer. The returned additivity residual checks each output against
    SHAP's expected value in the explainer's declared score space.
    """
    samples = np.asarray(x, dtype=float)
    reference = np.asarray(background, dtype=float)
    names = list(feature_names)
    if samples.ndim != 2 or reference.ndim != 2 or samples.shape[1] != reference.shape[1]:
        raise ValueError("x and background must be 2D matrices with matching feature counts")
    if samples.shape[1] != len(names) or not np.isfinite(samples).all() or not np.isfinite(reference).all():
        raise ValueError("feature names must match finite x and background columns")
    if method not in {"auto", "linear", "tree"}:
        raise ValueError("method must be auto, linear, or tree")
    if method == "auto":
        method = "linear" if hasattr(model, "coef_") else "tree"
    if method == "linear" and not hasattr(model, "coef_"):
        raise ValueError("linear SHAP requires an estimator with coef_")
    if method == "tree" and hasattr(model, "get_booster"):
        # SHAP 0.49 does not parse XGBoost 3.x's vector-valued JSON base_score.
        # XGBoost's pred_contribs is its native TreeSHAP implementation and
        # returns exact additive contributions in raw-margin (log-odds) space.
        import xgboost as xgb
        native = np.asarray(model.get_booster().predict(
            xgb.DMatrix(samples), pred_contribs=True, strict_shape=True), dtype=float)
        if native.ndim != 3 or native.shape[0] != len(samples) or native.shape[1] != 1:
            raise ValueError(f"unsupported XGBoost contribution shape {native.shape}")
        values = native[:, 0, :-1]
        expected = float(native[0, 0, -1])
        raw_scores = np.asarray(model.get_booster().predict(
            xgb.DMatrix(samples), output_margin=True, strict_shape=True), dtype=float).reshape(len(samples), -1)[:, 0]
        residual = raw_scores - (expected + values.sum(axis=1))
        return {
            "research": True,
            "method": "xgboost_native_treeshap",
            "score_space": "log_odds",
            "background_source": "xgboost_native_cover_weights",
            "expected_value": expected,
            "feature_names": names,
            "values": values,
            "additivity_residual": residual,
            "additivity_max_abs_residual": float(np.max(np.abs(residual))) if len(residual) else 0.0,
            "n_explained": int(len(samples)),
            "n_background": int(len(reference)),
        }
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP explanations require the optional shap package") from exc
    if method == "linear":
        explainer = shap.LinearExplainer(model, reference)
        explanation = explainer(samples)
        values, expected = _class_one(explanation, None, len(samples), samples.shape[1])
        score_space = "log_odds"
        raw_scores = np.asarray(model.decision_function(samples), dtype=float)
    else:
        explainer = shap.TreeExplainer(model, data=reference,
                                       feature_perturbation="interventional",
                                       model_output="probability")
        explanation = explainer(samples)
        values, expected = _class_one(explanation, getattr(explainer, "expected_value", None),
                                      len(samples), samples.shape[1])
        score_space = "probability"
        raw_scores = np.asarray(model.predict_proba(samples)[:, 1], dtype=float)
    residual = raw_scores - (expected + values.sum(axis=1))
    return {
        "research": True,
        "method": f"shap_{method}",
        "score_space": score_space,
        "background_source": "caller_supplied_training_data",
        "expected_value": expected,
        "feature_names": names,
        "values": values,
        "additivity_residual": residual,
        "additivity_max_abs_residual": float(np.max(np.abs(residual))) if len(residual) else 0.0,
        "n_explained": int(len(samples)),
        "n_background": int(len(reference)),
    }
