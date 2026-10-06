"""Train research-only patient-held-out classical sepsis baselines."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)

from .data import Preprocessor, Patient, PatientSplit, load_patients, split_patients
from .evaluation import evaluate_predictions, fit_calibrator, select_operating_threshold
from .model import ModelBundle
from .tabular import TabularBatch, build_tabular_batch
from .schema import FEATURE_COLUMNS


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | None]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, p)) if len(y) and y.sum() else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
    }


def _select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Select validation F1 operating point with deterministic high-threshold tie break."""
    candidates = np.unique(np.r_[0.0, p, 1.0])
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        pred = p >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f1 = (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def _batch_features(batch: TabularBatch, names: list[str]) -> tuple[np.ndarray, list[str]]:
    return batch.features, names


def _split_from_manifest(patients: list[Patient], path: Path, random_state: int) -> PatientSplit:
    by_id = {p.patient_id: p for p in patients}
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = raw.get("patient_ids", raw)
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test patient_ids")
    all_ids = [pid for name in ("train", "validation", "test") for pid in groups[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split manifest assigns a patient more than once")
    if set(all_ids) != set(by_id):
        missing, extra = set(by_id) - set(all_ids), set(all_ids) - set(by_id)
        raise ValueError(f"split manifest patient IDs do not match data (missing={len(missing)}, unknown={len(extra)})")
    return PatientSplit(*(tuple(by_id[pid] for pid in groups[name])
                          for name in ("train", "validation", "test")))


def _estimator(kind: str, random_state: int):
    if kind == "lr":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=random_state)
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("--model xgboost requires the optional xgboost package") from exc
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             eval_metric="logloss", base_score=0.5, n_jobs=-1,
                             random_state=random_state)
    raise ValueError(f"unsupported estimator: {kind}")


def train(data_dir: str | Path, output: str | Path, *, window: int = 12,
          horizon: int = 0, random_state: int = 42, persistence_hours: int = 2,
          estimator: str = "lr", source_type: str,
          evaluate_test: bool = False,
          split_manifest: str | Path | None = None,
          calibration: str = "none",
          min_sensitivity: float | None = None,
          max_false_episodes_per_hour: float | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    patients = load_patients(data_dir)
    manifest_path = Path(split_manifest) if split_manifest is not None else None
    if manifest_path and manifest_path.exists():
        split = _split_from_manifest(patients, manifest_path, random_state)
    else:
        # Plan's primary partition: 70/15/15 by patient.
        split = split_patients(patients, test_size=0.15, validation_size=0.15,
                               random_state=random_state)
    if not split.train or not split.validation:
        raise ValueError("training and validation partitions must contain patients")
    if manifest_path and not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "random_state": int(random_state), "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "patient_ids": {"train": [p.patient_id for p in split.train],
                            "validation": [p.patient_id for p in split.validation],
                            "test": [p.patient_id for p in split.test]},
        }, indent=2), encoding="utf-8")

    preprocessor = Preprocessor.fit(split.train)
    train_batch, feature_names = build_tabular_batch(split.train, preprocessor, window, horizon,
                                                    FEATURE_COLUMNS)
    if not len(train_batch):
        raise ValueError("training and validation partitions must have eligible windows")
    x_train, _ = _batch_features(train_batch, feature_names)
    y_train = train_batch.targets
    if len(np.unique(y_train)) < 2:
        raise ValueError("training split must include positive and negative horizon labels")
    model = _estimator(estimator, random_state)
    if estimator == "xgboost":
        # Balance only from training targets, keeping the held-out partitions untouched.
        positives = max(int(y_train.sum()), 1)
        model.set_params(scale_pos_weight=(len(y_train) - positives) / positives)
    fit_started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    if calibration not in {"none", "platt", "isotonic"}:
        raise ValueError("calibration must be none, platt, or isotonic")
    calibrator = None
    calibration_patient_ids: list[str] = []
    operation_patients = split.validation
    if calibration != "none":
        # Keep calibrator fitting and operating-point selection on distinct
        # patients within validation. Never use test predictions for either.
        cal_split = split_patients(split.validation, test_size=0.0, validation_size=0.5,
                                   random_state=random_state + 1)
        calibration_patients, operation_patients = cal_split.train, cal_split.validation
        if not calibration_patients or not operation_patients:
            raise ValueError("validation patients are too few to separate calibration and threshold selection")
        calibration_batch, _ = build_tabular_batch(calibration_patients, preprocessor, window,
                                                   horizon, FEATURE_COLUMNS)
        calibration_patient_ids = [p.patient_id for p in calibration_patients]
        if not len(calibration_batch):
            raise ValueError("calibration validation subset has no eligible windows")
        p_cal = model.predict_proba(_batch_features(calibration_batch, feature_names)[0])[:, 1]
        calibrator = fit_calibrator(calibration_batch.targets, p_cal, method=calibration)
    val_batch, _ = build_tabular_batch(operation_patients, preprocessor, window, horizon,
                                       FEATURE_COLUMNS)
    if not len(val_batch):
        raise ValueError("operating-point validation subset has no eligible windows")
    x_val, _ = _batch_features(val_batch, feature_names)
    p_val_raw = model.predict_proba(x_val)[:, 1]
    p_val = calibrator.predict(p_val_raw) if calibrator is not None else p_val_raw
    threshold_selection = select_operating_threshold(
        val_batch.targets, p_val, val_batch.patient_ids, val_batch.hours,
        persistence_hours=persistence_hours, min_sensitivity=min_sensitivity,
        max_false_episodes_per_hour=max_false_episodes_per_hour)
    if not threshold_selection["feasible"]:
        raise ValueError("validation threshold constraints are infeasible; no model bundle was written: " +
                         threshold_selection["message"])
    threshold = float(threshold_selection["threshold"])
    val_metrics = _metrics(val_batch.targets, p_val, threshold)
    val_alert_metrics = evaluate_predictions(
        val_batch.targets, p_val, val_batch.patient_ids, val_batch.hours,
        threshold=threshold, persistence_hours=persistence_hours)

    test_metrics = None
    test_alert_metrics = None
    if evaluate_test:
        test_batch = (build_tabular_batch(split.test, preprocessor, window, horizon,
                                          FEATURE_COLUMNS)[0] if split.test else None)
        if test_batch is not None and len(test_batch):
            x_test, _ = _batch_features(test_batch, feature_names)
            p_test_raw = model.predict_proba(x_test)[:, 1]
            p_test = calibrator.predict(p_test_raw) if calibrator is not None else p_test_raw
            test_metrics = _metrics(test_batch.targets, p_test, threshold)
            test_alert_metrics = evaluate_predictions(
                test_batch.targets, p_test, test_batch.patient_ids, test_batch.hours,
                threshold=threshold, persistence_hours=persistence_hours)

    try:
        import xgboost
        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = None
    ids = {name: [p.patient_id for p in getattr(split, name)]
           for name in ("train", "validation", "test")}
    background_rng = np.random.default_rng(random_state)
    background_ix = background_rng.choice(len(x_train), size=min(32, len(x_train)), replace=False)
    training_background = x_train[background_ix].astype(np.float32, copy=True)
    metadata: dict[str, Any] = {
        "research": True,
        "runtime_seconds_before_serialization": time.perf_counter() - started,
        "estimator_fit_seconds": fit_seconds,
        "training_source": str(Path(data_dir).resolve()), "source_type": source_type,
        "target": "published SepsisLabel at anchor hour + additional published-label horizon",
        "horizon_hours": int(horizon),
        "label_note": "PhysioNet 2019 SepsisLabel begins six hours before recorded clinical onset; horizon_hours adds a shift of the published label and is not an onset forecast.",
        "random_state": int(random_state), "window_hours": int(window),
        "split_patient_ids": ids,
        "split_patient_counts": {name: len(ids[name]) for name in ids},
        "split_window_counts": {"train": len(train_batch), "validation": len(val_batch),
                                "test": None},
        "validation_metrics": val_metrics, "validation_alert_metrics": val_alert_metrics,
        "test_metrics": test_metrics, "test_alert_metrics": test_alert_metrics,
        "test_evaluated": bool(evaluate_test), "threshold_selected_on": "validation alert metrics",
        "threshold_selection": threshold_selection,
        "threshold_constraints": {"min_sensitivity": min_sensitivity,
                                   "max_false_episodes_per_hour": max_false_episodes_per_hour},
        "calibration": calibration,
        "calibration_patient_ids": calibration_patient_ids,
        "operating_point_patient_ids": [p.patient_id for p in operation_patients],
        "estimator": estimator, "estimator_config": model.get_params(deep=False),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost_version},
        "preprocessing": "training-patient medians, means, and scales; causal forward fill within each input window",
    }
    bundle = ModelBundle(model=model, preprocessor=preprocessor, feature_names=feature_names,
                         window_hours=window, horizon_hours=horizon, threshold=threshold,
                         persistence_hours=persistence_hours, source_type=source_type,
                         metadata=metadata, background_values=training_background,
                         calibrator=calibrator)
    import joblib
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_model = io.BytesIO()
    joblib.dump(model, serialized_model)
    metadata["estimator_sha256"] = hashlib.sha256(serialized_model.getvalue()).hexdigest()
    metadata["bundle_version"] = 1
    joblib.dump(bundle, output_path)
    report_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    report_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing patient .psv files")
    parser.add_argument("--output", required=True, help="Path for the serialized model bundle")
    parser.add_argument("--model", choices=("lr", "rf", "xgboost"), default="lr")
    parser.add_argument("--source-type", choices=("physionet", "synthetic_demo"), required=True,
                        help="Explicit provenance label saved in the model bundle")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=0,
                        help="Additional hours ahead for the published-label target; 0 is official-label comparison")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--persistence-hours", type=int, default=2)
    parser.add_argument("--split-manifest", help="Read an existing patient split or write a new one at this path")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Explicitly evaluate the locked test partition once after model selection")
    parser.add_argument("--calibration", choices=("none", "platt", "isotonic"), default="none",
                        help="Optional calibration fit on validation patients separate from threshold selection")
    parser.add_argument("--min-sensitivity", type=float,
                        help="Optional validation operating-point sensitivity constraint")
    parser.add_argument("--max-false-episodes-per-hour", type=float,
                        help="Optional validation false alert episodes per negative-label hour constraint")
    args = parser.parse_args()
    result = train(args.data, args.output, window=args.window, horizon=args.horizon,
                   random_state=args.random_state, persistence_hours=args.persistence_hours,
                   estimator=args.model, source_type=args.source_type,
                   evaluate_test=args.evaluate_test, split_manifest=args.split_manifest,
                   calibration=args.calibration, min_sensitivity=args.min_sensitivity,
                   max_false_episodes_per_hour=args.max_false_episodes_per_hour)
    print(json.dumps({"model": str(Path(args.output).resolve()), "research": True,
                      "source_type": result["source_type"],
                      "split_patient_counts": result["split_patient_counts"],
                      "validation_metrics": result["validation_metrics"],
                      "test_metrics": result["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
