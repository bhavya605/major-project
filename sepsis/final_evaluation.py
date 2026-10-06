"""Freeze validation-only model selection, then evaluate a locked test cohort.

All outputs from this module are research results against the published
SepsisLabel timeline, not clinical performance estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .data import Patient, PatientSplit, load_patients, split_patients
from .evaluation import calibration_bins, evaluate_predictions
from .model import ModelBundle, load_model
from .schema import FEATURE_COLUMNS
from .sequences import SequenceDataset
from .tabular import build_tabular_batch
from .train import _split_from_manifest


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _source_fingerprint(data_dir: Path) -> str:
    files = sorted(data_dir.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"no .psv patient files found in {data_dir}")
    files += sorted(p for p in data_dir.rglob("*manifest*.json") if p not in files)
    return _json_sha([(p.relative_to(data_dir).as_posix(), _sha(p)) for p in files])


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    paths = [Path(__file__), root / "requirements-lock.txt"]
    paths += [package / name for name in (
        "data.py", "evaluation.py", "model.py", "schema.py", "sequences.py",
        "tabular.py", "train.py", "temporal_model.py", "temporal_train.py")]
    return _json_sha([(p.name, _sha(p)) for p in paths if p.is_file()])


def _split(patients: list[Patient], split_manifest: str | Path | None,
           random_state: int) -> tuple[PatientSplit, str]:
    if split_manifest is not None and Path(split_manifest).exists():
        path = Path(split_manifest)
        split = _split_from_manifest(patients, path, random_state)
        return split, _sha(path)
    split = split_patients(patients, test_size=.15, validation_size=.15,
                           random_state=random_state)
    ids = {name: [p.patient_id for p in getattr(split, name)]
           for name in ("train", "validation", "test")}
    return split, _json_sha(ids)


def freeze_selection(bundle_path: str | Path, data_dir: str | Path,
                     selection_path: str | Path, *,
                     split_manifest: str | Path | None = None) -> dict[str, Any]:
    """Freeze the already selected bundle and validation metadata, without test labels.

    Patient files are fingerprinted as acquisition artifacts, but only patient
    identifiers and the validation partition are read for selection metadata.
    No model fitting or threshold selection takes place here.
    """
    bundle_path, data_dir = Path(bundle_path), Path(data_dir)
    try:
        bundle = load_model(str(bundle_path))
    except Exception:
        from .temporal_model import load_temporal_model
        bundle = load_temporal_model(bundle_path)
    metadata = bundle.metadata or {}
    source_type = metadata.get("source_type", bundle.source_type)
    ids_from_bundle = metadata.get("split_patient_ids", {})
    if not all(key in ids_from_bundle for key in ("train", "validation", "test")):
        raise ValueError("bundle must contain train/validation/test patient IDs")
    observed_ids = {name: list(ids_from_bundle[name]) for name in ("train", "validation", "test")}
    file_ids = [path.stem for path in data_dir.rglob("*.psv")]
    all_ids = [pid for group in observed_ids.values() for pid in group]
    if len(all_ids) != len(set(all_ids)) or len(file_ids) != len(set(file_ids)):
        raise ValueError("split patient IDs and source patient files must be unique")
    if set(file_ids) != {pid for values in observed_ids.values() for pid in values}:
        raise ValueError("data patient IDs do not match the selected bundle")
    if split_manifest is not None and Path(split_manifest).exists():
        manifest_path = Path(split_manifest)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = raw.get("patient_ids", raw)
        if set(groups) != {"train", "validation", "test"}:
            raise ValueError("split manifest must contain train, validation, and test patient_ids")
        manifest_ids = [pid for group in ("train", "validation", "test") for pid in groups[group]]
        if len(manifest_ids) != len(set(manifest_ids)):
            raise ValueError("split manifest assigns a patient more than once")
        if {k: set(groups[k]) for k in groups} != {k: set(v) for k, v in observed_ids.items()}:
            raise ValueError("split manifest membership does not match the selected bundle")
        split_sha = _sha(manifest_path)
    else:
        split_sha = _json_sha(observed_ids)
    if bool(metadata.get("test_evaluated", False)) or metadata.get("test_metrics") is not None:
        raise ValueError("bundle already contains test results; freeze a validation-only selection")
    cal_ids = set(metadata.get("calibration_patient_ids", []))
    op_ids = set(metadata.get("operating_point_patient_ids", []))
    validation_ids = set(observed_ids["validation"])
    if not cal_ids <= validation_ids or not op_ids <= validation_ids or cal_ids & op_ids:
        raise ValueError("calibration and threshold-selection patients must be disjoint validation subsets")
    if int(bundle.window_hours) < 1 or int(bundle.horizon_hours) < 0:
        raise ValueError("bundle has invalid target configuration")
    config = {
        "estimator": metadata.get("estimator", metadata.get("model_type")),
        "window_hours": int(bundle.window_hours), "horizon_hours": int(bundle.horizon_hours),
        "threshold": float(bundle.threshold), "persistence_hours": int(bundle.persistence_hours),
        "calibration": metadata.get("calibration", "none"),
        "calibration_patient_ids": metadata.get("calibration_patient_ids", []),
        "operating_point_patient_ids": metadata.get("operating_point_patient_ids", []),
        "target": metadata.get("target", "published SepsisLabel at anchor hour + additional published-label horizon"),
        "label_note": metadata.get("label_note", "PhysioNet SepsisLabel already starts six hours before recorded clinical onset; horizon_hours adds a published-label shift."),
    }
    artifact = {
        "format": "sepsis-frozen-selection-v1", "research": True,
        "source_type": source_type, "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": _sha(bundle_path), "data_dir": str(data_dir.resolve()),
        "data_acquisition_sha256": _source_fingerprint(data_dir),
        "evaluation_code_sha256": _code_fingerprint(),
        "split_sha256": split_sha, "split_patient_ids": observed_ids,
        "split_manifest_path": (str(Path(split_manifest).resolve())
                                if split_manifest is not None and Path(split_manifest).exists()
                                else None),
        "validation_patient_ids": observed_ids["validation"],
        "test_patient_ids": observed_ids["test"], "selected_config": config,
        "validation_metrics": metadata.get("validation_metrics"),
        "threshold_selection": metadata.get("threshold_selection"),
        "threshold_selected_on": metadata.get("threshold_selected_on", "validation"),
    }
    artifact["selection_sha256"] = _json_sha(artifact)
    path = Path(selection_path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("selection_sha256") == artifact["selection_sha256"]:
            return existing
        raise FileExistsError(f"selection lock already exists with different contents: {path}")
    _atomic_json(path, artifact)
    return artifact


def _verify_selection(selection: dict[str, Any], bundle_path: Path,
                      data_dir: Path, split_manifest: str | Path | None,
                      patients: list[Patient]) -> tuple[PatientSplit, ModelBundle]:
    if selection.get("format") != "sepsis-frozen-selection-v1":
        raise ValueError("unsupported frozen selection artifact")
    if selection.get("selection_sha256") != _json_sha({k: v for k, v in selection.items()
                                                        if k != "selection_sha256"}):
        raise ValueError("frozen selection artifact hash mismatch")
    if _sha(bundle_path) != selection.get("bundle_sha256"):
        raise ValueError("selected model bundle hash changed after freeze")
    if _source_fingerprint(data_dir) != selection.get("data_acquisition_sha256"):
        raise ValueError("data acquisition fingerprint changed after freeze")
    if _code_fingerprint() != selection.get("evaluation_code_sha256"):
        raise ValueError("evaluation code/dependency lock changed after freeze")
    try:
        bundle = load_model(str(bundle_path))
    except Exception:
        from .temporal_model import load_temporal_model
        bundle = load_temporal_model(bundle_path)
    if bundle.source_type != selection.get("source_type"):
        raise ValueError("bundle source type changed after freeze")
    cfg = selection["selected_config"]
    actual = (int(bundle.window_hours), int(bundle.horizon_hours), float(bundle.threshold),
              int(bundle.persistence_hours))
    frozen = (cfg["window_hours"], cfg["horizon_hours"], cfg["threshold"], cfg["persistence_hours"])
    if actual != tuple(frozen):
        raise ValueError("model target/threshold configuration changed after freeze")
    split, split_sha = _split(patients, split_manifest, int((bundle.metadata or {}).get("random_state", 42)))
    if split_sha != selection.get("split_sha256"):
        raise ValueError("split manifest hash changed after freeze")
    ids = {name: [p.patient_id for p in getattr(split, name)]
           for name in ("train", "validation", "test")}
    frozen_ids = selection.get("split_patient_ids", {})
    if {k: set(v) for k, v in ids.items()} != {k: set(v) for k, v in frozen_ids.items()}:
        raise ValueError("patient membership changed after freeze")
    if ids["test"] != selection.get("test_patient_ids"):
        raise ValueError("test patient ordering/membership changed after freeze")
    return split, bundle


def _verify_frozen_artifacts(selection: dict[str, Any], bundle_path: Path,
                             data_dir: Path) -> None:
    if selection.get("format") != "sepsis-frozen-selection-v1":
        raise ValueError("unsupported frozen selection artifact")
    if selection.get("selection_sha256") != _json_sha({k: v for k, v in selection.items()
                                                        if k != "selection_sha256"}):
        raise ValueError("frozen selection artifact hash mismatch")
    if _sha(bundle_path) != selection.get("bundle_sha256"):
        raise ValueError("selected model bundle hash changed after freeze")
    if _source_fingerprint(data_dir) != selection.get("data_acquisition_sha256"):
        raise ValueError("data acquisition fingerprint changed after freeze")
    manifest = selection.get("split_manifest_path")
    if manifest and _sha(Path(manifest)) != selection.get("split_sha256"):
        raise ValueError("split manifest hash changed after freeze")


def _predict(bundle: Any, patients: tuple[Patient, ...]):
    if not patients:
        raise ValueError("frozen test partition is empty")
    if isinstance(bundle, ModelBundle):
        batch, _ = build_tabular_batch(patients, bundle.preprocessor, bundle.window_hours,
                                       bundle.horizon_hours, FEATURE_COLUMNS)
        probabilities = bundle.model.predict_proba(batch.features)[:, 1]
        if bundle.calibrator is not None:
            probabilities = bundle.calibrator.predict(probabilities)
        return batch.targets.astype(np.int8), probabilities, batch.patient_ids, batch.hours
    from .temporal_model import TemporalModelBundle
    if isinstance(bundle, TemporalModelBundle):
        from .temporal_train import _predict as predict_temporal
        dataset = SequenceDataset(patients, bundle.preprocessor, bundle.window_hours,
                                  bundle.horizon_hours)
        probabilities = predict_temporal(bundle.model, dataset, batch_size=256)
        if bundle.calibrator is not None:
            probabilities = bundle.calibrator.predict(probabilities)
        return dataset.targets.astype(np.int8), probabilities, dataset.patient_ids, dataset.hours
    raise ValueError("unsupported bundle type")


def evaluate_test(selection_path: str | Path, *, bundle_path: str | Path | None = None,
                  data_dir: str | Path | None = None,
                  split_manifest: str | Path | None = None,
                  output_dir: str | Path | None = None,
                  bootstrap_replicates: int = 1000, seed: int = 42,
                  retry_failed: bool = False, n_bins: int = 10) -> dict[str, Any]:
    """Evaluate one frozen selection on test rows and persist the immutable result."""
    lock_path = Path(selection_path)
    selection = json.loads(lock_path.read_text(encoding="utf-8"))
    model_path = Path(bundle_path or selection["bundle_path"])
    data_path = Path(data_dir or selection["data_dir"])
    manifest_path = split_manifest or selection.get("split_manifest_path")
    out = Path(output_dir) if output_dir else lock_path.parent / (lock_path.stem + "_test")
    status_path = out / "status.json"
    report_path = out / "research_test_report.json"
    pred_path = out / "research_test_predictions.csv"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            for path, key in ((report_path, "report_sha256"), (pred_path, "predictions_sha256")):
                if not path.is_file() or _sha(path) != status.get(key):
                    raise ValueError("completed evaluation output hash mismatch")
            return json.loads(report_path.read_text(encoding="utf-8"))
        if not retry_failed:
            raise RuntimeError("evaluation lock is incomplete/failed; pass retry_failed=True to retry")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory exists without a matching evaluation lock")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(status_path, {"research": True, "status": "running",
                               "selection_sha256": selection.get("selection_sha256")})
    try:
        _verify_frozen_artifacts(selection, model_path, data_path)
        patients = load_patients(data_path)
        split, bundle = _verify_selection(selection, model_path, data_path, manifest_path, patients)
        targets, scores, ids, hours = _predict(bundle, split.test)
        if len(targets) == 0:
            raise ValueError("test partition has no eligible hourly anchors")
        threshold = float(bundle.threshold)
        events = {}
        for patient in split.test:
            positive = np.flatnonzero(patient.labels == 1)
            if len(positive) and patient.labels[0] == 0:
                events[patient.patient_id] = max(0, int(positive[0]) - bundle.horizon_hours)
        metrics = evaluate_predictions(targets, scores, ids, hours, threshold=threshold,
                                       persistence_hours=bundle.persistence_hours,
                                       first_positive_hours=events,
                                       bootstrap_replicates=bootstrap_replicates, seed=seed)
        admission_positive_ids = {p.patient_id for p in split.test if p.labels[0] == 1}
        pretransition_mask = np.asarray([str(pid) not in admission_positive_ids for pid in ids])
        if pretransition_mask.any():
            pre_ids = np.asarray(ids)[pretransition_mask]
            pre_events = {pid: hour for pid, hour in events.items() if pid in set(pre_ids)}
            pretransition_metrics = evaluate_predictions(
                targets[pretransition_mask], scores[pretransition_mask], pre_ids,
                np.asarray(hours)[pretransition_mask], threshold=threshold,
                persistence_hours=bundle.persistence_hours, first_positive_hours=pre_events,
                bootstrap_replicates=bootstrap_replicates, seed=seed)
        else:
            pretransition_metrics = None
        patient_groups: dict[str, str] = {}
        for file_path in data_path.rglob("*.psv"):
            pieces = [part.lower() for part in file_path.relative_to(data_path).parts[:-1]]
            source = next((part for part in pieces if "seta" in part), None)
            if source is None:
                source = next((part for part in pieces if "setb" in part), None)
            if source is not None:
                patient_groups[file_path.stem] = "source_a" if "seta" in source else "source_b"
        subgroup_metrics: dict[str, Any] = {}
        for group in sorted(set(patient_groups.get(str(pid), "unstratified") for pid in ids)):
            mask = np.asarray([patient_groups.get(str(pid), "unstratified") == group for pid in ids])
            subgroup_metrics[group] = {
                "patient_count": len(set(str(pid) for pid in np.asarray(ids)[mask])),
                "hourly_row_count": int(mask.sum()),
                "metrics": evaluate_predictions(
                    targets[mask], scores[mask], np.asarray(ids)[mask], np.asarray(hours)[mask],
                    threshold=threshold, persistence_hours=bundle.persistence_hours),
            }
        report = {
            "research": True, "source_type": bundle.source_type,
            "selection_sha256": selection["selection_sha256"],
            "bundle_sha256": selection["bundle_sha256"],
            "data_acquisition_sha256": selection["data_acquisition_sha256"],
            "split_sha256": selection["split_sha256"],
            "target": selection["selected_config"]["target"],
            "label_note": selection["selected_config"]["label_note"],
            "additional_target_shift_hours": int(bundle.horizon_hours),
            "event_time_definition": "first published SepsisLabel-positive hour minus the additional target horizon; it is not clinical onset",
            "admission_positive_patients_excluded_from_pretransition_event_map": sum(
                bool(patient.labels[0] == 1) for patient in split.test),
            "pretransition_cohort_counts": {
                "patients": len(set(str(pid) for pid in np.asarray(ids)[pretransition_mask])),
                "hourly_rows": int(pretransition_mask.sum()),
            },
            "test_patient_ids": list(selection["test_patient_ids"]),
            "threshold_frozen_from_validation": threshold,
            "calibration_frozen_from_validation": selection["selected_config"]["calibration"],
            "metrics": metrics, "reliability_bins": calibration_bins(targets, scores, n_bins),
            "pretransition_metrics_excluding_admission_positive": pretransition_metrics,
            "prediction_rows": int(len(targets)),
            "sample_counts": {"patients": len(set(ids)), "hourly_rows": int(len(targets)),
                              "positive_hours": int(targets.sum()),
                              "negative_hours": int(len(targets) - targets.sum())},
            "hospital_subgroup_metrics": subgroup_metrics,
        }
        _atomic_json(report_path, report)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False,
                                         dir=out, prefix=pred_path.name + ".", suffix=".tmp") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "hour", "target", "risk_probability", "research"])
            writer.writerows((str(pid), int(hour), int(y), format(float(p), ".10g"), "true")
                             for pid, hour, y, p in zip(ids, hours, targets, scores))
            stream.flush(); os.fsync(stream.fileno())
            temp_path = Path(stream.name)
        os.replace(temp_path, pred_path)
        _atomic_json(status_path, {"research": True, "status": "completed",
                                   "selection_sha256": selection["selection_sha256"],
                                   "report_sha256": _sha(report_path),
                                   "predictions_sha256": _sha(pred_path)})
        return report
    except Exception as exc:
        _atomic_json(status_path, {"research": True, "status": "failed",
                                   "selection_sha256": selection.get("selection_sha256"),
                                   "error": f"{type(exc).__name__}: {exc}"})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-selection")
    freeze.add_argument("--bundle", required=True); freeze.add_argument("--data", required=True)
    freeze.add_argument("--selection", required=True); freeze.add_argument("--split-manifest")
    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--selection", required=True); evaluate.add_argument("--bundle")
    evaluate.add_argument("--data"); evaluate.add_argument("--split-manifest")
    evaluate.add_argument("--output-dir"); evaluate.add_argument("--bootstrap-replicates", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=42); evaluate.add_argument("--n-bins", type=int, default=10)
    evaluate.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze-selection":
        result = freeze_selection(args.bundle, args.data, args.selection,
                                  split_manifest=args.split_manifest)
        summary = {"research": True, "selection_path": str(Path(args.selection).resolve()),
                   "selection_sha256": result["selection_sha256"],
                   "source_type": result["source_type"]}
    else:
        result = evaluate_test(args.selection, bundle_path=args.bundle, data_dir=args.data,
                               split_manifest=args.split_manifest, output_dir=args.output_dir,
                               bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
                               n_bins=args.n_bins, retry_failed=args.retry_failed)
        primary = result["metrics"]
        summary = {"research": True, "report_path": str((Path(args.output_dir) if args.output_dir else
                    Path(args.selection).parent / (Path(args.selection).stem + "_test")) /
                    "research_test_report.json"), "status": "completed",
                   "prediction_rows": result["prediction_rows"],
                   "metrics": {key: primary.get(key) for key in (
                       "n_hours", "positive_rate", "auroc", "auprc", "brier", "threshold",
                       "sensitivity", "precision", "specificity", "f1",
                       "patient_detection_coverage", "patient_bootstrap_ci_95")}}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
