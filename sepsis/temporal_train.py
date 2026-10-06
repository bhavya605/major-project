"""Train compact temporal models for explicitly labeled sepsis research.

Example: ``python -m sepsis.temporal_train --data data/demo --output artifacts/gru.pt``.
The default target is the current published label (horizon zero); the PhysioNet
label itself is already shifted six hours before recorded clinical onset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (Patient, PatientSplit, Preprocessor,
                   load_patients, split_patients)
from .evaluation import (evaluate_predictions, fit_calibrator,
                         select_operating_threshold)
from .schema import FEATURE_COLUMNS
from .sequences import SequenceDataset
from .temporal_model import (TemporalModelBundle, _architecture,
                            save_temporal_model)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _average_precision(y: np.ndarray, probabilities: np.ndarray) -> float:
    if not len(y) or not np.any(y == 1):
        raise ValueError("validation windows need at least one positive target for AUPRC early stopping")
    return float(average_precision_score(y, probabilities))


def _validation_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Select F1 threshold from validation predictions only, tie-breaking high."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.unique(np.r_[0.0, probabilities, 1.0]):
        predicted = probabilities >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_f1, best_threshold = f1, float(threshold)
    return best_threshold


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = probability >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None,
        "auprc": float(average_precision_score(y, probability)) if np.any(y == 1) else None,
        "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, probability)) if len(y) else None,
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
    }


def _split_from_manifest(patients: list[Patient], path: Path) -> PatientSplit:
    by_id = {patient.patient_id: patient for patient in patients}
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = raw.get("patient_ids", raw)
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test patient_ids")
    all_ids = [pid for name in ("train", "validation", "test") for pid in groups[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split manifest assigns a patient more than once")
    if set(all_ids) != set(by_id):
        missing, unknown = set(by_id) - set(all_ids), set(all_ids) - set(by_id)
        raise ValueError(f"split manifest patient IDs do not match data (missing={len(missing)}, unknown={len(unknown)})")
    return PatientSplit(*(tuple(by_id[pid] for pid in groups[name])
                          for name in ("train", "validation", "test")))


def _tensors(batch) -> tuple[torch.Tensor, ...]:
    return (
        torch.as_tensor(batch.values, dtype=torch.float32),
        torch.as_tensor(batch.observed_mask, dtype=torch.bool),
        torch.as_tensor(batch.valid_time_mask, dtype=torch.bool),
        torch.as_tensor(batch.targets, dtype=torch.float32),
    )


def _predict(model: nn.Module, batch, batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    dataset = batch if isinstance(batch, SequenceDataset) else TensorDataset(*_tensors(batch))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for values, observed, valid, targets in loader:
            logits = model(values, observed, valid)
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def _published_event_anchor_hours(patients: tuple[Patient, ...], horizon: int) -> dict[str, int]:
    """Map first positive target labels back to their input anchor hours."""
    result: dict[str, int] = {}
    for patient in patients:
        positive = np.flatnonzero(patient.labels == 1)
        if len(positive):
            # The target is label[anchor + horizon], so its transition is shifted
            # back to the corresponding input hour before evaluation.
            result[patient.patient_id] = max(0, int(positive[0]) - horizon)
    return result


def _episode_metrics(batch, probability: np.ndarray, threshold: float,
                     persistence_hours: int, patients: tuple[Patient, ...],
                     horizon: int) -> dict[str, Any]:
    return evaluate_predictions(
        batch.targets, probability, batch.patient_ids, batch.hours,
        threshold=threshold, persistence_hours=persistence_hours,
        first_positive_hours=_published_event_anchor_hours(patients, horizon),
    )


def train(data_dir: str | Path, output: str | Path, *, model_type: str = "gru",
          window: int = 12, horizon: int = 0, random_state: int = 42,
          persistence_hours: int = 2, epochs: int = 12, batch_size: int = 256,
          learning_rate: float = 1e-3, patience: int = 3,
          gradient_clip: float = 1.0, loss: str = "weighted_bce",
          focal_gamma: float = 2.0, hidden_size: int = 32,
          num_layers: int = 1, dropout: float = 0.1,
          d_model: int = 32, num_heads: int = 4,
          use_observation_mask: bool = True,
          evaluate_test: bool = False,
          calibration: str = "none",
          min_sensitivity: float | None = None,
          max_false_episodes_per_hour: float | None = None,
          source_type: str,
          split_manifest: str | Path | None = None) -> dict[str, Any]:
    """Fit one CPU temporal model and persist its validation-selected bundle."""
    started = time.perf_counter()
    if model_type not in {"gru", "lstm", "transformer"}:
        raise ValueError("model_type must be gru, lstm, or transformer")
    if calibration not in {"none", "platt", "isotonic"}:
        raise ValueError("calibration must be none, platt, or isotonic")
    if source_type not in {"synthetic_demo", "physionet"}:
        raise ValueError("source_type must be synthetic_demo or physionet")
    if window <= 0 or horizon < 0:
        raise ValueError("window must be positive and horizon nonnegative")
    if epochs <= 0 or batch_size <= 0 or patience <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, patience, and learning_rate must be positive")
    if persistence_hours <= 0 or gradient_clip <= 0:
        raise ValueError("persistence_hours and gradient_clip must be positive")
    if loss not in {"weighted_bce", "focal"} or focal_gamma < 0:
        raise ValueError("loss must be weighted_bce or focal; focal_gamma must be nonnegative")
    if model_type == "transformer" and d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")

    _seed_everything(random_state)
    patients = load_patients(data_dir)
    manifest_path = Path(split_manifest) if split_manifest is not None else None
    if manifest_path and manifest_path.exists():
        split = _split_from_manifest(patients, manifest_path)
    else:
        split = split_patients(patients, test_size=0.15, validation_size=0.15,
                               random_state=random_state)
    if manifest_path and not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "random_state": int(random_state),
            "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "patient_ids": {
                "train": [patient.patient_id for patient in split.train],
                "validation": [patient.patient_id for patient in split.validation],
                "test": [patient.patient_id for patient in split.test],
            },
        }, indent=2), encoding="utf-8")
    detected_source_type = ("synthetic_demo" if (Path(data_dir) / "demo_manifest.json").exists()
                            else "physionet")
    if source_type != detected_source_type:
        raise ValueError(f"source_type={source_type!r} disagrees with detected data provenance {detected_source_type!r}")
    if not split.train or not split.validation:
        raise ValueError("patient split must have nonempty train and validation patients")
    calibration_patients: tuple[Patient, ...] = ()
    operating_patients = split.validation
    if calibration != "none":
        # Match tabular training: fit calibration and select the operating point
        # on disjoint validation patients. Use the operating group for early stopping too.
        cal_split = split_patients(split.validation, test_size=0.0, validation_size=0.5,
                                   random_state=random_state + 1)
        calibration_patients, operating_patients = cal_split.train, cal_split.validation
        if not calibration_patients or not operating_patients:
            raise ValueError("validation patients are too few to separate calibration and threshold selection")
    preprocessor = Preprocessor.fit(split.train)
    train_batch = SequenceDataset(split.train, preprocessor, window, horizon)
    val_batch = SequenceDataset(operating_patients, preprocessor, window, horizon)
    if not len(train_batch) or not len(val_batch):
        raise ValueError("requested window/horizon produced no eligible train or validation windows")
    train_y = train_batch.targets.astype(np.float32)
    val_y = val_batch.targets.astype(np.int8)
    positives = int(train_y.sum())
    negatives = int(len(train_y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training windows must contain both target classes")
    if not np.any(val_y == 1):
        raise ValueError("validation windows need a positive target for AUPRC early stopping")

    params: dict[str, Any]
    if model_type == "transformer":
        params = {"d_model": d_model, "num_heads": num_heads, "num_layers": num_layers,
                  "dim_feedforward": d_model * 2, "dropout": dropout,
                  "use_observation_mask": use_observation_mask}
    else:
        params = {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout,
                  "use_observation_mask": use_observation_mask}
    model = _architecture(model_type, len(FEATURE_COLUMNS), params).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(train_batch, batch_size=batch_size, shuffle=True,
                        generator=generator, drop_last=False)
    best_auprc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    remaining = patience
    training_loss_history: list[float] = []

    fitting_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for values, observed, valid, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values, observed, valid)
            per_row = bce(logits, targets)
            if loss == "weighted_bce":
                weights = torch.where(targets > 0.5, pos_weight[0], 1.0)
                per_row = per_row * weights
            else:
                probability = torch.sigmoid(logits)
                pt = torch.where(targets > 0.5, probability, 1 - probability)
                weights = torch.where(targets > 0.5, pos_weight[0], 1.0)
                per_row = (1 - pt).pow(focal_gamma) * per_row * weights
            batch_loss = per_row.mean()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            losses.append(float(batch_loss.detach().item()))
        training_loss_history.append(float(np.mean(losses)))
        val_probability = _predict(model, val_batch, batch_size)
        current_auprc = _average_precision(val_y, val_probability)
        if current_auprc > best_auprc:
            best_auprc = current_auprc
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    fitting_seconds = time.perf_counter() - fitting_started
    model.load_state_dict(best_state)
    model_digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        model_digest.update(name.encode("utf-8"))
        model_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    calibrator = None
    calibration_window_count = 0
    if calibration != "none":
        calibration_batch = SequenceDataset(calibration_patients, preprocessor, window, horizon)
        calibration_window_count = len(calibration_batch)
        if not len(calibration_batch):
            raise ValueError("calibration validation subset has no eligible windows")
        raw_calibration = _predict(model, calibration_batch, batch_size)
        try:
            calibrator = fit_calibrator(calibration_batch.targets.astype(np.int8),
                                        raw_calibration, method=calibration)
        except ValueError as exc:
            raise ValueError(f"validation calibration subset is infeasible: {exc}") from exc
    val_probability_raw = _predict(model, val_batch, batch_size)
    val_probability = (calibrator.predict(val_probability_raw) if calibrator is not None
                       else val_probability_raw)
    threshold_selection = select_operating_threshold(
        val_y, val_probability, val_batch.patient_ids, val_batch.hours,
        persistence_hours=persistence_hours, min_sensitivity=min_sensitivity,
        max_false_episodes_per_hour=max_false_episodes_per_hour,
    )
    if not threshold_selection["feasible"]:
        raise ValueError("validation threshold constraints are infeasible; no model bundle was written: " +
                         threshold_selection["message"])
    threshold = float(threshold_selection["threshold"])
    val_metrics = _metrics(val_y, val_probability, threshold)
    validation_episode_metrics = _episode_metrics(
        val_batch, val_probability, threshold, persistence_hours,
        split.validation, horizon,
    )

    metadata: dict[str, Any] = {
        "research": True,
        "runtime_seconds_before_serialization": time.perf_counter() - started,
        "fitting_and_early_stopping_seconds": fitting_seconds,
        "training_source": str(Path(data_dir).resolve()),
        "source_type": source_type,
        "target": "published SepsisLabel at anchor hour + additional published-label horizon",
        "horizon_hours": int(horizon),
        "label_note": "PhysioNet SepsisLabel already starts six hours before recorded clinical onset; an additional horizon shifts the published label and is not a clinical-onset forecast.",
        "random_state": int(random_state),
        "split_patient_ids": {
            "train": [patient.patient_id for patient in split.train],
            "validation": [patient.patient_id for patient in split.validation],
            "test": [patient.patient_id for patient in split.test],
        },
        "split_manifest": str(manifest_path.resolve()) if manifest_path else None,
        "split_patient_counts": {"train": len(split.train), "validation": len(split.validation),
                                 "test": len(split.test)},
        "split_window_counts": {"train": len(train_batch), "validation": len(val_batch),
                                "test": None},
        "best_epoch": best_epoch,
        "epochs_completed": len(training_loss_history),
        "training_loss_history": training_loss_history,
        "validation_metrics": val_metrics,
        "validation_auprc_at_best_epoch": float(best_auprc),
        "threshold_selected_on": "validation alert metrics",
        "threshold_selection": threshold_selection,
        "threshold_constraints": {"min_sensitivity": min_sensitivity,
                                   "max_false_episodes_per_hour": max_false_episodes_per_hour},
        "calibration": calibration,
        "calibration_patient_ids": [patient.patient_id for patient in calibration_patients],
        "operating_point_patient_ids": [patient.patient_id for patient in operating_patients],
        "calibration_window_count": calibration_window_count,
        "persistence_hours": int(persistence_hours),
        "loss": loss,
        "focal_gamma": float(focal_gamma),
        "class_weight_positive": float(negatives / positives),
        "gradient_clip_norm": float(gradient_clip),
        "model_parameters": params,
        "input_representation": "values_plus_observation_mask" if use_observation_mask else "values_only",
        "bundle_version": 1,
        "model_sha256": model_digest.hexdigest(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "estimator": f"PyTorch {model_type.upper()} temporal classifier",
        "test_evaluated": bool(evaluate_test),
    }
    test_metrics = None
    if evaluate_test and split.test:
        test_batch = SequenceDataset(split.test, preprocessor, window, horizon)
        if len(test_batch):
            test_probability_raw = _predict(model, test_batch, batch_size)
            test_probability = (calibrator.predict(test_probability_raw) if calibrator is not None
                                else test_probability_raw)
            test_metrics = _metrics(test_batch.targets.astype(np.int8), test_probability, threshold)
            test_episode_metrics = _episode_metrics(
                test_batch, test_probability, threshold, persistence_hours,
                split.test, horizon,
            )
        else:
            test_batch = None
            test_episode_metrics = None
    else:
        test_batch = None
        test_episode_metrics = None
    if evaluate_test:
        metadata["split_window_counts"]["test"] = len(test_batch) if test_batch is not None else 0
        metadata["test_metrics"] = test_metrics
        metadata["test_episode_metrics"] = test_episode_metrics
    metadata["validation_episode_metrics"] = validation_episode_metrics
    metadata["episode_event_time_definition"] = (
        "First positive published SepsisLabel hour shifted back by the configured target horizon to the corresponding input-anchor hour; not reconstructed clinical onset."
    )

    bundle = TemporalModelBundle(
        model=model, preprocessor=preprocessor, feature_names=tuple(FEATURE_COLUMNS),
        window_hours=window, horizon_hours=horizon, threshold=threshold,
        persistence_hours=persistence_hours, source_type=source_type,
        metadata=metadata, architecture=model_type,
        calibrator=calibrator,
    )
    output_path = save_temporal_model(bundle, output)
    report_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing patient PSV files")
    parser.add_argument("--output", required=True, help="Path for saved temporal model bundle")
    parser.add_argument("--model", choices=("gru", "lstm", "transformer"), default="gru")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=0,
                        help="Additional future published-label horizon (default: official current label)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--persistence-hours", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=("weighted_bce", "focal"), default="weighted_bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--values-only", action="store_true",
                        help="Train on normalized values without observation-mask features")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Explicitly evaluate locked test patients after validation selection")
    parser.add_argument("--calibration", choices=("none", "platt", "isotonic"), default="none",
                        help="Fit optional probability calibration on validation patients separate from threshold selection")
    parser.add_argument("--min-sensitivity", type=float,
                        help="Optional validation operating-point sensitivity constraint")
    parser.add_argument("--max-false-episodes-per-hour", type=float,
                        help="Optional validation false alert episodes per negative-label hour constraint")
    parser.add_argument("--source-type", choices=("synthetic_demo", "physionet"), required=True,
                        help="Explicit provenance label; must match demo_manifest.json detection")
    parser.add_argument("--split-manifest", help="Read an existing patient split or write one at this path")
    args = parser.parse_args()
    result = train(
        args.data, args.output, model_type=args.model, window=args.window,
        horizon=args.horizon, random_state=args.random_state,
        persistence_hours=args.persistence_hours, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        patience=args.patience, gradient_clip=args.gradient_clip,
        loss=args.loss, focal_gamma=args.focal_gamma, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, d_model=args.d_model,
        num_heads=args.num_heads, evaluate_test=args.evaluate_test,
        use_observation_mask=not args.values_only,
        source_type=args.source_type,
        split_manifest=args.split_manifest,
        calibration=args.calibration,
        min_sensitivity=args.min_sensitivity,
        max_false_episodes_per_hour=args.max_false_episodes_per_hour,
    )
    print(json.dumps({"research": True, "model": str(Path(args.output).resolve()),
                      "source_type": result["source_type"],
                      "split_patient_counts": result["split_patient_counts"],
                      "validation_metrics": result["validation_metrics"],
                      "test_metrics": result.get("test_metrics")}, indent=2))


if __name__ == "__main__":
    main()
