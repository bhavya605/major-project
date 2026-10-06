"""Temporal explanation faithfulness and stability checks for research bundles.

All reported quantities describe model behavior on sampled TRAIN windows. They
are software research diagnostics, not clinical or benchmark performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import Patient
from .sequences import SequenceDataset
from .schema import FEATURE_COLUMNS, load_patient_file
from .temporal import integrated_gradients, missingness_integrated_gradients
from .temporal_model import load_temporal_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else (1.0 if not np.any(a) and not np.any(b) else 0.0)


def _hour_offset(local_index: int, window_hours: int) -> int:
    """Offset from the window's current (rightmost) hour, including left padding."""
    return int(local_index - (window_hours - 1))


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation with average ranks for tied absolute attributions."""
    from scipy.stats import rankdata
    ra, rb = rankdata(-np.abs(np.asarray(a).ravel())), rankdata(-np.abs(np.asarray(b).ravel()))
    return _cosine(ra - ra.mean(), rb - rb.mean())


def _select_train_hours(patients, max_samples: int, seed: int) -> list[tuple[int, int]]:
    """Select hours based only on patient identity and sequence length, never labels."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    candidates = [(pi, hour) for pi, patient in enumerate(patients)
                  for hour in range(len(patient.values))]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=min(max_samples, len(candidates)), replace=False)
    return [candidates[int(i)] for i in chosen]


def _baseline_from_train(patients, feature_count: int, means: np.ndarray,
                         scales: np.ndarray) -> np.ndarray:
    # Statistics are computed only from observed TRAIN values; no held-out patient
    # or label can affect the alternative reference input.
    columns = []
    for feature in range(feature_count):
        observed = np.concatenate([p.values[:, feature][np.isfinite(p.values[:, feature])]
                                  for p in patients])
        median = float(np.median(observed)) if len(observed) else float(means[feature])
        columns.append((median - means[feature]) / scales[feature])
    return np.asarray(columns, dtype=np.float32)


def _load_train_patients(data_dir: str | Path, train_ids: list[str], all_ids: list[str]) -> list[Patient]:
    """Check the cohort using filenames, then parse labels/features for TRAIN only."""
    paths = sorted(Path(data_dir).rglob("*.psv"))
    by_id: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_id:
            raise ValueError(f"duplicate patient ID {path.stem!r} in supplied data")
        by_id[path.stem] = path
    if set(by_id) != set(all_ids):
        raise ValueError("split manifest patient IDs do not exactly match supplied data filenames")
    patients = []
    for patient_id in train_ids:
        frame = load_patient_file(by_id[patient_id])
        patients.append(Patient(patient_id,
                                frame.loc[:, FEATURE_COLUMNS].to_numpy(),
                                frame["SepsisLabel"].to_numpy()))
    return patients


def _integrated(model, x, observed, valid, baseline, steps: int) -> np.ndarray:
    return integrated_gradients(model, x, observed, valid, baseline=baseline, steps=steps)[0].detach().cpu().numpy()


def _logit(model, x, observed, valid) -> float:
    with torch.no_grad():
        return float(model(x, observed, valid)[0].item())


def _window(patient, hour: int, bundle):
    one = type("WindowPatient", (), {})
    # SequenceDataset materializes causal, left-padded windows; its target is
    # intentionally ignored by this analysis.
    p = one()
    p.patient_id, p.values, p.labels = patient.patient_id, patient.values[:hour + 1], patient.labels[:hour + 1]
    ds = SequenceDataset([p], bundle.preprocessor, bundle.window_hours, horizon=0)
    return tuple(t.unsqueeze(0) for t in ds[-1][:3])


def analyze(bundle_path: str | Path, data_dir: str | Path, split_manifest: str | Path,
            output_dir: str | Path, *, max_samples: int = 12, seed: int = 42,
            steps: int = 64) -> dict[str, Any]:
    if steps <= 0 or steps > 256:
        raise ValueError("steps must be between 1 and 256")
    bundle_path, manifest_path = Path(bundle_path), Path(split_manifest)
    if bundle_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
        raise ValueError("tabular bundles are not supported by the temporal faithfulness suite yet")
    bundle = load_temporal_model(bundle_path)
    if bundle.architecture not in {"gru", "lstm", "transformer"}:
        raise ValueError("faithfulness suite supports GRU, LSTM, and Transformer bundles only")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = manifest.get("patient_ids", manifest)
    if set(groups) != {"train", "validation", "test"}:
        raise ValueError("split manifest must contain train, validation, and test patient_ids")
    all_ids = [pid for part in ("train", "validation", "test") for pid in groups[part]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("split manifest assigns a patient more than once")
    bundle_groups = bundle.metadata.get("split_patient_ids")
    if not isinstance(bundle_groups, dict) or any(list(groups[k]) != list(bundle_groups.get(k, []))
                                                  for k in ("train", "validation", "test")):
        raise ValueError("split manifest patient IDs/order do not match frozen bundle metadata")
    train_patients = _load_train_patients(data_dir, list(groups["train"]), all_ids)
    if not train_patients:
        raise ValueError("split manifest TRAIN partition is empty")
    if bundle.metadata.get("model_sha256") is None:
        raise ValueError("bundle metadata is missing model_sha256")

    selected = _select_train_hours(train_patients, max_samples, seed)
    baseline_feature = _baseline_from_train(train_patients, len(bundle.feature_names),
                                             bundle.preprocessor.means, bundle.preprocessor.scales)
    output_samples = []
    global_abs = np.zeros(len(bundle.feature_names), dtype=float)
    for pi, hour in selected:
        patient = train_patients[pi]
        x, observed, valid = _window(patient, hour, bundle)
        base_zero = torch.zeros_like(x)
        baseline = torch.as_tensor(np.broadcast_to(baseline_feature, tuple(x.shape)).copy(), dtype=x.dtype)
        baseline = baseline * valid.unsqueeze(-1)
        attr = _integrated(bundle.model, x, observed, valid, base_zero, steps)
        alt_attr = _integrated(bundle.model, x, observed, valid, baseline, steps)
        global_abs += np.abs(attr).sum(axis=0)
        valid_np, obs_np, x_np = valid[0].numpy(), observed[0].numpy(), x[0].numpy()
        active = valid_np[:, None] & obs_np
        # Stability: small normalized-input perturbation on observed inputs,
        # preserving masks and valid-time padding.
        rng = np.random.default_rng(seed + pi * 1009 + hour)
        perturb = torch.as_tensor(rng.normal(0, 0.01, size=x.shape).astype(np.float32))
        perturb *= (observed & valid.unsqueeze(-1)).to(perturb.dtype)
        pert_attr = _integrated(bundle.model, x + perturb, observed, valid, base_zero, steps)
        rank_corr, cosine = _rank_correlation(attr, pert_attr), _cosine(attr, pert_attr)

        logit = _logit(bundle.model, x, observed, valid)
        zero_logit = _logit(bundle.model, base_zero, observed, valid)
        base_logit = _logit(bundle.model, baseline, observed, valid)
        if getattr(bundle.model, "use_observation_mask", False):
            mask_attr = missingness_integrated_gradients(bundle.model, x, observed, valid, steps=steps)[0].detach().cpu().numpy()
            zero_mask_logit = _logit(bundle.model, x, torch.zeros_like(observed), valid)
            mask_complete = float(mask_attr.sum() - (logit - zero_mask_logit))
        else:
            mask_attr, zero_mask_logit, mask_complete = None, None, None

        # Delete the largest signed-value attribution cell by moving the
        # observed normalized value to the zero training-mean baseline. Compare
        # with one randomly chosen observed cell at the same hour.
        top_candidates = np.argwhere(active)
        top_drop = random_drop = None
        top_identity = random_identity = None
        if len(top_candidates):
            order = np.argsort((-np.abs(attr) * active).ravel())
            flat_top = next(int(i) for i in order if active.ravel()[i])
            tr, tf = np.unravel_index(flat_top, attr.shape)
            random_candidates = np.flatnonzero(active[tr])
            other_candidates = random_candidates[random_candidates != tf]
            random_comparison_fallback = not len(other_candidates)
            random_feature = int(rng.choice(other_candidates if len(other_candidates) else random_candidates))
            modified = x.clone(); modified[0, tr, tf] = 0.0
            random_modified = x.clone(); random_modified[0, tr, random_feature] = 0.0
            top_drop = logit - _logit(bundle.model, modified, observed, valid)
            random_drop = logit - _logit(bundle.model, random_modified, observed, valid)
            top_identity = {"hour_offset": _hour_offset(tr, x.shape[1]), "feature": bundle.feature_names[tf]}
            random_identity = {"hour_offset": _hour_offset(tr, x.shape[1]), "feature": bundle.feature_names[random_feature]}

        # Nearby-hour explanation consistency uses the immediately preceding
        # anchor for the same training patient when available.
        nearby = None
        if hour > 0:
            nx, no, nv = _window(patient, hour - 1, bundle)
            near_attr = _integrated(bundle.model, nx, no, nv, torch.zeros_like(nx), steps)
            nearby = {"patient_id": patient.patient_id, "hour": hour - 1,
                      "cosine": _cosine(attr, near_attr), "rank_correlation": _rank_correlation(attr, near_attr),
                      "description": "adjacent-anchor explanation similarity; descriptive, noncausal"}

        output_samples.append({
            "sample_id": f"{patient.patient_id}@{hour}", "patient_id": patient.patient_id,
            "anchor_hour_index": int(hour), "label_used_for_selection": False,
            "input_logit": logit,
            "value_ig": {"steps": steps, "baseline_zero_training_normalized": True,
                         "sum": float(attr.sum()), "logit_difference": float(logit - zero_logit),
                         "signed_raw_logit_completeness_residual": float(attr.sum() - (logit - zero_logit)),
                         "alternative_baseline_training_normalized_feature_medians": baseline_feature.tolist(),
                         "alternative_baseline_logit": base_logit,
                         "alternative_baseline_attribution_cosine": _cosine(attr, alt_attr),
                         "alternative_baseline_absolute_rank_correlation": _rank_correlation(attr, alt_attr),
                         "alternative_completeness_residual": float(alt_attr.sum() - (logit - base_logit))},
            "conditional_mask_ig": (None if mask_attr is None else {
                "sum": float(mask_attr.sum()), "all_missing_mask_logit": zero_mask_logit,
                "completeness_residual": mask_complete,
                "note": "Separate conditional attribution with values fixed; never summed with value IG."}),
            "stability": {"observed_input_perturbation_sd_normalized": 0.01,
                          "cosine": cosine, "absolute_attribution_rank_correlation": rank_corr,
                          "nearby_hour": nearby},
            "replacement_toward_zero_baseline": {"top_attributed_observed_cell": top_identity,
                "matched_random_observed_cell": random_identity, "top_logit_drop": top_drop,
                "random_logit_drop": random_drop,
                "random_comparison_reused_top_feature": random_comparison_fallback if len(top_candidates) else None,
                "description": "raw-logit changes after replacing one observed cell toward the baseline; descriptive, noncausal"},
        })
    sample_counts = np.zeros(len(bundle.feature_names), dtype=int)
    window_counts = np.zeros(len(bundle.feature_names), dtype=int)
    for row in output_samples:
        # Counts reflect how often each feature was observed among the selected
        # valid time positions, not attribution signs.
        pi, hour = next((i, h) for i, h in selected if f"{train_patients[i].patient_id}@{h}" == row["sample_id"])
        _, observed, valid = _window(train_patients[pi], hour, bundle)
        feature_observed = (observed[0].numpy() & valid[0].numpy()[:, None]).any(axis=0)
        sample_counts += (observed[0].numpy() & valid[0].numpy()[:, None]).sum(axis=0)
        window_counts += feature_observed
    ranking = [{"feature": bundle.feature_names[i], "sum_absolute_value_ig": float(global_abs[i]),
                "observed_cell_count": int(sample_counts[i]),
                "windows_with_observation_count": int(window_counts[i])}
               for i in range(len(bundle.feature_names))]
    ranking.sort(key=lambda row: row["sum_absolute_value_ig"], reverse=True)
    result = {"research": True, "title": "Research-only temporal explanation validation",
        "clinical_or_physionet_performance_claim": False,
        "nonclinical_note": "Synthetic outputs are software demonstrations only; attribution diagnostics do not establish clinical validity.",
        "model_sha256": _sha256(bundle_path), "bundle_metadata_model_sha256": bundle.metadata["model_sha256"],
        "split_manifest_sha256": _sha256(manifest_path), "architecture": bundle.architecture,
        "source_type": bundle.source_type,
        "source_provenance": {"training_source": bundle.metadata.get("training_source"),
                              "bundle_source_type": bundle.metadata.get("source_type", bundle.source_type)},
        "partition_used": "train", "selection": {"method": "uniform seeded sample without replacement over TRAIN patient-hours",
            "max_samples": max_samples, "selected_samples": len(output_samples), "seed": seed,
            "labels_used_for_selection": False},
        "method_steps": ["validate manifest partition IDs against frozen bundle metadata and supplied data",
            "sample TRAIN patient-hours without using labels", "compute value IG with observed and valid-time masks fixed",
            "compute separate conditional mask IG when supported", "compare zero and TRAIN normalized-median value baselines",
            "compare attribution ranks/cosines under small input perturbations and adjacent anchors",
            "replace top-attributed observed value toward baseline and compare a random observed channel at the same timestep"],
        "samples": output_samples, "global_feature_ranking": ranking}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "explanation_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ig-steps", type=int, default=64)
    args = parser.parse_args()
    result = analyze(args.bundle, args.data, args.split_manifest, args.output,
                     max_samples=args.max_samples, seed=args.seed, steps=args.ig_steps)
    print(f"Wrote research-only explanation diagnostics for {len(result['samples'])} TRAIN windows.")


if __name__ == "__main__":
    main()
