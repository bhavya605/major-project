"""Research evaluation for hourly published-label sepsis risk predictions."""
from __future__ import annotations

from collections import defaultdict
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _arrays(y: Sequence[int], probabilities: Sequence[float], patient_ids: Sequence[str],
            hours: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    ids = np.asarray(patient_ids, dtype=str)
    times = np.asarray(hours, dtype=int)
    if labels.ndim != 1 or scores.ndim != 1 or ids.ndim != 1 or times.ndim != 1:
        raise ValueError("labels, probabilities, patient_ids, and hours must be one-dimensional")
    if not (len(labels) == len(scores) == len(ids) == len(times)):
        raise ValueError("labels, probabilities, patient_ids, and hours must have equal lengths")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("labels must be binary and probabilities finite in [0, 1]")
    if len(set(zip(ids.tolist(), times.tolist()))) != len(labels):
        raise ValueError("each patient/hour pair must be unique")
    return labels, scores, ids, times


def _persistent_alerts(scores: np.ndarray, ids: np.ndarray, times: np.ndarray,
                       threshold: float, persistence_hours: int) -> np.ndarray:
    if persistence_hours < 1:
        raise ValueError("persistence_hours must be at least one")
    above = scores >= threshold
    alerts = np.zeros(len(scores), dtype=bool)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(ids):
        grouped[str(patient)].append(index)
    for indices in grouped.values():
        indices.sort(key=lambda i: times[i])
        run: list[int] = []
        for index in indices:
            if above[index] and (not run or times[index] == times[run[-1]] + 1):
                run.append(index)
            else:
                if len(run) >= persistence_hours:
                    alerts[run[persistence_hours - 1:]] = True
                run = [index] if above[index] else []
        if len(run) >= persistence_hours:
            alerts[run[persistence_hours - 1:]] = True
    return alerts


def _metrics(y: np.ndarray, alerts: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(alerts & (y == 1)))
    fp = int(np.sum(alerts & (y == 0)))
    tn = int(np.sum(~alerts & (y == 0)))
    fn = int(np.sum(~alerts & (y == 1)))
    positives = tp + fn
    predicted = tp + fp
    negatives = tn + fp
    return {
        "n_hours": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else None,
        "auroc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, probabilities)) if len(y) and positives else None,
        "brier": float(brier_score_loss(y, probabilities)) if len(y) else None,
        "sensitivity": float(tp / positives) if positives else None,
        "precision": float(tp / predicted) if predicted else None,
        "npv": float(tn / (tn + fn)) if tn + fn else None,
        "specificity": float(tn / negatives) if negatives else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _patient_rows(ids: np.ndarray) -> dict[str, np.ndarray]:
    """Build row positions per patient in one pass over the cohort."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, patient in enumerate(ids):
        grouped[str(patient)].append(row)
    return {patient: np.asarray(rows, dtype=np.intp) for patient, rows in grouped.items()}


def _alert_episodes(alerts: np.ndarray, labels: np.ndarray, ids: np.ndarray,
                    times: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, patient in enumerate(ids):
        grouped[str(patient)].append(index)
    episodes: list[dict[str, Any]] = []
    for patient, indices in grouped.items():
        indices.sort(key=lambda i: times[i])
        run: list[int] = []
        for index in indices + [-1]:
            contiguous = bool(run and index >= 0 and times[index] == times[run[-1]] + 1)
            if index >= 0 and alerts[index] and (not run or contiguous):
                run.append(index)
                continue
            if run:
                start, end = run[0], run[-1]
                positive = bool(np.any(labels[run] == 1))
                episodes.append({"patient_id": patient, "start_hour": int(times[start]),
                                 "end_hour": int(times[end]), "duration_hours": len(run),
                                 "overlaps_positive_label": positive,
                                 "false_episode": not positive})
            run = [index] if index >= 0 and alerts[index] else []
    return episodes


def _evaluate_core(y: np.ndarray, scores: np.ndarray, ids: np.ndarray, times: np.ndarray,
                   threshold: float, persistence_hours: int,
                   first_positive_hours: Mapping[str, int] | None = None,
                   patient_rows: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    alerts = _persistent_alerts(scores, ids, times, threshold, persistence_hours)
    metrics = _metrics(y, alerts, scores)
    episodes = _alert_episodes(alerts, y, ids, times)
    negative_hours = int(np.sum(y == 0))
    false_episodes = sum(item["false_episode"] for item in episodes)
    patient_rows = patient_rows if patient_rows is not None else _patient_rows(ids)
    patients = sorted(patient_rows)
    event_hour: dict[str, int] = dict(first_positive_hours or {})
    for patient in patients:
        if patient not in event_hour:
            rows = patient_rows[patient]
            rows = rows[y[rows] == 1]
            if len(rows):
                event_hour[patient] = int(np.min(times[rows]))
    positive_patients = [patient for patient in patients if patient in event_hour]
    detected = 0
    early_detected = 0
    lead_times: list[int] = []
    for patient in positive_patients:
        rows = patient_rows[patient]
        rows = rows[alerts[rows]]
        if not len(rows):
            continue
        detected += 1
        first_alert = int(np.min(times[rows]))
        lead = int(event_hour[patient] - first_alert)
        if lead >= 0:
            early_detected += 1
            lead_times.append(lead)
    metrics.update({
        "research": True,
        "threshold": float(threshold), "persistence_hours": int(persistence_hours),
        "alert_hours": int(alerts.sum()), "alert_episodes": len(episodes),
        "false_alert_episodes": int(false_episodes), "negative_label_hours": negative_hours,
        "false_episodes_per_negative_hour": float(false_episodes / negative_hours) if negative_hours else None,
        "monitored_eligible_hours": int(len(y)),
        "false_episodes_per_monitored_hour": float(false_episodes / len(y)) if len(y) else None,
        "positive_patients": len(positive_patients), "patients_with_any_alert": int(detected),
        "patient_detection_coverage": float(detected / len(positive_patients)) if positive_patients else None,
        "patients_alerted_by_first_positive": int(early_detected),
        "early_detection_coverage": float(early_detected / len(positive_patients)) if positive_patients else None,
        "published_label_lead_hours": lead_times,
        "median_published_label_lead_hours": float(np.median(lead_times)) if lead_times else None,
        "iqr_published_label_lead_hours": [float(value) for value in np.percentile(lead_times, [25, 75])] if lead_times else None,
        "lead_time_population": "Patients whose first alert occurs by the declared published-label transition; late or undetected patients are counted separately in coverage.",
        "alert_episodes_detail": episodes,
        "interpretation": "Research metrics against the published SepsisLabel timeline; not clinical-onset performance.",
    })
    return metrics


def evaluate_predictions(y: Sequence[int], probabilities: Sequence[float],
                         patient_ids: Sequence[str], hours: Sequence[int], *,
                         threshold: float, persistence_hours: int = 2,
                         first_positive_hours: Mapping[str, int] | None = None,
                         bootstrap_replicates: int = 0, seed: int = 42) -> dict[str, Any]:
    """Evaluate hourly risk and persistent alert episodes on held-out patients."""
    labels, scores, ids, times = _arrays(y, probabilities, patient_ids, hours)
    result = _evaluate_core(labels, scores, ids, times, threshold, persistence_hours,
                            first_positive_hours)
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates cannot be negative")
    if bootstrap_replicates:
        unique_patients = np.unique(ids)
        cohort_rows = _patient_rows(ids)
        rng = np.random.default_rng(seed)
        sampled: dict[str, list[float]] = defaultdict(list)
        for _ in range(bootstrap_replicates):
            draw = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            ix: list[int] = []
            boot_ids: list[str] = []
            boot_events: dict[str, int] = {}
            for draw_idx, patient in enumerate(draw):
                rows = cohort_rows[str(patient)]
                ix.extend(rows.tolist())
                boot_patient = f"{draw_idx}:{patient}"
                boot_ids.extend([boot_patient] * len(rows))
                if first_positive_hours is not None and patient in first_positive_hours:
                    boot_events[boot_patient] = int(first_positive_hours[patient])
            boot_ids_array = np.asarray(boot_ids)
            boot = _evaluate_core(labels[ix], scores[ix], boot_ids_array, times[ix],
                                  threshold, persistence_hours, boot_events or None,
                                  _patient_rows(boot_ids_array))
            for key in ("sensitivity", "precision", "npv", "f1", "brier", "auprc", "auroc", "specificity",
                        "patient_detection_coverage", "false_episodes_per_negative_hour",
                        "median_published_label_lead_hours"):
                if boot.get(key) is not None and np.isfinite(boot[key]):
                    sampled[key].append(float(boot[key]))
        result["patient_bootstrap_ci_95"] = {
            key: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            for key, values in sampled.items() if values
        }
        result["patient_bootstrap_valid_replicates"] = {
            key: len(values) for key, values in sampled.items()
        }
        result["patient_bootstrap_replicates"] = int(bootstrap_replicates)
        result["patient_bootstrap_seed"] = int(seed)
    return result


def select_operating_threshold(y: Sequence[int], probabilities: Sequence[float],
                               patient_ids: Sequence[str], hours: Sequence[int], *,
                               persistence_hours: int = 2,
                               min_sensitivity: float | None = None,
                               max_false_episodes_per_hour: float | None = None) -> dict[str, Any]:
    """Select on validation data under explicit sensitivity and false-episode constraints."""
    labels, scores, ids, times = _arrays(y, probabilities, patient_ids, hours)
    if min_sensitivity is not None and not 0 <= min_sensitivity <= 1:
        raise ValueError("min_sensitivity must be in [0, 1]")
    if max_false_episodes_per_hour is not None and max_false_episodes_per_hour < 0:
        raise ValueError("max_false_episodes_per_hour must be nonnegative")
    if persistence_hours < 1:
        raise ValueError("persistence_hours must be at least one")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    # Alerts begin at the hour a trailing persistence window is complete.
    # Each row's activation score is the minimum risk in that trailing window.
    effective = np.full(len(scores), -np.inf)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, patient in enumerate(ids):
        grouped[str(patient)].append(i)
    for indices in grouped.values():
        indices.sort(key=lambda i: times[i])
        run: list[int] = []
        for index in indices + [-1]:
            if index >= 0 and (not run or times[index] == times[run[-1]] + 1):
                run.append(index)
                continue
            length, k = len(run), persistence_hours
            if length >= k:
                # Sliding minimum for each trailing k-hour window.
                minima: list[float] = []
                q: deque[int] = deque()
                for j, row in enumerate(run):
                    while q and scores[run[q[-1]]] >= scores[row]:
                        q.pop()
                    q.append(j)
                    while q and q[0] <= j - k:
                        q.popleft()
                    if j >= k - 1:
                        minima.append(float(scores[run[q[0]]]))
                for j in range(k - 1, length):
                    effective[run[j]] = minima[j - k + 1]
            run = [index] if index >= 0 else []

    order = np.argsort(-effective, kind="stable")
    parent = np.arange(len(labels), dtype=np.int64)
    size = np.ones(len(labels), dtype=np.int64)
    positive = labels.astype(np.int64).copy()
    active = np.zeros(len(labels), dtype=bool)
    false_components = 0
    tp = fp = 0
    by_key = {(str(ids[i]), int(times[i])): i for i in range(len(labels))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a: int, b: int) -> None:
        nonlocal false_components
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if positive[ra] == 0:
            false_components -= 1
        if positive[rb] == 0:
            false_components -= 1
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        positive[ra] += positive[rb]
        if positive[ra] == 0:
            false_components += 1

    best_threshold = None
    best_key = None
    best_feasible = False
    candidate_index = 0
    positives = int(np.sum(labels == 1))
    for threshold in candidates[::-1]:
        while candidate_index < len(order) and effective[order[candidate_index]] >= threshold:
            i = int(order[candidate_index])
            candidate_index += 1
            active[i] = True
            tp += int(labels[i] == 1)
            fp += int(labels[i] == 0)
            if labels[i] == 0:
                false_components += 1
            for neighbor_time in (int(times[i]) - 1, int(times[i]) + 1):
                neighbor = by_key.get((str(ids[i]), neighbor_time))
                if neighbor is not None and active[neighbor]:
                    union(i, neighbor)
        sensitivity = tp / positives if positives else None
        precision = tp / (tp + fp) if tp + fp else None
        f1 = 2 * tp / (2 * tp + fp + positives - tp) if 2 * tp + fp + positives - tp else 0.0
        neg_hours = len(labels) - positives
        false_rate = false_components / neg_hours if neg_hours else None
        ok = ((min_sensitivity is None or (sensitivity is not None and sensitivity >= min_sensitivity))
              and (max_false_episodes_per_hour is None or
                   (false_rate is not None and false_rate <= max_false_episodes_per_hour)))
        key = ((f1, float(threshold)) if min_sensitivity is None and max_false_episodes_per_hour is None
               else (precision if precision is not None else -1, f1, float(threshold)))
        if ok and (best_key is None or key > best_key):
            best_key, best_threshold, best_feasible = key, float(threshold), True
    if not best_feasible:
        return {"research": True, "feasible": False, "threshold": None,
                "constraints": {"min_sensitivity": min_sensitivity,
                                "max_false_episodes_per_hour": max_false_episodes_per_hour},
                "candidate_count": len(candidates),
                "message": "No validation threshold satisfies all requested constraints."}
    best = _evaluate_core(labels, scores, ids, times, best_threshold, persistence_hours)
    return {"research": True, "feasible": True, "threshold": best["threshold"], "validation_metrics": best,
            "constraints": {"min_sensitivity": min_sensitivity,
                            "max_false_episodes_per_hour": max_false_episodes_per_hour},
            "candidate_count": len(candidates)}


def calibration_bins(y: Sequence[int], probabilities: Sequence[float],
                     n_bins: int = 10) -> list[dict[str, Any]]:
    """Return equal-width reliability bins for validation or held-out reporting."""
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("y and probabilities must be matching finite vectors")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for index in range(n_bins):
        mask = (scores >= edges[index]) & (scores <= edges[index + 1] if index == n_bins - 1 else scores < edges[index + 1])
        if mask.any():
            rows.append({"research": True, "bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]),
                         "count": int(mask.sum()), "mean_probability": float(scores[mask].mean()),
                         "observed_rate": float(labels[mask].mean())})
    return rows


class ProbabilityCalibrator:
    """Small adapter fitted on validation probabilities only."""
    def __init__(self, method: str, estimator: Any):
        self.method, self.estimator = method, estimator

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        scores = np.asarray(probabilities, dtype=float)
        if self.method == "platt":
            eps = np.finfo(float).eps
            logits = np.log(np.clip(scores, eps, 1 - eps) / np.clip(1 - scores, eps, 1))[:, None]
            return self.estimator.predict_proba(logits)[:, 1]
        return np.asarray(self.estimator.predict(scores), dtype=float).clip(0, 1)


def fit_calibrator(y: Sequence[int], probabilities: Sequence[float], *,
                   method: str = "platt") -> ProbabilityCalibrator:
    """Fit Platt or isotonic calibration; caller must provide validation predictions."""
    labels = np.asarray(y, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or len(np.unique(labels)) != 2:
        raise ValueError("calibration requires matching labels and both classes")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if method == "platt":
        eps = np.finfo(float).eps
        logits = np.log(np.clip(scores, eps, 1 - eps) / np.clip(1 - scores, eps, 1))[:, None]
        estimator = LogisticRegression(C=1e6, solver="lbfgs", random_state=0).fit(logits, labels)
    elif method == "isotonic":
        estimator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores, labels)
    else:
        raise ValueError("method must be 'platt' or 'isotonic'")
    return ProbabilityCalibrator(method, estimator)
