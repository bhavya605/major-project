"""Patient-level loading, preprocessing, splitting, and causal windows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .schema import FEATURE_COLUMNS, LABEL_COLUMN, load_patient_file


@dataclass(frozen=True)
class Patient:
    """One patient's hourly feature matrix and published SepsisLabel sequence."""
    patient_id: str
    values: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        labels = np.asarray(self.labels)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"values must have shape (hours, {len(FEATURE_COLUMNS)})")
        if len(values) == 0:
            raise ValueError("patient must contain at least one hourly row")
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("labels must be a 1D array matching values rows")
        if np.isinf(values).any():
            raise ValueError("values cannot contain infinity")
        iculos_index = FEATURE_COLUMNS.index("ICULOS")
        iculos = values[:, iculos_index]
        if not np.isfinite(iculos).all() or (len(iculos) > 1 and np.any(np.diff(iculos) != 1)):
            raise ValueError("ICULOS must be finite and advance one hour per row")
        if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError("labels must contain only 0/1 values")
        if np.any(np.diff(labels.astype(np.int8)) < 0):
            raise ValueError("labels must be monotonic (0s followed by 1s)")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "labels", labels.astype(np.int8, copy=False))


@dataclass(frozen=True)
class PatientSplit:
    train: tuple[Patient, ...]
    validation: tuple[Patient, ...]
    test: tuple[Patient, ...]


@dataclass(frozen=True)
class WindowBatch:
    values: np.ndarray
    observed_mask: np.ndarray
    valid_time_mask: np.ndarray
    targets: np.ndarray
    patient_ids: tuple[str, ...]
    hours: np.ndarray

    @property
    def masks(self) -> np.ndarray:
        """Convenience alias for the per-feature observation mask."""
        return self.observed_mask

    def __len__(self) -> int:
        return len(self.targets)


def load_patients(directory: str | Path) -> list[Patient]:
    """Load PSV files recursively, rejecting duplicate patient stems across sources."""
    root = Path(directory)
    paths = sorted(root.rglob("*.psv"))
    if not paths:
        raise FileNotFoundError(f"no .psv patient files found in {root}")
    patients = []
    seen: set[str] = set()
    for path in paths:
        patient_id = path.stem
        if patient_id in seen:
            raise ValueError(f"duplicate patient ID {patient_id!r} in {root}; source A/B files overlap")
        seen.add(patient_id)
        frame = load_patient_file(path)
        patients.append(Patient(patient_id, frame.loc[:, FEATURE_COLUMNS].to_numpy(),
                                frame[LABEL_COLUMN].to_numpy()))
    return patients


def split_patients(
    patients: Sequence[Patient], test_size: float = 0.15, validation_size: float = 0.15,
    random_state: int = 42,
) -> PatientSplit:
    """Deterministically split by patient, approximately stratifying sepsis presence.

    Tiny cohorts can make exact stratification impossible. In that case the split
    stays deterministic, always leaves at least one training patient, and assigns
    available held-out patients without duplicating or losing any patient.
    """
    patients = list(patients)
    if not patients:
        raise ValueError("patients must not be empty")
    if len({p.patient_id for p in patients}) != len(patients):
        raise ValueError("patient_id values must be unique")
    if not (0 <= test_size < 1 and 0 <= validation_size < 1 and test_size + validation_size < 1):
        raise ValueError("test_size and validation_size must be nonnegative and sum to less than 1")
    n = len(patients)
    n_test = min(int(round(n * test_size)), max(0, n - 1))
    n_val = min(int(round(n * validation_size)), max(0, n - 1 - n_test))
    rng = np.random.default_rng(random_state)
    buckets = {0: [], 1: []}
    for patient in patients:
        buckets[int(np.any(patient.labels == 1))].append(patient)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Allocate each stratum proportionally to requested split sizes, then adjust
    # rounding to the exact cohort totals while preserving one training record.
    requested = np.array([n - n_test - n_val, n_val, n_test], dtype=int)
    assignments: list[list[Patient]] = [[], [], []]
    for bucket in buckets.values():
        count = len(bucket)
        if count == 0:
            continue
        raw = count * requested / n
        alloc = np.floor(raw).astype(int)
        remain = count - int(alloc.sum())
        for idx in np.argsort(-(raw - alloc), kind="stable")[:remain]:
            alloc[idx] += 1
        offset = 0
        for split_idx, amount in enumerate(alloc):
            assignments[split_idx].extend(bucket[offset:offset + amount])
            offset += amount
    for group in assignments:
        rng.shuffle(group)

    # Rounded proportional allocation can leave a requested partition short.
    # Move records from the largest split, retaining at least one train patient.
    for target_idx in (1, 2):
        while len(assignments[target_idx]) < requested[target_idx]:
            donors = [i for i in (0, 1, 2) if i != target_idx and len(assignments[i]) > requested[i]]
            if not donors:
                break
            donor = max(donors, key=lambda i: len(assignments[i]) - requested[i])
            assignments[target_idx].append(assignments[donor].pop())
    if not assignments[0]:
        donor = max((1, 2), key=lambda i: len(assignments[i]))
        if assignments[donor]:
            assignments[0].append(assignments[donor].pop())
    return PatientSplit(*(tuple(group) for group in assignments))


@dataclass(frozen=True)
class Preprocessor:
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def __post_init__(self) -> None:
        medians = np.asarray(self.medians, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        shape = (len(FEATURE_COLUMNS),)
        if medians.shape != shape or means.shape != shape or scales.shape != shape:
            raise ValueError(f"preprocessor statistics must each have shape {shape}")
        if not (np.isfinite(medians).all() and np.isfinite(means).all() and
                np.isfinite(scales).all() and np.all(scales > 0)):
            raise ValueError("preprocessor statistics must be finite with positive scales")
        object.__setattr__(self, "medians", medians)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)

    @classmethod
    def fit(cls, patients: Sequence[Patient]) -> "Preprocessor":
        """Fit imputation and scaling statistics using training patients only."""
        if not patients:
            raise ValueError("patients must not be empty")
        # Keep only one observed feature column at a time. A full-cohort copy
        # of the raw hours-by-features matrix adds hundreds of MB unnecessarily.
        medians = np.empty(len(FEATURE_COLUMNS), dtype=np.float64)
        means = np.empty_like(medians)
        scales = np.empty_like(medians)
        for feature in range(len(FEATURE_COLUMNS)):
            observed = np.concatenate([
                patient.values[np.isfinite(patient.values[:, feature]), feature]
                for patient in patients
            ])
            if len(observed) == 0:
                medians[feature] = means[feature] = 0.0
                scales[feature] = 1.0
            else:
                medians[feature] = np.median(observed)
                with np.errstate(over="ignore", invalid="ignore"):
                    means[feature] = np.mean(observed)
                    std = np.std(observed)
                if not np.isfinite(medians[feature]) or not np.isfinite(means[feature]):
                    raise ValueError(f"training statistic for {FEATURE_COLUMNS[feature]} is not finite")
                scales[feature] = std if std > 0 and np.isfinite(std) else 1.0
        return cls(medians, means, scales)

    def transform(self, patient: Patient) -> tuple[np.ndarray, np.ndarray]:
        """Causally forward-fill each feature, median-fill its leading gaps, scale.

        The returned observation mask reflects original observed cells, before
        either kind of imputation. No values from later hours are used.
        """
        return _transform_values(patient.values, self)


def _transform_values(values: np.ndarray, preprocessor: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(preprocessor.medians):
        raise ValueError("values has an incompatible shape")
    observed = np.isfinite(raw)
    filled = raw.copy()
    for row in range(len(filled)):
        missing = ~observed[row]
        if row:
            filled[row, missing] = filled[row - 1, missing]
        still_missing = ~np.isfinite(filled[row])
        filled[row, still_missing] = preprocessor.medians[still_missing]
    scaled = (filled - preprocessor.means) / preprocessor.scales
    if not np.isfinite(scaled).all():
        raise ValueError("preprocessing produced non-finite transformed values")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled32 = scaled.astype(np.float32)
    if not np.isfinite(scaled32).all():
        raise ValueError("preprocessing exceeds the supported float32 range")
    return scaled32, observed


def build_windows(
    patient: Patient, preprocessor: Preprocessor, window: int, horizon: int = 0,
) -> WindowBatch:
    """Build left-padded causal windows targeting the published label at t+h.

    Each anchor hour t predicts `labels[t + horizon]`. Hours with label 1 at
    t are excluded. PhysioNet 2019 labels already turn positive SIX HOURS before
    the clinical onset; a future published-label horizon therefore shifts that
    benchmark label in time and does not mean onset occurs at the future hour.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    targets: list[int] = []
    hours: list[int] = []
    for hour in range(len(patient.values) - horizon):
        # Current-label comparison includes positive anchors. Future-label
        # forecasting is pre-onset only and excludes already-positive anchors.
        if horizon > 0 and patient.labels[hour] == 1:
            continue
        start = max(0, hour - window + 1)
        segment = patient.values[start:hour + 1]
        transformed, observed = _transform_values(segment, preprocessor)
        pad = window - len(segment)
        if pad:
            transformed = np.pad(transformed, ((pad, 0), (0, 0)), constant_values=0)
            observed = np.pad(observed, ((pad, 0), (0, 0)), constant_values=False)
        valid = np.concatenate((np.zeros(pad, dtype=bool), np.ones(window - pad, dtype=bool)))
        rows.append(transformed)
        obs_rows.append(observed)
        valid_rows.append(valid)
        targets.append(int(patient.labels[hour + horizon]))
        hours.append(hour)
    shape = (0, window, len(FEATURE_COLUMNS))
    values_array = np.stack(rows).astype(np.float32) if rows else np.empty(shape, dtype=np.float32)
    obs_array = np.stack(obs_rows).astype(bool) if obs_rows else np.empty(shape, dtype=bool)
    valid_array = np.stack(valid_rows).astype(bool) if valid_rows else np.empty((0, window), dtype=bool)
    return WindowBatch(values_array, obs_array, valid_array, np.asarray(targets, dtype=np.int64),
                       tuple(patient.patient_id for _ in targets), np.asarray(hours, dtype=np.int64))


def build_dataset(
    patients: Sequence[Patient], preprocessor: Preprocessor, window: int, horizon: int = 0,
) -> WindowBatch:
    """Combine per-patient windows while preserving patient IDs and local hours."""
    batches = [build_windows(patient, preprocessor, window, horizon) for patient in patients]
    if not batches:
        raise ValueError("patients must not be empty")
    return WindowBatch(
        np.concatenate([b.values for b in batches]),
        np.concatenate([b.observed_mask for b in batches]),
        np.concatenate([b.valid_time_mask for b in batches]),
        np.concatenate([b.targets for b in batches]),
        tuple(pid for b in batches for pid in b.patient_ids),
        np.concatenate([b.hours for b in batches]),
    )

