"""Compact patient-wise tabular summaries for research baseline training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import Patient, Preprocessor, build_windows
from .model import summarize_windows


@dataclass(frozen=True)
class TabularBatch:
    features: np.ndarray
    targets: np.ndarray
    patient_ids: tuple[str, ...]
    hours: np.ndarray

    def __len__(self) -> int:
        return len(self.targets)


def build_tabular_batch(patients: Sequence[Patient], preprocessor: Preprocessor,
                        window: int, horizon: int = 0,
                        feature_names: Sequence[str] | None = None) -> tuple[TabularBatch, list[str]]:
    """Summarize one patient's windows at a time into a single float32 matrix."""
    if not patients:
        raise ValueError("patients must not be empty")
    # Count eligible anchors first so the only cohort-sized allocation is the
    # compact feature matrix and its evaluation metadata.
    count = 0
    for patient in patients:
        anchors = max(0, len(patient.values) - horizon)
        if horizon > 0:
            anchors = sum(patient.labels[t] == 0 for t in range(anchors))
        count += anchors
    width = len(feature_names) * 8 if feature_names is not None else len(preprocessor.medians) * 8
    features = np.empty((count, width), dtype=np.float32)
    targets = np.empty(count, dtype=np.int64)
    hours = np.empty(count, dtype=np.int64)
    ids: list[str] = []
    names: list[str] | None = None
    offset = 0
    for patient in patients:
        batch = build_windows(patient, preprocessor, window, horizon)
        if not len(batch):
            continue
        summary, names = summarize_windows(batch.values, batch.observed_mask,
                                           batch.valid_time_mask, feature_names)
        end = offset + len(batch)
        features[offset:end] = summary
        targets[offset:end] = batch.targets
        hours[offset:end] = batch.hours
        ids.extend(batch.patient_ids)
        offset = end
    if names is None:
        _, names = summarize_windows(np.empty((0, window, len(preprocessor.medians)),
                                               dtype=np.float32), feature_names=feature_names)
    return TabularBatch(features, targets, tuple(ids), hours), names
