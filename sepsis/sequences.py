"""Lazy PyTorch windows for explicitly labeled sepsis research."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Patient, Preprocessor, _transform_values
from .schema import FEATURE_COLUMNS


class SequenceDataset(Dataset):
    """Materialize one left-padded causal window at a time.

    Only eligible ``(patient index, local hour)`` pairs and target metadata are
    retained. In particular, the dataset never stores an ``N x T x F`` array.
    The tuple order and dtypes match ``temporal_train._tensors``.
    """

    def __init__(self, patients: Sequence[Patient], preprocessor: Preprocessor,
                 window: int, horizon: int = 0):
        if not patients:
            raise ValueError("patients must not be empty")
        if window <= 0:
            raise ValueError("window must be positive")
        if horizon < 0:
            raise ValueError("horizon must be nonnegative")
        self.patients = tuple(patients)
        self.preprocessor = preprocessor
        self.window = int(window)
        self.horizon = int(horizon)

        pairs = [(pi, hour)
                 for pi, patient in enumerate(self.patients)
                 for hour in range(max(0, len(patient.values) - horizon))
                 if horizon == 0 or patient.labels[hour] == 0]
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.targets = np.asarray(
            [int(self.patients[pi].labels[hour + horizon]) for pi, hour in pairs],
            dtype=np.int64,
        )
        self.patient_ids = tuple(self.patients[pi].patient_id for pi, _ in pairs)
        self.hours = np.asarray([hour for _, hour in pairs], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pi, hour = self._pairs[index]
        patient = self.patients[int(pi)]
        start = max(0, int(hour) - self.window + 1)
        values, observed = _transform_values(patient.values[start:int(hour) + 1],
                                              self.preprocessor)
        pad = self.window - len(values)
        if pad:
            values = np.pad(values, ((pad, 0), (0, 0)), constant_values=0)
            observed = np.pad(observed, ((pad, 0), (0, 0)), constant_values=False)
        valid = np.arange(self.window) >= pad
        return (
            torch.from_numpy(np.asarray(values, dtype=np.float32)),
            torch.from_numpy(np.asarray(observed, dtype=np.bool_)),
            torch.from_numpy(valid),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )
