"""Canonical PhysioNet 2019 Sepsis Challenge PSV schema and validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
)
LABEL_COLUMN = "SepsisLabel"
EXPECTED_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + (LABEL_COLUMN,)


class SchemaError(ValueError):
    """Raised when a patient file does not match the expected PSV schema."""


def validate_patient_frame(frame: pd.DataFrame, source: str | Path = "<dataframe>") -> pd.DataFrame:
    """Validate and return one patient's hourly values in canonical column order.

    Feature cells may be missing, but nonnumeric and infinite values are rejected.
    Labels must be complete binary values and, as in the challenge data, may only
    transition from 0 to 1 once (they cannot return to 0).
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    columns = list(frame.columns)
    if len(frame) == 0:
        raise SchemaError(f"{source}: patient file contains no hourly rows")
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise SchemaError(f"{source}: duplicate column names: {duplicates}")
    missing = [name for name in EXPECTED_COLUMNS if name not in columns]
    extra = [name for name in columns if name not in EXPECTED_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns {missing}")
        if extra:
            details.append(f"unexpected columns {extra}")
        raise SchemaError(f"{source}: " + "; ".join(details))

    result = frame.loc[:, EXPECTED_COLUMNS].copy()
    for name in FEATURE_COLUMNS:
        try:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{source}: feature {name!r} contains a nonnumeric value") from exc
        values = result[name].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise SchemaError(f"{source}: feature {name!r} contains infinity")

    # One row represents one ICU hour. ICULOS is the available time axis, so it
    # must be present and strictly increasing; otherwise windows cannot be
    # interpreted as causal hourly histories. Gaps are retained and reported by
    # the audit command rather than silently interpolated.
    iculos = result["ICULOS"].to_numpy(dtype=float)
    if not np.isfinite(iculos).all():
        raise SchemaError(f"{source}: ICULOS must be finite for every hourly row")
    if len(iculos) > 1 and np.any(np.diff(iculos) != 1):
        raise SchemaError(f"{source}: ICULOS must advance by one hour (no gaps, duplicate or reversed hours)")

    try:
        labels = pd.to_numeric(result[LABEL_COLUMN], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{source}: {LABEL_COLUMN} contains a nonnumeric value") from exc
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise SchemaError(f"{source}: {LABEL_COLUMN} must contain only 0/1 values")
    if np.any(np.diff(labels) < 0):
        raise SchemaError(f"{source}: {LABEL_COLUMN} must be monotonic (0s followed by 1s)")
    result[LABEL_COLUMN] = labels.astype(np.int8)
    return result


def load_patient_file(path: str | Path) -> pd.DataFrame:
    """Read and validate one pipe-separated PhysioNet patient file."""
    path = Path(path)
    try:
        frame = pd.read_csv(path, sep="|")
    except Exception as exc:
        raise SchemaError(f"{path}: could not read PSV data: {exc}") from exc
    return validate_patient_frame(frame, path)


