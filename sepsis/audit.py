"""Streaming integrity and descriptive audit for PSV research datasets.

All results are data-quality summaries, not model or clinical performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .schema import EXPECTED_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMN, SchemaError, validate_patient_frame


def _source(path: Path) -> str:
    for parent in path.parents:
        name = parent.name.lower()
        if name == "training_seta":
            return "A"
        if name == "training_setb":
            return "B"
    return "unknown"


def _is_synthetic(root: Path) -> bool:
    return (root / "demo_manifest.json").is_file()


def _verified_acquisition(root: Path, files: list[dict[str, Any]]) -> bool:
    """Recognize only PSV files checksum-matched to this downloader's manifest."""
    manifest_path = root / "download_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_type") != "physionet_public_training_data":
            return False
        if manifest.get("official_file_base") != "https://physionet.org/files/challenge-2019/1.0.0/training/":
            return False
        expected = {entry["path"]: entry for entry in manifest.get("files", [])}
        if len(expected) != len(files):
            return False
        for item in files:
            entry = expected.get(item["relative_path"])
            if not entry or entry.get("sha256") != item["sha256"]:
                return False
            url = entry.get("url", "")
            parsed = urlparse(url)
            expected_url_path = urlparse(manifest["official_file_base"]).path + item["relative_path"]
            if (parsed.scheme != "https" or parsed.netloc != "physionet.org" or
                    parsed.path != expected_url_path):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def audit_dataset(directory: str | Path, output_dir: str | Path | None = None,
                  *, chunk_rows: int = 4096) -> dict[str, Any]:
    """Audit each PSV in chunks and optionally write JSON and missingness CSV.

    The JSON contains a SHA-256 per file, source counts, row/label counts,
    length summaries, data-quality findings, and provenance. Invalid files are
    counted and described rather than being silently omitted.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    files = sorted(root.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"no .psv files found in {root}")

    feature_missing: Counter[str] = Counter()
    feature_cells: Counter[str] = Counter()
    by_hour: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    per_file: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    patient_ids: dict[str, list[str]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    lengths: list[int] = []
    total_rows = positive_rows = positive_patients = 0

    for path in files:
        rel = path.relative_to(root).as_posix()
        pid = path.stem
        source = _source(path)
        source_counts[source] += 1
        patient_ids[pid].append(rel)
        file_hash = hashlib.sha256()
        rows = pos = 0
        gap_hours = 0
        min_hour: float | None = None
        max_hour: float | None = None
        previous_label: int | None = None
        previous_hour: float | None = None
        error: str | None = None
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_hash.update(block)
            # Inspect the raw header before pandas can rename duplicate labels.
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                header = next(csv.reader(stream, delimiter="|"), [])
            if len(set(header)) != len(header):
                raise SchemaError(f"duplicate column names: {[c for c in header if header.count(c) > 1]}")
            missing = [c for c in EXPECTED_COLUMNS if c not in header]
            extra = [c for c in header if c not in EXPECTED_COLUMNS]
            if missing or extra:
                raise SchemaError(f"schema mismatch; missing={missing}; extra={extra}")
            with pd.read_csv(path, sep="|", chunksize=chunk_rows) as chunks:
                for chunk in chunks:
                    frame = validate_patient_frame(chunk, path)
                    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
                    hours = frame["ICULOS"].to_numpy(dtype=float)
                    if previous_label == 1 and len(labels) and labels[0] == 0:
                        raise SchemaError(f"{path}: SepsisLabel decreases across read chunks")
                    if previous_hour is not None and len(hours):
                        delta = float(hours[0] - previous_hour)
                        if delta != 1:
                            raise SchemaError(f"{path}: ICULOS must advance one hour across read chunks")
                        if delta > 1:
                            gap_hours += max(1, int(np.ceil(delta)) - 1)
                    if len(hours) > 1:
                        gap_hours += int(np.sum(np.diff(hours) > 1))
                    previous_label = int(labels[-1]) if len(labels) else previous_label
                    previous_hour = float(hours[-1]) if len(hours) else previous_hour
                    if len(hours):
                        min_hour = float(hours[0]) if min_hour is None else min(min_hour, float(hours[0]))
                        max_hour = float(hours[-1]) if max_hour is None else max(max_hour, float(hours[-1]))
                    pos += int(labels.sum())
                    for name in FEATURE_COLUMNS:
                        observed = int(frame[name].notna().sum())
                        missing_count = len(frame) - observed
                        feature_cells[name] += len(frame)
                        feature_missing[name] += missing_count
                    # Missingness by observed ICU hour; fractional LOS is reported
                    # separately and binned to its nearest hour only for this table.
                    bins = np.rint(hours).astype(int)
                    unique_bins, inverse, bin_counts = np.unique(
                        bins, return_inverse=True, return_counts=True)
                    absent = frame[list(FEATURE_COLUMNS)].isna().to_numpy(dtype=np.int64)
                    missing_by_bin = np.zeros((len(unique_bins), len(FEATURE_COLUMNS)), dtype=np.int64)
                    np.add.at(missing_by_bin, inverse, absent)
                    for feature, name in enumerate(FEATURE_COLUMNS):
                        for index, hour_bin in enumerate(unique_bins):
                            counts = by_hour[(name, int(hour_bin))]
                            counts[0] += int(missing_by_bin[index, feature])
                            counts[1] += int(bin_counts[index])
                    rows += len(frame)
        except Exception as exc:
            error = str(exc)
            issues.append({"file": rel, "error": error})
        total_rows += rows
        positive_rows += pos
        if pos:
            positive_patients += 1
        lengths.append(rows)
        per_file.append({
            "patient_id": pid, "relative_path": rel, "source": source,
            "rows": rows, "positive_label_rows": pos,
            "has_positive_label": bool(pos), "first_iculos": min_hour,
            "last_iculos": max_hour, "hour_gaps": gap_hours,
            "sha256": file_hash.hexdigest(), "valid": error is None,
            "error": error,
        })

    duplicates = {pid: paths for pid, paths in patient_ids.items() if len(paths) > 1}
    for pid, paths in duplicates.items():
        issues.append({"file": ", ".join(paths), "error": f"duplicate patient ID {pid!r}"})
    nvalid = sum(bool(item["valid"]) for item in per_file)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    synthetic = _is_synthetic(root)
    official_download = not synthetic and _verified_acquisition(root, per_file)
    report: dict[str, Any] = {
        "report_type": "research_dataset_integrity_audit",
        "source_type": ("synthetic_demo" if synthetic else
                        "physionet_public_training_data" if official_download else "unknown"),
        "dataset_root": str(root.resolve()),
        "provenance": {
            "official_reference": "https://physionet.org/content/challenge-2019/1.0.0/",
            "source_provenance_verified": bool(synthetic or official_download),
            "note": "Source is unknown unless a synthetic demo marker exists or official acquisition manifest hashes match every PSV file.",
        },
        "file_count": len(files), "valid_file_count": nvalid,
        "invalid_file_count": len(files) - nvalid,
        "source_file_counts": dict(sorted(source_counts.items())),
        "patient_id_duplicate_count": len(duplicates), "duplicate_patient_ids": duplicates,
        "row_count": total_rows, "positive_label_rows": positive_rows,
        "positive_label_rate": (positive_rows / total_rows) if total_rows else None,
        "positive_patient_count": positive_patients,
        "patient_length_rows": {
            "min": int(lengths_arr.min()), "median": float(np.median(lengths_arr)),
            "max": int(lengths_arr.max()),
        },
        "feature_missingness": {
            name: {"missing": int(feature_missing[name]), "cells": int(feature_cells[name]),
                  "rate": float(feature_missing[name] / feature_cells[name]) if feature_cells[name] else None}
            for name in FEATURE_COLUMNS
        },
        "quality_issue_count": len(issues), "quality_issues": issues,
        "files": per_file,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with (out / "missingness_by_feature_hour.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("feature", "iculos_hour_rounded", "missing_count", "row_count", "missing_rate"))
            for (name, hour), (missing_count, count) in sorted(by_hour.items()):
                writer.writerow((name, hour, missing_count, count, missing_count / count if count else ""))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing PSV files, including source subfolders")
    parser.add_argument("--output-dir", help="Optional directory for audit.json and missingness CSV")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    report = audit_dataset(args.data, args.output_dir, chunk_rows=args.chunk_rows)
    # Keep CLI stdout compact; the full per-file checksums are saved on request.
    summary = {key: report[key] for key in (
        "report_type", "source_type", "file_count", "valid_file_count",
        "invalid_file_count", "source_file_counts", "row_count",
        "positive_label_rows", "positive_patient_count", "quality_issue_count",
    )}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
