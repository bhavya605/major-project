"""Streaming exploratory summaries for ICU sepsis research data.

All outputs are descriptive research artifacts, not clinical guidance or model
performance. Patient PSV files are processed in chunks; no values are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from .audit import _source, _is_synthetic
from .schema import FEATURE_COLUMNS, LABEL_COLUMN, validate_patient_frame

VITALS = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp")
RELATIVE_HOURS = tuple(range(-24, 25))
# Broad descriptive screening bounds; these are not clinical decision thresholds.
FLAG_RANGES = {
    "HR": (30, 220), "O2Sat": (70, 100), "Temp": (30, 43),
    "SBP": (50, 250), "MAP": (30, 180), "DBP": (20, 150),
    "Resp": (4, 60), "EtCO2": (5, 100),
}
OFFICIAL_FILE_BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_selected_files(root: Path, files: list[dict[str, Any]]) -> bool:
    """Verify selected file paths/hashes against entries in the acquisition manifest."""
    manifest_path = root / "download_manifest.json"
    if not manifest_path.is_file() or not files:
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("source_type") != "physionet_public_training_data" or
                manifest.get("official_file_base") != OFFICIAL_FILE_BASE):
            return False
        expected = {entry["path"]: entry for entry in manifest.get("files", [])}
        for item in files:
            entry = expected.get(item["relative_path"])
            if not entry or entry.get("sha256") != item["sha256"]:
                return False
            parsed = urlparse(entry.get("url", ""))
            expected_path = urlparse(OFFICIAL_FILE_BASE).path + item["relative_path"]
            if parsed.scheme != "https" or parsed.netloc != "physionet.org" or parsed.path != expected_path:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _svg_plot(path: Path, title: str, labels: list[str], values: list[float],
              *, y_label: str, series: list[tuple[str, list[float]]] | None = None) -> None:
    """Write a small, dependency-free SVG bar or multi-line plot."""
    width, height = 1000, 600
    left, right, top, bottom = 90, 30, 70, 160
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = values if series is None else [v for _, vs in series for v in vs if np.isfinite(v)]
    ymax = max(all_values, default=1.0)
    if ymax <= 0:
        ymax = 1.0
    ymax *= 1.08
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="sans-serif">{escape(title)}</text>',
             f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-size="14" font-family="sans-serif">{escape(y_label)}</text>']
    for i in range(6):
        yv = ymax * i / 5
        y = top + plot_h - plot_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{yv:.2g}</text>')
    if series is None:
        n = max(1, len(values)); slot = plot_w / n; bw = max(1, slot * .8)
        for i, value in enumerate(values):
            x = left + i * slot + (slot - bw) / 2
            bh = max(0, value / ymax * plot_h)
            parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#4c78a8"/>')
            if len(labels) <= 45 or i % max(1, len(labels)//30) == 0:
                lx, ly = x+bw/2, top+plot_h+12
                parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" transform="rotate(-55 {lx:.1f} {ly:.1f})" text-anchor="end" font-size="9" font-family="sans-serif">{escape(labels[i])}</text>')
    else:
        colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6")
        n = max(1, len(labels))
        for si, (name, vals) in enumerate(series):
            segments: list[list[str]] = [[]]
            for i, val in enumerate(vals):
                if np.isfinite(val):
                    x = left + (i / max(1, n-1)) * plot_w
                    y = top + plot_h - val / ymax * plot_h
                    segments[-1].append(f'{x:.1f},{y:.1f}')
                elif segments[-1]:
                    segments.append([])
            for points in segments:
                if points:
                    parts.append(f'<polyline fill="none" stroke="{colors[si % len(colors)]}" stroke-width="2" points="{" ".join(points)}"/>')
            parts.append(f'<text x="{left+10+si*105}" y="{height-15}" font-size="12" fill="{colors[si % len(colors)]}" font-family="sans-serif">{escape(name)}</text>')
        for i, label in enumerate(labels):
            if i % 4 == 0:
                x = left + (i / max(1, n-1)) * plot_w
                parts.append(f'<text x="{x:.1f}" y="{top+plot_h+18}" text-anchor="middle" font-size="10" font-family="sans-serif">{escape(label)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def explore_dataset(directory: str | Path, output_dir: str | Path,
                    *, chunk_rows: int = 4096, split_manifest: str | Path | None = None,
                    partition: str | None = None) -> dict[str, Any]:
    """Stream patient PSV files and write research cohort summaries and plots."""
    root, out = Path(directory), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    files = sorted(root.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"no .psv files found in {root}")
    if (split_manifest is None) != (partition is None):
        raise ValueError("--split-manifest and --partition must be supplied together")
    selected_ids: set[str] | None = None
    if split_manifest is not None:
        split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        partitions = split.get("patient_ids")
        if not isinstance(partitions, dict) or partition not in partitions:
            raise ValueError(f"split manifest has no patient_ids[{partition!r}]")
        flattened = [pid for ids in partitions.values() for pid in ids]
        if len(flattened) != len(set(flattened)):
            raise ValueError("split manifest assigns a patient ID more than once")
        available_ids = [p.stem for p in files]
        if len(available_ids) != len(set(available_ids)):
            raise ValueError("dataset contains duplicate patient IDs")
        unknown_manifest = sorted(set(flattened) - set(available_ids))
        unassigned = sorted(set(available_ids) - set(flattened))
        if unknown_manifest or unassigned:
            raise ValueError(f"split/dataset membership mismatch; unknown_manifest_ids={unknown_manifest}; unassigned_dataset_ids={unassigned}")
        selected_ids = set(partitions[partition])
        files = [p for p in files if p.stem in selected_ids]
    out.mkdir(parents=True, exist_ok=True)
    missing: Counter[str] = Counter(); cells: Counter[str] = Counter()
    source_files: Counter[str] = Counter(); source_rows: Counter[str] = Counter()
    label_rows: Counter[str] = Counter(); source_patients: Counter[str] = Counter()
    source_labels: Counter[tuple[str, str]] = Counter()
    lengths: list[int] = []; file_summaries: list[dict[str, Any]] = []
    admission_positive_patients = 0
    source_admission_positive: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    traj_sum = {name: np.zeros(len(RELATIVE_HOURS), dtype=float) for name in VITALS}
    traj_count = {name: np.zeros(len(RELATIVE_HOURS), dtype=np.int64) for name in VITALS}
    for path in files:
        source = _source(path); source_files[source] += 1
        file_hash = _sha256(path)
        rows = pos = 0; first_positive: float | None = None; first_time: float | None = None
        previous_label: int | None = None; previous_time: float | None = None
        local_sum = {name: np.zeros(len(RELATIVE_HOURS), dtype=float) for name in VITALS}
        local_count = {name: np.zeros(len(RELATIVE_HOURS), dtype=np.int64) for name in VITALS}
        prior_times: list[float] = []
        prior_values: dict[str, list[float]] = {name: [] for name in VITALS}

        def accumulate(times: np.ndarray, values_by_name: dict[str, np.ndarray]) -> None:
            offsets = np.rint(times - first_positive).astype(int)
            keep = (offsets >= RELATIVE_HOURS[0]) & (offsets <= RELATIVE_HOURS[-1])
            indexes = offsets[keep] - RELATIVE_HOURS[0]
            for vital in VITALS:
                vals = values_by_name[vital][keep]
                finite = np.isfinite(vals)
                np.add.at(local_sum[vital], indexes[finite], vals[finite])
                np.add.at(local_count[vital], indexes[finite], 1)

        try:
            with pd.read_csv(path, sep="|", chunksize=chunk_rows) as chunks:
                for block in chunks:
                    frame = validate_patient_frame(block, path)
                    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
                    times = frame["ICULOS"].to_numpy(dtype=float)
                    if first_time is None and len(times):
                        first_time = float(times[0])
                    if previous_label == 1 and len(labels) and labels[0] == 0:
                        raise ValueError(f"{path}: SepsisLabel decreases across read chunks")
                    if previous_time is not None and len(times) and times[0] - previous_time != 1:
                        raise ValueError(f"{path}: ICULOS must advance one hour across read chunks")
                    if len(labels):
                        previous_label = int(labels[-1])
                        previous_time = float(times[-1])
                    found_reference = first_positive is None and labels.any()
                    if found_reference:
                        first_positive = float(times[np.flatnonzero(labels)[0]])
                    pos += int(labels.sum()); rows += len(frame)
                    source_labels[(source, "positive")] += int(labels.sum())
                    source_labels[(source, "negative")] += int(len(labels) - labels.sum())
                    for name in FEATURE_COLUMNS:
                        absent = int(frame[name].isna().sum())
                        missing[name] += absent; cells[name] += len(frame)
                    for name, (low, high) in FLAG_RANGES.items():
                        values = frame[name].to_numpy(dtype=float)
                        flag_counts[name] += int(np.sum(np.isfinite(values) & ((values < low) | (values > high))))
                    values_by_vital = {name: frame[name].to_numpy(dtype=float) for name in VITALS}
                    if found_reference:
                        if prior_times:
                            accumulate(np.asarray(prior_times, dtype=float),
                                       {name: np.asarray(vals, dtype=float) for name, vals in prior_values.items()})
                            prior_times.clear()
                            for vals in prior_values.values():
                                vals.clear()
                        # The reference is now known, so include all rows from this
                        # chunk; pre-positive rows here belong on negative lags.
                        accumulate(times, values_by_vital)
                    elif first_positive is None:
                        # Keep at most the 24 preceding hourly observations needed by the plot.
                        prior_times.extend(times.tolist())
                        for name in VITALS:
                            prior_values[name].extend(values_by_vital[name].tolist())
                        if len(prior_times) > 24:
                            trim = len(prior_times) - 24
                            del prior_times[:trim]
                            for vals in prior_values.values():
                                del vals[:trim]
                    else:
                        accumulate(times, values_by_vital)
            source_rows[source] += rows; label_rows["positive"] += pos; label_rows["negative"] += rows-pos
            source_patients[source] += 1; lengths.append(rows)
            admission_positive = bool(first_positive is not None and first_positive == first_time)
            if admission_positive:
                admission_positive_patients += 1
                source_admission_positive[source] += 1
            if first_positive is not None and not admission_positive:
                for name in VITALS:
                    traj_sum[name] += local_sum[name]; traj_count[name] += local_count[name]
            file_summaries.append({"patient_id": path.stem, "relative_path": path.relative_to(root).as_posix(),
                                   "source": source, "rows": rows, "positive_label_rows": pos,
                                   "has_positive_label": bool(pos), "first_positive_iculos": first_positive,
                                   "sha256": file_hash,
                                   "admission_positive": admission_positive})
        except Exception as exc:
            file_summaries.append({"patient_id": path.stem, "relative_path": path.relative_to(root).as_posix(),
                                   "source": source, "rows": rows, "sha256": file_hash, "error": str(exc)})
    valid = [f for f in file_summaries if "error" not in f]
    nrows = sum(f["rows"] for f in valid)
    lengths.sort()
    synthetic = _is_synthetic(root)
    verified_source = _verified_selected_files(root, valid)
    provenance = "synthetic_demo" if synthetic else "physionet_public_training_data" if verified_source else "unknown"
    report: dict[str, Any] = {
        "report_type": "research_exploratory_data_analysis", "research_only": True,
        "source_type": provenance, "source_provenance_verified": bool(synthetic or verified_source),
        "dataset_root": str(root.resolve()), "file_count": len(files), "valid_file_count": len(valid),
        "invalid_file_count": len(files)-len(valid), "source_patient_counts": dict(sorted(source_patients.items())),
        "source_row_counts": dict(sorted(source_rows.items())), "row_count": nrows,
        "label_row_counts": dict(sorted(label_rows.items())),
        "positive_label_rate": label_rows["positive"]/nrows if nrows else None,
        "positive_patient_count": sum(bool(f.get("has_positive_label")) for f in valid),
        "admission_positive_patient_count": admission_positive_patients,
        "source_admission_positive_patient_counts": dict(sorted(source_admission_positive.items())),
        "transition_trajectory_patient_count": sum(bool(f.get("has_positive_label")) and not f.get("admission_positive") for f in valid),
        "patient_length_rows": {"min": min(lengths) if lengths else 0,
                                 "median": float(np.median(lengths)) if lengths else None,
                                 "max": max(lengths) if lengths else 0},
        "feature_missingness": {name: {"missing": int(missing[name]), "cells": int(cells[name]),
                                      "rate": missing[name]/cells[name] if cells[name] else None}
                                for name in FEATURE_COLUMNS},
        "suspicious_reading_flags": {name: {"below_or_above_range_count": int(flag_counts[name]),
                                            "descriptive_range": list(bounds),
                                            "interpretation": "broad descriptive data-quality screen; not a clinical rule; values are retained"}
                                     for name, bounds in FLAG_RANGES.items()},
        "trajectory_definition": {"reference": "first row with SepsisLabel=1 in each patient",
                                  "window_hours": [RELATIVE_HOURS[0], RELATIVE_HOURS[-1]],
                                  "clinical_onset": False,
                                  "note": "PhysioNet 2019 SepsisLabel already begins six hours before recorded clinical onset; these trajectories are relative to the first published-positive label and do not identify clinical onset. Patients positive on their first ICU row are counted separately and excluded from transition trajectories."},
        "selected_partition": partition, "selected_patient_ids": sorted(selected_ids) if selected_ids is not None else sorted(f["patient_id"] for f in valid),
        "test_partition_used": partition == "test" if partition is not None else None, "files": file_summaries,
    }
    report["trajectory_mean_by_relative_hour"] = {
        name: {str(hour): float(traj_sum[name][i]/traj_count[name][i]) if traj_count[name][i] else None
               for i, hour in enumerate(RELATIVE_HOURS)} for name in VITALS}
    report["trajectory_sample_count_by_relative_hour"] = {
        name: {str(hour): int(traj_count[name][i]) for i, hour in enumerate(RELATIVE_HOURS)} for name in VITALS}
    (out / "eda.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "missingness.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("feature", "missing", "cells", "missing_rate"))
        for name, entry in report["feature_missingness"].items():
            writer.writerow((name, entry["missing"], entry["cells"], entry["rate"]))
    with (out / "label_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("source", "label", "rows"))
        for source in sorted(source_files):
            for label in ("negative", "positive"):
                writer.writerow((source, label, source_labels[(source, label)]))
        for label, count in sorted(label_rows.items()):
            writer.writerow(("all", label, count))
    _svg_plot(out / "missingness.svg", "Research EDA: feature missingness", list(FEATURE_COLUMNS),
              [report["feature_missingness"][n]["rate"] or 0 for n in FEATURE_COLUMNS], y_label="Missing fraction")
    with (out / "trajectory_mean.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("relative_hour_from_first_positive_label", *(f"{vital}_mean" for vital in VITALS), *(f"{vital}_n" for vital in VITALS)))
        for i, hour in enumerate(RELATIVE_HOURS):
            writer.writerow((hour, *(report["trajectory_mean_by_relative_hour"][n][str(hour)] for n in VITALS),
                             *(report["trajectory_sample_count_by_relative_hour"][n][str(hour)] for n in VITALS)))
    for vital in VITALS:
        _svg_plot(out / f"vital_trajectory_{vital}.svg", f"Research EDA: {vital} mean by first positive label",
                  [str(h) for h in RELATIVE_HOURS], [], y_label=f"Mean {vital} (source units)",
                  series=[(vital, [report["trajectory_mean_by_relative_hour"][vital][str(h)] if report["trajectory_mean_by_relative_hour"][vital][str(h)] is not None else float("nan") for h in RELATIVE_HOURS])])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing patient PSV files")
    parser.add_argument("--output-dir", required=True, help="Directory for research EDA artifacts")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--split-manifest", help="Locked JSON patient split manifest")
    parser.add_argument("--partition", choices=("train", "validation", "test"), help="Partition to select from split manifest")
    args = parser.parse_args()
    report = explore_dataset(args.data, args.output_dir, chunk_rows=args.chunk_rows,
                             split_manifest=args.split_manifest, partition=args.partition)
    print(json.dumps({k: report[k] for k in ("report_type", "source_type", "file_count", "valid_file_count",
                                              "invalid_file_count", "source_patient_counts", "row_count",
                                              "label_row_counts", "positive_patient_count", "admission_positive_patient_count",
                                              "transition_trajectory_patient_count", "selected_partition", "test_partition_used")}, indent=2))


if __name__ == "__main__":
    main()
