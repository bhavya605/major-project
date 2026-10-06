"""Create a fully synthetic, non-clinical PSV dataset for the first milestone."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .schema import FEATURE_COLUMNS, LABEL_COLUMN


def generate_demo(output_dir: str | Path, *, patients: int = 48,
                  hours: int = 48, seed: int = 7) -> Path:
    """Write seeded synthetic patients with a trend signal and random missingness.

    This toy simulator exists only to exercise the software pipeline. Its rows
    and labels are invented and have no clinical interpretation.
    """
    if patients < 40:
        raise ValueError("demo requires at least 40 patients")
    if hours < 18:
        raise ValueError("demo requires at least 18 hours per patient")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    event_count = max(12, int(round(patients * 0.4)))
    event_count = min(event_count, patients - 8)
    event_ids = set(rng.choice(patients, size=event_count, replace=False).tolist())
    # The first eight features carry an invented gradual trend before the
    # published-label transition; all remaining features are noisy background.
    for pid in range(patients):
        is_event = pid in event_ids
        event_label_hour = int(rng.integers(max(14, hours // 2), hours - 5)) if is_event else hours + 1
        trend_start = max(0, event_label_hour - int(rng.integers(9, 14)))
        base = rng.normal(0.0, 0.6, size=len(FEATURE_COLUMNS))
        values = rng.normal(base, 0.35, size=(hours, len(FEATURE_COLUMNS)))
        if is_event:
            ramp = np.clip((np.arange(hours) - trend_start) / max(1, event_label_hour - trend_start), 0, 1)
            for feature_index in range(8):
                direction = 1.0 if feature_index % 2 == 0 else -1.0
                values[:, feature_index] += direction * ramp * rng.uniform(1.0, 2.2)
        # Preserve the benchmark's monotonic hourly time axis. ICULOS is always
        # observed; missingness applies to the other synthetic measurements.
        values[:, FEATURE_COLUMNS.index("ICULOS")] = np.arange(1, hours + 1, dtype=float)
        # Deterministic missingness process with a modest increase late in stay.
        missing_prob = np.linspace(0.08, 0.22, hours)[:, None]
        missing = rng.random(values.shape) < missing_prob
        missing[:, FEATURE_COLUMNS.index("ICULOS")] = False
        values[missing] = np.nan
        labels = (np.arange(hours) >= event_label_hour).astype(np.int8)
        filename = root / f"demo_{pid:03d}.psv"
        with filename.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="|")
            writer.writerow((*FEATURE_COLUMNS, LABEL_COLUMN))
            for row, label in zip(values, labels):
                writer.writerow(["" if not np.isfinite(value) else f"{value:.5f}" for value in row] + [int(label)])
    manifest = {
        "source_type": "synthetic_demo",
        "synthetic": True,
        "clinical_data": False,
        "seed": seed,
        "patient_count": patients,
        "hours_per_patient": hours,
        "event_patient_count": event_count,
        "label_note": "Invented monotonic target used only to exercise software; it makes no clinical claim.",
    }
    (root / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo", help="Directory to create synthetic .psv files in")
    parser.add_argument("--patients", type=int, default=48)
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    path = generate_demo(args.output, patients=args.patients, hours=args.hours, seed=args.seed)
    print(f"Wrote {args.patients} synthetic patients to {path.resolve()} (software demo only; not clinical data).")


if __name__ == "__main__":
    main()
