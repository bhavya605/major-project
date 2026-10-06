"""Create deterministic, patient-level research partitions from a validated audit."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PARTITIONS = ("train", "validation", "test")
FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
FULL_SOURCE_COUNTS = {"A": 20336, "B": 20000}


def _read_audit(path: Path, require_full_release: bool) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw_bytes = path.read_bytes()
    try:
        audit = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("audit is not valid JSON") from exc
    if audit.get("report_type") != "research_dataset_integrity_audit":
        raise ValueError("audit report_type is missing or unsupported")
    files = audit.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("audit must contain a nonempty per-file files list")
    if audit.get("file_count") != len(files) or audit.get("valid_file_count") != len(files):
        raise ValueError("audit file counts are incomplete or inconsistent")
    if audit.get("invalid_file_count") != 0 or audit.get("quality_issue_count") != 0:
        raise ValueError("audit contains invalid files or quality issues")
    if audit.get("patient_id_duplicate_count") != 0:
        raise ValueError("audit contains duplicate patient IDs")
    ids: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or item.get("valid") is not True:
            raise ValueError("audit contains an invalid or malformed file entry")
        pid, source = item.get("patient_id"), item.get("source")
        if not isinstance(pid, str) or not pid.strip() or pid in ids:
            raise ValueError("audit patient IDs are missing, empty, or duplicated")
        if source not in {"A", "B", "unknown"}:
            raise ValueError(f"invalid source for patient {pid}")
        if not isinstance(item.get("has_positive_label"), bool):
            raise ValueError(f"missing positive-label status for patient {pid}")
        ids.add(pid)
    if require_full_release:
        provenance = audit.get("provenance")
        if (audit.get("source_type") != "physionet_public_training_data" or
                not isinstance(provenance, dict) or
                provenance.get("source_provenance_verified") is not True or
                audit.get("file_count") != audit.get("valid_file_count")):
            raise ValueError("--require-full-release requires verified official provenance and a complete valid audit")
        if audit.get("source_file_counts") != FULL_SOURCE_COUNTS:
            raise ValueError("--require-full-release requires exactly 20,336 source A and 20,000 source B patients")
        root_value = audit.get("dataset_root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError("--require-full-release requires the audited dataset_root")
        acquisition_path = Path(root_value) / "download_manifest.json"
        try:
            acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("--require-full-release requires a complete acquisition manifest under dataset_root") from exc
        if (acquisition.get("report_type") != "research_dataset_acquisition_manifest" or
                acquisition.get("source_type") != "physionet_public_training_data" or
                acquisition.get("complete") is not True or
                acquisition.get("is_full_selected_sources") is not True or
                acquisition.get("requested_limit_per_source") is not None or
                acquisition.get("source_file_counts") != FULL_SOURCE_COUNTS or
                len(acquisition.get("files", [])) != sum(FULL_SOURCE_COUNTS.values())):
            raise ValueError("--require-full-release acquisition manifest is incomplete or describes a limited/subset download")
    return audit, files, hashlib.sha256(raw_bytes).hexdigest()


def _counts_to_total(n: int) -> list[int]:
    """Largest-remainder allocation gives exact cohort size and near 70/15/15."""
    raw = [n * FRACTIONS[name] for name in PARTITIONS]
    counts = [math.floor(v) for v in raw]
    for i in sorted(range(3), key=lambda j: (raw[j] - counts[j], -j), reverse=True)[:n - sum(counts)]:
        counts[i] += 1
    return counts


def create_partition(audit_path: str | Path, output_path: str | Path, *,
                     random_state: int = 42, require_full_release: bool = False) -> dict[str, Any]:
    """Build a patient split, preserving source/positive-status balance where feasible."""
    audit_path, output_path = Path(audit_path), Path(output_path)
    audit, files, audit_hash = _read_audit(audit_path, require_full_release)
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for item in files:
        strata[(item["source"], item["has_positive_label"])].append(item["patient_id"])

    groups: dict[str, list[str]] = {name: [] for name in PARTITIONS}
    fallback: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    target_totals = _counts_to_total(len(files))
    prepared: list[tuple[tuple[str, bool], np.ndarray, list[int], list[float], int]] = []
    base_totals = [0, 0, 0]
    for (source, positive), ids in sorted(strata.items()):
        shuffled = np.asarray(sorted(ids), dtype=object)
        rng.shuffle(shuffled)
        raw = [len(ids) * FRACTIONS[name] for name in PARTITIONS]
        allocations = [math.floor(v) for v in raw]
        for i in range(3):
            base_totals[i] += allocations[i]
        prepared.append(((source, positive), shuffled, allocations,
                         [raw[i] - allocations[i] for i in range(3)], len(ids) - sum(allocations)))

    # Allocate each stratum's rounding remainder jointly. This guarantees exact
    # global totals even when uneven strata would otherwise fill a target early.
    required = tuple(target_totals[i] - base_totals[i] for i in range(3))
    choices_by_row: list[list[tuple[int, ...]]] = []
    for _, _, _, remainders, left in prepared:
        choices = list(itertools.combinations(range(3), left))
        rng.shuffle(choices)
        choices.sort(key=lambda combo: sum(remainders[i] for i in combo), reverse=True)
        choices_by_row.append(choices)

    @lru_cache(None)
    def allocate_row(row: int, remaining: tuple[int, int, int]) -> tuple[tuple[int, ...], ...] | None:
        if row == len(prepared):
            return () if remaining == (0, 0, 0) else None
        for combo in choices_by_row[row]:
            next_remaining = list(remaining)
            for i in combo:
                next_remaining[i] -= 1
            if min(next_remaining) < 0:
                continue
            tail = allocate_row(row + 1, tuple(next_remaining))
            if tail is not None:
                return (combo,) + tail
        return None

    selected = allocate_row(0, required)
    if selected is None:
        raise ValueError("could not reconcile stratum rounding to exact global partition totals")
    for ((source, positive), shuffled, allocations, _, _), combo in zip(prepared, selected):
        for i in combo:
            allocations[i] += 1
        if len(shuffled) < 3:
            fallback.append({"source": source, "has_positive_label": positive,
                             "patient_count": len(shuffled), "reason": "stratum_too_small_for_all_three_partitions"})
        cursor = 0
        for name, count in zip(PARTITIONS, allocations):
            groups[name].extend(str(v) for v in shuffled[cursor:cursor + count])
            cursor += count
    for name in PARTITIONS:
        rng.shuffle(groups[name])

    source_status_counts: dict[str, dict[str, dict[str, int]]] = {}
    metadata = {item["patient_id"]: item for item in files}
    for partition, pids in groups.items():
        summary: dict[str, dict[str, int]] = {}
        for pid in pids:
            item = metadata[pid]
            key = f"{item['source']}:{'positive' if item['has_positive_label'] else 'negative'}"
            summary.setdefault(key, {"patients": 0})["patients"] += 1
        source_status_counts[partition] = summary
    manifest: dict[str, Any] = {
        "research": True,
        "report_type": "research_patient_partition_manifest",
        "random_state": int(random_state),
        "fractions": FRACTIONS,
        "patient_ids": groups,
        "patient_counts": {name: len(groups[name]) for name in PARTITIONS},
        "source_status_counts": source_status_counts,
        "stratification": "joint source and positive-label status; small strata may not occur in every partition",
        "stratification_fallbacks": fallback,
        "provenance": {"audit_path": str(audit_path.resolve()), "audit_sha256": audit_hash,
                       "source_type": audit.get("source_type", "unknown"),
                       "source_provenance_verified": audit.get("provenance", {}).get("source_provenance_verified", False),
                       "file_count": len(files)},
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("patient_ids") != groups:
            raise FileExistsError(f"refusing to overwrite a different patient split: {output_path}")
        return existing
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, help="Completed audit.json")
    parser.add_argument("--output", required=True, help="Destination patient split JSON")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--require-full-release", action="store_true",
                        help="Require verified official PhysioNet provenance")
    args = parser.parse_args()
    result = create_partition(args.audit, args.output, random_state=args.random_state,
                              require_full_release=args.require_full_release)
    print(json.dumps({"research": True, "output": str(Path(args.output).resolve()),
                      "patient_counts": result["patient_counts"],
                      "audit_sha256": result["provenance"]["audit_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
