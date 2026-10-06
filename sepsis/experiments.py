"""Run reproducible validation-only model comparisons on a shared patient split."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def run_experiments(data, output, source_type, *, models, windows=(12,), horizons=(0,), epochs=12,
                    values_only=False, loss="weighted_bce", focal_gamma=2.0):
    """Train candidates with validation-only selection; never pass --evaluate-test."""
    allowed = {"lr", "rf", "xgboost", "gru", "lstm", "transformer"}
    if not models or not set(models) <= allowed:
        raise ValueError("Select one or more supported model families")
    if values_only and not set(models) <= {"gru", "lstm", "transformer"}:
        raise ValueError("--values-only comparisons apply to temporal model families")
    if loss not in {"weighted_bce", "focal"} or focal_gamma < 0:
        raise ValueError("loss must be weighted_bce or focal; focal_gamma must be nonnegative")
    if loss == "focal" and not set(models) <= {"gru", "lstm", "transformer"}:
        raise ValueError("Focal loss comparisons apply to temporal model families")
    if source_type not in {"physionet", "synthetic_demo"}:
        raise ValueError("Explicit data source type is required")
    if any(w <= 0 for w in windows) or any(h < 0 for h in horizons) or epochs <= 0:
        raise ValueError("Windows/epochs must be positive and horizons nonnegative")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = Path(data).resolve()
    manifests = [data / "demo_manifest.json", data / "download_manifest.json"]
    provenance = next((p for p in manifests if p.is_file()), None)
    if provenance is None:
        raise ValueError("Comparison runs require a dataset provenance manifest")
    provenance_hash = hashlib.sha256(provenance.read_bytes()).hexdigest()
    implementation = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(source.name.encode())
        implementation.update(source.read_bytes())
    implementation_hash = implementation.hexdigest()
    source_manifest = json.loads(provenance.read_text(encoding="utf-8"))
    if "complete" in source_manifest and not source_manifest["complete"]:
        raise ValueError("Acquisition is incomplete; finish downloading before freezing experiments")
    records = []
    report = {"research": True, "data": str(data), "source_type": source_type,
              "provenance_sha256": provenance_hash, "implementation_sha256": implementation_hash,
              "test_evaluated": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "runs": records}
    split = root / "patient_split.json"
    for window in windows:
        for horizon in horizons:
            for family in models:
                temporal = family in {"gru", "lstm", "transformer"}
                name = f"{family}_w{window}_h{horizon}" + ("_values_only" if values_only else "")
                if temporal and loss == "focal":
                    name += f"_focal_g{focal_gamma:g}"
                artifact = root / (name + (".pt" if temporal else ".joblib"))
                command = [sys.executable, "-m", "sepsis.temporal_train" if temporal else "sepsis.train",
                           "--data", str(data), "--output", str(artifact), "--model", family,
                           "--window", str(window), "--horizon", str(horizon),
                           "--source-type", source_type, "--split-manifest", str(split)]
                if temporal:
                    command += ["--epochs", str(epochs)]
                    if values_only:
                        command += ["--values-only"]
                    if loss == "focal":
                        command += ["--loss", "focal", "--focal-gamma", str(focal_gamma)]
                signature = hashlib.sha256(json.dumps({"command": command, "data": provenance_hash,
                                                       "implementation": implementation_hash}).encode()).hexdigest()
                marker = root / (name + ".run.json")
                previous = json.loads(marker.read_text()) if marker.is_file() else {}
                record = {"name": name, "model": family, "window": window, "horizon": horizon,
                          "input_representation": "values_only" if values_only else "values_and_observation_mask",
                          "artifact": str(artifact), "command": command, "signature": signature}
                records.append(record)
                metrics_path = artifact.with_suffix(artifact.suffix + ".metrics.json")
                if (previous.get("signature") == signature and previous.get("status") == "complete"
                    and artifact.is_file() and metrics_path.is_file()
                    and previous.get("artifact_sha256") == hashlib.sha256(artifact.read_bytes()).hexdigest()):
                    record.update(status="complete", resumed=True)
                else:
                    record.update(status="running", resumed=False)
                    atomic_json(root / "comparison.json", report)
                    print(f"Training {name}; test partition remains locked", flush=True)
                    with (root / (name + ".log")).open("w", encoding="utf-8") as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    record["status"] = "complete" if result.returncode == 0 else "failed"
                    record["exit_code"] = result.returncode
                    atomic_json(marker, record)
                    if result.returncode:
                        atomic_json(root / "comparison.json", report)
                        raise RuntimeError(f"{name} failed; inspect {root / (name + '.log')}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics.get("test_metrics") is not None:
                    raise ValueError("Validation comparison encountered an artifact with evaluated test results")
                record["validation_metrics"] = metrics.get("validation_metrics")
                record["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                atomic_json(marker, record)
                atomic_json(root / "comparison.json", report)
                print(f"Completed {name}", flush=True)
    report["complete"] = True
    atomic_json(root / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-type", choices=["physionet", "synthetic_demo"], required=True)
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "xgboost", "gru", "lstm", "transformer"])
    parser.add_argument("--windows", nargs="+", type=int, default=[12])
    parser.add_argument("--horizons", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--values-only", action="store_true", help="Temporal-only missingness ablation")
    parser.add_argument("--loss", choices=("weighted_bce", "focal"), default="weighted_bce",
                        help="Temporal training loss; focal runs receive distinct names")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    result = run_experiments(args.data, args.output, args.source_type, models=args.models,
                             windows=args.windows, horizons=args.horizons, epochs=args.epochs,
                             values_only=args.values_only, loss=args.loss, focal_gamma=args.focal_gamma)
    print(json.dumps({"comparison": str(Path(args.output).resolve() / "comparison.json"),
                      "completed_runs": len(result["runs"]), "test_evaluated": False}))


if __name__ == "__main__":
    main()
