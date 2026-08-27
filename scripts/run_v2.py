#!/usr/bin/env python3
"""Run the frozen unified-V2 baseline or selected-model configurations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINERS = {
    "simclr": "train_simclr.py",
    "deepcluster": "train_deepcluster.py",
    "simsiam": "train_simsiam.py",
    "swav": "train_swav.py",
}

MANIFESTS = {
    "baselines": Path("configs/v2_baselines/manifest.csv"),
    "selected": Path("configs/v2_selected/manifest.csv"),
}


def parse_arguments() -> argparse.Namespace:
    """Parse one local run or one PBS-array task."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(MANIFESTS), required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--trial", help="Trial ID from the cohort manifest")
    selector.add_argument(
        "--array-index",
        type=int,
        help="One-based row index, suitable for PBS_ARRAY_INDEX",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write status metadata without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expanded_path(value: str, label: str) -> Path:
    """Expand an environment-backed path and reject unresolved variables."""

    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise SystemExit(f"Unresolved environment variable in {label}: {value}")
    return Path(expanded).resolve()


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def config_identity(payload: dict[str, Any], method: str) -> tuple[str, str]:
    """Return the run name and output root for either config schema."""

    if method == "deepcluster":
        return str(payload["run_name"]), str(payload["output_root"])
    payload_method = str(payload["model"]["method"]).strip().lower()
    if payload_method != method:
        raise SystemExit(
            f"Manifest/config method mismatch: {method} != {payload_method}"
        )
    return str(payload["run"]["name"]), str(payload["run"]["output_root"])


def select_rows(
    rows: list[dict[str, str]], arguments: argparse.Namespace
) -> list[dict[str, str]]:
    """Select all trials, a named trial, or a one-based array row."""

    if arguments.array_index is not None:
        if not 1 <= arguments.array_index <= len(rows):
            raise SystemExit(
                f"Array index must lie in 1..{len(rows)}: "
                f"{arguments.array_index}"
            )
        return [rows[arguments.array_index - 1]]
    if arguments.trial:
        selected = [row for row in rows if row["trial_id"] == arguments.trial]
        if not selected:
            raise SystemExit(f"Unknown trial ID: {arguments.trial}")
        return selected
    return rows


def validate_row(
    project_root: Path, row: dict[str, str]
) -> tuple[Path, dict[str, Any], str, Path]:
    """Validate manifest-to-config identity and return resolved run metadata."""

    required = {
        "trial_id",
        "method",
        "run_name",
        "config_path",
        "config_sha256",
        "expected_units",
        "unit_name",
    }
    missing = required - set(row)
    if missing:
        raise SystemExit(f"Manifest columns missing: {sorted(missing)}")
    method = row["method"].strip().lower()
    if method not in TRAINERS:
        raise SystemExit(f"Unsupported method: {method}")
    config_path = (project_root / row["config_path"]).resolve()
    try:
        config_path.relative_to(project_root)
    except ValueError as error:
        raise SystemExit(f"Config escapes project root: {config_path}") from error
    if not config_path.is_file():
        raise SystemExit(f"Configuration missing: {config_path}")
    actual_hash = sha256(config_path)
    if actual_hash != row["config_sha256"]:
        raise SystemExit(
            "Configuration SHA-256 mismatch:\n"
            f"expected={row['config_sha256']}\nactual={actual_hash}"
        )
    payload = read_json(config_path)
    run_name, output_root_text = config_identity(payload, method)
    if run_name != row["run_name"]:
        raise SystemExit(
            f"Manifest/config run-name mismatch: {row['run_name']} != {run_name}"
        )
    if "pooled_broad_p80_10_10_v2" not in json.dumps(payload):
        raise SystemExit("Configuration is not a unified-V2 configuration.")
    output_root = expanded_path(output_root_text, "output_root")
    return config_path, payload, actual_hash, output_root / run_name


def run_trial(
    project_root: Path,
    cohort: str,
    row: dict[str, str],
    device: str,
    dry_run: bool,
) -> None:
    """Run and verify one immutable training trial."""

    config_path, _, config_hash, run_directory = validate_row(project_root, row)
    method = row["method"].strip().lower()
    trainer = project_root / "scripts" / TRAINERS[method]
    if not trainer.is_file():
        raise SystemExit(f"Trainer missing: {trainer}")
    command = [
        str(Path(sys.executable).resolve()),
        str(trainer),
        "--config",
        str(config_path),
        "--run-name",
        row["run_name"],
        "--device",
        device,
    ]
    status_root_value = os.environ.get("PNP_RERUN_STATUS_ROOT")
    if status_root_value:
        status_root = expanded_path(status_root_value, "PNP_RERUN_STATUS_ROOT")
    else:
        runs_root = os.environ.get("PNP_RUNS_ROOT")
        if not runs_root:
            raise SystemExit("PNP_RUNS_ROOT is not set.")
        status_root = (
            expanded_path(runs_root, "PNP_RUNS_ROOT")
            / "hpc"
            / "v2"
            / "status"
            / cohort
        )
    metadata: dict[str, Any] = {
        "schema_version": "plagnetplus_ssl_unified_v2_rerun_status_v1",
        "cohort": cohort,
        "trial_id": row["trial_id"],
        "method": method,
        "run_name": row["run_name"],
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "run_directory": str(run_directory),
        "command": command,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
        "started_utc": utc_now(),
        "dry_run": dry_run,
    }
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print("COMMAND:", shlex.join(command), flush=True)
    if dry_run:
        print("V2 RERUN DRY RUN: PASS")
        print("TRAINING STARTED: NO")
        print("TEST IMAGES ACCESSED: NO")
        return
    if run_directory.exists():
        raise SystemExit(f"Refusing to overwrite run: {run_directory}")
    started = status_root / f"{row['trial_id']}.started.json"
    completed = status_root / f"{row['trial_id']}.completed.json"
    failed = status_root / f"{row['trial_id']}.failed.json"
    for terminal in (completed, failed):
        if terminal.exists():
            raise SystemExit(f"Terminal status file already exists: {terminal}")
    write_json_atomic(started, metadata)
    process = subprocess.run(command, cwd=project_root, check=False)
    if process.returncode != 0:
        write_json_atomic(
            failed,
            {**metadata, "failed_utc": utc_now(), "returncode": process.returncode},
        )
        raise SystemExit(f"Training failed with return code {process.returncode}.")
    summary_path = run_directory / "run_summary.json"
    if not summary_path.is_file():
        write_json_atomic(
            failed,
            {**metadata, "failed_utc": utc_now(), "reason": "summary_missing"},
        )
        raise SystemExit(f"Run summary missing: {summary_path}")
    summary = read_json(summary_path)
    checks = summary.get("checks", {})
    if row["unit_name"] == "cycles":
        # DeepCluster's native summary schema uses ``cycles_completed``.
        # Accept the older wrapper spelling as a compatibility fallback, but
        # always write the canonical wrapper field to the status record.
        summary_unit_keys = ("cycles_completed", "completed_cycles")
        status_unit_key = "completed_cycles"
    else:
        summary_unit_keys = ("completed_epochs",)
        status_unit_key = "completed_epochs"
    completed_units = next(
        (summary[key] for key in summary_unit_keys if key in summary),
        None,
    )
    expected_units = int(row["expected_units"])
    errors = []
    if summary.get("status") != "PASS":
        errors.append(f"status={summary.get('status')!r}")
    if completed_units != expected_units:
        errors.append(
            f"{summary_unit_keys[0]}={completed_units!r}, expected {expected_units}"
        )
    if checks.get("test_images_accessed", False) is not False:
        errors.append("test_images_accessed is not false")
    if checks.get("test_embeddings_exported", False) is not False:
        errors.append("test_embeddings_exported is not false")
    if errors:
        write_json_atomic(
            failed,
            {
                **metadata,
                "failed_utc": utc_now(),
                "reason": "summary_contract_failed",
                "errors": errors,
            },
        )
        raise SystemExit("Run summary contract failed: " + "; ".join(errors))
    completion = {
        **metadata,
        "completed_utc": utc_now(),
        "returncode": 0,
        "run_summary": str(summary_path),
        "run_summary_sha256": sha256(summary_path),
        "reported_status": summary["status"],
        status_unit_key: completed_units,
        "gpu_hours": summary.get("gpu_hours"),
    }
    write_json_atomic(completed, completion)
    print("V2 RERUN EXECUTION: PASS")
    print(f"RUN DIRECTORY: {run_directory}")
    print("TEST IMAGES ACCESSED: NO")


def main() -> None:
    """Validate the frozen cohort manifest and dispatch selected trials."""

    arguments = parse_arguments()
    project_root = arguments.project_root.expanduser().resolve()
    manifest_path = project_root / MANIFESTS[arguments.cohort]
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in select_rows(rows, arguments):
        run_trial(
            project_root,
            arguments.cohort,
            row,
            arguments.device,
            arguments.dry_run,
        )


if __name__ == "__main__":
    main()
