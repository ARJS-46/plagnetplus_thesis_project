#!/usr/bin/env python3
"""Run one manifest-selected SimSiam or SwAV grid-search trial."""

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

from plagnetplus.training.simsiam_config import (
    load_simsiam_training_config,
)

from plagnetplus.training.swav_config import (
    load_swav_training_config,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and run one trial from the "
            "SimSiam/SwAV screening manifest."
        )
    )

    parser.add_argument(
        "--project-root",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--array-index",
        required=True,
        type=int,
        help=(
            "One-based PBS array index. "
            "Valid values are 1..number of trials."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_manifest(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(
            csv.DictReader(stream)
        )

    if not rows:
        raise RuntimeError(
            f"Grid manifest is empty: {path}"
        )

    return rows


def read_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def expanded_path(value: str) -> Path:
    return Path(
        os.path.expandvars(
            os.path.expanduser(value)
        )
    ).resolve()


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def main() -> None:
    arguments = parse_arguments()

    project_root = (
        arguments.project_root
        .expanduser()
        .resolve()
    )

    manifest_path = (
        arguments.manifest
        .expanduser()
        .resolve()
    )

    if not project_root.is_dir():
        raise SystemExit(
            "Project root does not exist: "
            f"{project_root}"
        )

    if not manifest_path.is_file():
        raise SystemExit(
            "Manifest does not exist: "
            f"{manifest_path}"
        )

    rows = read_manifest(
        manifest_path
    )

    if not (
        1
        <= arguments.array_index
        <= len(rows)
    ):
        raise SystemExit(
            "Array index is outside the valid "
            f"range 1..{len(rows)}: "
            f"{arguments.array_index}"
        )

    row_number = (
        arguments.array_index - 1
    )

    row = rows[row_number]

    required_columns = {
        "trial_id",
        "method",
        "run_name",
        "config_path",
        "config_sha256",
    }

    missing_columns = (
        required_columns
        - set(row)
    )

    if missing_columns:
        raise SystemExit(
            "Manifest row is missing columns: "
            f"{sorted(missing_columns)}"
        )

    trial_id = row["trial_id"]
    method = row["method"].strip().lower()
    expected_run_name = row["run_name"]

    config_path = (
        project_root
        / row["config_path"]
    ).resolve()

    try:
        config_path.relative_to(
            project_root
        )
    except ValueError as error:
        raise SystemExit(
            "Configuration path escapes the "
            f"project root: {config_path}"
        ) from error

    if not config_path.is_file():
        raise SystemExit(
            "Configuration file does not exist: "
            f"{config_path}"
        )

    actual_config_sha256 = sha256(
        config_path
    )

    if actual_config_sha256 != (
        row["config_sha256"]
    ):
        raise SystemExit(
            "Configuration SHA-256 mismatch:\n"
            f"expected={row['config_sha256']}\n"
            f"actual={actual_config_sha256}\n"
            f"path={config_path}"
        )

    payload = read_json(
        config_path
    )

    payload_method = str(
        payload["model"]["method"]
    ).strip().lower()

    payload_run_name = str(
        payload["run"]["name"]
    )

    if payload_method != method:
        raise SystemExit(
            "Manifest/config method mismatch: "
            f"{method} != {payload_method}"
        )

    if payload_run_name != (
        expected_run_name
    ):
        raise SystemExit(
            "Manifest/config run-name mismatch: "
            f"{expected_run_name} != "
            f"{payload_run_name}"
        )

    output_root = expanded_path(
        str(
            payload["run"]["output_root"]
        )
    )

    run_directory = (
        output_root
        / expected_run_name
    )

    if method == "simsiam":
        validated = (
            load_simsiam_training_config(
                config_path
            )
        )

        training_script = (
            project_root
            / "tools"
            / "training"
            / "train_simsiam.py"
        )

    elif method == "swav":
        validated = (
            load_swav_training_config(
                config_path
            )
        )

        training_script = (
            project_root
            / "tools"
            / "training"
            / "train_swav.py"
        )

    else:
        raise SystemExit(
            f"Unsupported grid method: {method}"
        )

    if not training_script.is_file():
        raise SystemExit(
            "Training script is missing: "
            f"{training_script}"
        )

    if validated.run_name != (
        expected_run_name
    ):
        raise SystemExit(
            "Validated run name does not match "
            f"the manifest: {validated.run_name}"
        )

    python_executable = Path(
        sys.executable
    ).resolve()

    command = [
        str(python_executable),
        str(training_script),
        "--config",
        str(config_path),
        "--run-name",
        expected_run_name,
        "--device",
        "cuda",
    ]

    status_root_value = os.environ.get(
        "PNP_GRID_STATUS_ROOT"
    )

    if status_root_value:
        status_root = expanded_path(
            status_root_value
        )
    else:
        runs_root_value = os.environ.get(
            "PNP_RUNS_ROOT"
        )

        if not runs_root_value:
            raise SystemExit(
                "PNP_RUNS_ROOT is not set."
            )

        status_root = (
            expanded_path(
                runs_root_value
            )
            / "hpc"
            / "grid_search"
            / "simsiam_swav_screen_v1"
            / "status"
        )

    metadata = {
        "schema_version": (
            "plagnetplus_grid_trial_status_v1"
        ),
        "array_index": (
            arguments.array_index
        ),
        "manifest_row_index": (
            row_number
        ),
        "trial_id": trial_id,
        "method": method,
        "run_name": expected_run_name,
        "config_path": str(
            config_path
        ),
        "config_sha256": (
            actual_config_sha256
        ),
        "output_root": str(
            output_root
        ),
        "run_directory": str(
            run_directory
        ),
        "python": str(
            python_executable
        ),
        "command": command,
        "hostname": os.uname().nodename,
        "pbs_job_id": os.environ.get(
            "PBS_JOBID"
        ),
        "pbs_array_index": (
            os.environ.get(
                "PBS_ARRAY_INDEX"
            )
        ),
        "started_utc": utc_now(),
        "dry_run": bool(
            arguments.dry_run
        ),
    }

    print(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    print(
        "COMMAND:",
        shlex.join(command),
        flush=True,
    )

    if arguments.dry_run:
        print(
            "GRID TRIAL DRY RUN: PASS",
            flush=True,
        )

        print(
            "TRAINING STARTED: NO",
            flush=True,
        )

        print(
            "TEST IMAGES ACCESSED: NO",
            flush=True,
        )

        return

    if run_directory.exists():
        raise SystemExit(
            "Refusing to overwrite an existing "
            f"run directory: {run_directory}"
        )

    started_path = (
        status_root
        / f"{trial_id}.started.json"
    )

    completed_path = (
        status_root
        / f"{trial_id}.completed.json"
    )

    failed_path = (
        status_root
        / f"{trial_id}.failed.json"
    )

    for stale_path in (
        completed_path,
        failed_path,
    ):
        if stale_path.exists():
            raise SystemExit(
                "Refusing to run while an existing "
                f"terminal status file exists: "
                f"{stale_path}"
            )

    write_json_atomic(
        started_path,
        metadata,
    )

    try:
        completed_process = subprocess.run(
            command,
            cwd=project_root,
            check=False,
        )

    except BaseException as error:
        failure_metadata = {
            **metadata,
            "failed_utc": utc_now(),
            "exception_type": (
                type(error).__name__
            ),
            "exception": str(error),
        }

        write_json_atomic(
            failed_path,
            failure_metadata,
        )

        raise

    if completed_process.returncode != 0:
        failure_metadata = {
            **metadata,
            "failed_utc": utc_now(),
            "returncode": (
                completed_process.returncode
            ),
        }

        write_json_atomic(
            failed_path,
            failure_metadata,
        )

        raise SystemExit(
            "Training command failed with "
            f"return code "
            f"{completed_process.returncode}."
        )

    if not run_directory.is_dir():
        failure_metadata = {
            **metadata,
            "failed_utc": utc_now(),
            "returncode": 0,
            "reason": (
                "Training returned zero but the "
                "expected run directory was absent."
            ),
        }

        write_json_atomic(
            failed_path,
            failure_metadata,
        )

        raise SystemExit(
            "Training completed without creating "
            f"the expected directory: "
            f"{run_directory}"
        )

    run_summary_path = (
        run_directory
        / "run_summary.json"
    )

    if not run_summary_path.is_file():
        failure_metadata = {
            **metadata,
            "failed_utc": utc_now(),
            "returncode": 0,
            "reason": (
                "Expected run_summary.json "
                "was not created."
            ),
        }

        write_json_atomic(
            failed_path,
            failure_metadata,
        )

        raise SystemExit(
            "Expected run summary is missing: "
            f"{run_summary_path}"
        )

    run_summary = read_json(
        run_summary_path
    )

    if run_summary.get("status") != "PASS":
        failure_metadata = {
            **metadata,
            "failed_utc": utc_now(),
            "returncode": 0,
            "reason": (
                "Run summary status was not PASS."
            ),
            "reported_status": (
                run_summary.get(
                    "status"
                )
            ),
        }

        write_json_atomic(
            failed_path,
            failure_metadata,
        )

        raise SystemExit(
            "Run summary did not report PASS."
        )

    completion_metadata = {
        **metadata,
        "completed_utc": utc_now(),
        "returncode": 0,
        "run_summary": str(
            run_summary_path
        ),
        "run_summary_sha256": sha256(
            run_summary_path
        ),
        "reported_status": (
            run_summary["status"]
        ),
        "completed_epochs": (
            run_summary.get(
                "completed_epochs"
            )
        ),
        "gpu_hours": (
            run_summary.get(
                "gpu_hours"
            )
        ),
    }

    write_json_atomic(
        completed_path,
        completion_metadata,
    )

    print(
        "GRID TRIAL EXECUTION: PASS",
        flush=True,
    )

    print(
        f"RUN DIRECTORY: {run_directory}",
        flush=True,
    )

    print(
        "TEST IMAGES ACCESSED: NO",
        flush=True,
    )


if __name__ == "__main__":
    main()
