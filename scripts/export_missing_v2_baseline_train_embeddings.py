#!/usr/bin/env python3
"""Export the two train embeddings missing from the completed V2 baselines.

SimSiam and SwAV export train embeddings as part of their baseline runs. The
older completed SimCLR and DeepCluster entry points exported validation only.
This helper verifies their frozen checkpoint and manifest hashes, then invokes
the existing method-specific exporters for the train split. It never reads the
test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


BENCHMARK_ID = "pooled_broad_p80_10_10_v2"
TRAIN_ROWS = 91_357
RUN_RELATIVE_PATHS = {
    "simclr": Path(
        "hpc/v2/baselines/simclr/"
        "simclr_resnet18_pooled_broad_natural_seed42_v2"
    ),
    "deepcluster": Path(
        "hpc/v2/baselines/deepcluster/"
        "deepcluster_resnet18_pooled_broad_natural_seed42_v2"
    ),
}


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, observed={observed}"
        )
    return path


def final_checkpoint(run_root: Path, summary: dict[str, Any], method: str) -> Path:
    key = "checkpoint" if method == "simclr" else "final_checkpoint"
    record = summary.get(key)
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise RuntimeError(f"Missing frozen checkpoint record in {run_root}.")
    path = run_root / "checkpoints" / Path(str(record["path"])).name
    return verified_file(path, str(record["sha256"]), f"{method} checkpoint")


def run_command(command: list[str], project_root: Path) -> None:
    environment = os.environ.copy()
    source_root = str(project_root / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not current_pythonpath
        else os.pathsep.join((source_root, current_pythonpath))
    )
    subprocess.run(command, cwd=project_root, env=environment, check=True)


def main() -> None:
    arguments = parse_arguments()
    project_root = arguments.project_root.expanduser().resolve()
    data_root = arguments.data_root.expanduser().resolve()
    runs_root = arguments.runs_root.expanduser().resolve()
    unified_root = data_root / "processed" / "plagnetplus" / "ssl_unified_v2"
    benchmark_root = unified_root / "benchmarks" / BENCHMARK_ID
    train_manifest = benchmark_root / "manifests" / "crystals_train.csv"
    image_root_map = unified_root / "metadata" / "image_root_map.json"
    transform_config = (
        benchmark_root
        / "metadata"
        / "normalization_v2"
        / "ssl_transform_config_v2.json"
    )
    for path in (train_manifest, image_root_map, transform_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Pass only the frozen training manifest and deterministic preprocessing; no
    # test manifest is provided by this recovery helper.
    common_arguments = [
        "--manifest",
        str(train_manifest),
        "--image-root-map",
        str(image_root_map),
        "--transform-config",
        str(transform_config),
        "--split",
        "train",
        "--expected-rows",
        str(TRAIN_ROWS),
        "--expected-benchmark-id",
        BENCHMARK_ID,
        "--device",
        arguments.device,
        "--batch-size",
        str(arguments.batch_size),
        "--num-workers",
        str(arguments.num_workers),
    ]

    # Only SimCLR and DeepCluster need this recovery export; their final frozen
    # checkpoints are verified before invoking the method-specific exporter.
    for method in ("simclr", "deepcluster"):
        run_root = runs_root / RUN_RELATIVE_PATHS[method]
        summary_path = run_root / "run_summary.json"
        record_path = run_root / "metadata" / "run_record.json"
        if not summary_path.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"Incomplete {method} metadata below {run_root}.")
        summary = read_json(summary_path)
        record = read_json(record_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"{method} run is not PASS: {run_root}")
        manifest_sha256 = str(record.get("manifest_sha256", ""))
        verified_file(train_manifest, manifest_sha256, "V2 train manifest")
        checkpoint = final_checkpoint(run_root, summary, method)
        output = run_root / "embeddings" / "train_backbone_embeddings.npz"
        output_summary = output.with_suffix(".summary.json")
        if not arguments.overwrite and (output.exists() or output_summary.exists()):
            raise FileExistsError(
                f"Output already exists for {method}; use --overwrite to replace it: {output}"
            )

        if method == "simclr":
            fingerprint = str(summary.get("manifest_fingerprint_sha256", ""))
            command = [
                sys.executable,
                str(project_root / "scripts" / "export_simclr_embeddings.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--summary",
                str(output_summary),
                "--expected-manifest-fingerprint",
                fingerprint,
                "--expected-checkpoint-sha256",
                str(summary["checkpoint"]["sha256"]),
                "--expected-manifest-sha256",
                manifest_sha256,
                *common_arguments,
            ]
        else:
            fingerprint = str(summary.get("benchmark_fingerprint", ""))
            command = [
                sys.executable,
                str(project_root / "scripts" / "export_deepcluster_embeddings.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--expected-benchmark-fingerprint",
                fingerprint,
                *common_arguments,
            ]
            if arguments.overwrite:
                command.append("--overwrite")
        if len(fingerprint) != 64:
            raise RuntimeError(f"Missing V2 manifest fingerprint for {method}.")
        print(f"Exporting verified {method} train embeddings to {output}", flush=True)
        run_command(command, project_root)

    print("V2 BASELINE TRAIN EMBEDDING EXPORTS: PASS")
    print("TEST MANIFEST LOADED: NO")
    print("TEST IMAGES ACCESSED: NO")


if __name__ == "__main__":
    main()
