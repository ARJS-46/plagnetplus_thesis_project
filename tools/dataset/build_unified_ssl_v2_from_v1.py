#!/usr/bin/env python3
"""Build the corrected PlagNetPlus unified SSL dataset v2.

The builder derives an immutable, index-only v2 dataset from the frozen v1
dataset. It removes two acquisition variants from the training population,
retains the existing validation and test memberships, regenerates contiguous
canonical manifest orders, records v1-to-v2 order lineage, and writes fresh
metadata and checksums. No image files are copied, modified, or opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_PROJECT_ROOT = Path(
    os.environ.get(
        "PNP_PROJECT_ROOT",
        Path(__file__).resolve().parents[3],
    )
)

DEFAULT_V1_ROOT = (
    DEFAULT_PROJECT_ROOT
    / "data/processed/plagnetplus/ssl_unified_v1"
)

DEFAULT_PARENT_BENCHMARK_ID = "pooled_broad_p80_10_10_v1"
DEFAULT_V2_BENCHMARK_ID = "pooled_broad_p80_10_10_v2"

EXCLUDED_ACQUISITIONS = (
    "6861_recoated",
    "M83-7-1-II_no_overlap",
)

EXPECTED_PARENT_HASHES = {
    "benchmarks/pooled_broad_p80_10_10_v1/manifests/"
    "crystals_train.csv": (
        "20636d38d2e8f127572943cdaf0256cb"
        "5cefb600cb46b176b7efd02d98ca7807"
    ),
    "benchmarks/pooled_broad_p80_10_10_v1/manifests/"
    "crystals_validation.csv": (
        "f5bf6d9489d4df8b91f9f1394b6c583d"
        "fd1508b4a540c53be9515e73fa91a22d"
    ),
    "benchmarks/pooled_broad_p80_10_10_v1/manifests/"
    "crystals_test.csv": (
        "60210e746ca5968e9a06e0d1cc35711b"
        "956d7b9c9a4c29e7429c30ac61dcf617"
    ),
}

EXPECTED_EXCLUDED_COUNTS = {
    "6861_recoated": 4_309,
    "M83-7-1-II_no_overlap": 6_016,
}

EXPECTED_V2_SPLIT_COUNTS = {
    "train": 91_357,
    "validation": 14_919,
    "test": 16_988,
}

EXPECTED_V2_LOCALITY_COUNTS = {
    "Aegina": 12_344,
    "FCT": 21_305,
    "Methana": 68_008,
    "MtStHelens": 3_101,
    "Rumania": 18_506,
}

EXPECTED_V2_ROWS = 123_264
EXPECTED_V2_ACQUISITIONS = 27
EXPECTED_V2_PHYSICAL_SAMPLES = 27
EXPECTED_V2_SPLIT_GROUPS = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ssl_unified_v2 by filtering two acquisition variants "
            "from ssl_unified_v1 without accessing image files."
        )
    )
    parser.add_argument(
        "--v1-root",
        type=Path,
        default=DEFAULT_V1_ROOT,
        help="Frozen ssl_unified_v1 root.",
    )
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=None,
        help=(
            "Destination root. Defaults to an ssl_unified_v2 sibling of "
            "--v1-root. The destination must not already exist."
        ),
    )
    parser.add_argument(
        "--parent-benchmark-id",
        default=DEFAULT_PARENT_BENCHMARK_ID,
    )
    parser.add_argument(
        "--benchmark-id",
        default=DEFAULT_V2_BENCHMARK_ID,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_checksum_file(
    base: Path,
    relative_paths: Iterable[Path],
) -> None:
    records = []
    for relative_path in sorted(relative_paths, key=lambda item: item.as_posix()):
        absolute_path = base / relative_path
        records.append(
            f"{sha256_file(absolute_path)}  {relative_path.as_posix()}"
        )
    (base / "SHA256SUMS").write_text(
        "\n".join(records) + "\n",
        encoding="utf-8",
    )


def empty_file_with_parent_header(parent: Path, output: Path) -> None:
    with parent.open("r", encoding="utf-8", newline="") as source:
        header = source.readline()
    if not header:
        raise RuntimeError(f"Parent error table has no header: {parent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header, encoding="utf-8")


def stable_manifest_fingerprint(rows: Iterable[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for values in rows:
        digest.update("\t".join(values).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_parent_hashes(v1_root: Path) -> None:
    failures = []
    for relative_name, expected_hash in EXPECTED_PARENT_HASHES.items():
        path = v1_root / relative_name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        passed = actual_hash == expected_hash
        print(
            f"Parent hash {relative_name}: "
            f"{'PASS' if passed else 'CHANGED'}"
        )
        if not passed:
            failures.append(relative_name)
    if failures:
        raise RuntimeError(
            "Frozen v1 manifests differ from the audited files: "
            f"{failures}"
        )


def csv_writer(path: Path, fieldnames: Sequence[str]) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    return handle, writer


def main() -> None:
    args = parse_args()
    v1_root = args.v1_root.expanduser().resolve()
    v2_root = (
        args.v2_root.expanduser().resolve()
        if args.v2_root is not None
        else v1_root.with_name("ssl_unified_v2")
    )

    parent_benchmark_root = (
        v1_root / "benchmarks" / args.parent_benchmark_id
    )

    required_paths = [
        v1_root / "manifests" / "crystals_broad.csv",
        v1_root / "manifests" / "build_errors.tsv",
        v1_root / "metadata" / "image_root_map.json",
        v1_root / "metadata" / "unified_ssl_manifest_summary_v1.json",
        parent_benchmark_root / "manifests" / "crystals_all.csv",
        parent_benchmark_root / "metadata" / "freeze_errors.tsv",
        parent_benchmark_root / "metadata" / "split_group_assignments.csv",
        parent_benchmark_root / "metadata" / "pooled_broad_split_summary_v1.json",
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if v2_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing destination: {v2_root}"
        )

    if v2_root.parent != v1_root.parent:
        raise RuntimeError(
            "For atomic installation, v2 must be a sibling of v1."
        )

    free_bytes = shutil.disk_usage(v2_root.parent).free
    if free_bytes < 1_000_000_000:
        raise RuntimeError(
            "At least 1 GB of free space is required to build v2."
        )

    print("========== VERIFY FROZEN V1 ==========")
    check_parent_hashes(v1_root)

    build_root = Path(
        tempfile.mkdtemp(
            prefix=".ssl_unified_v2_build_",
            dir=v2_root.parent,
        )
    )

    installed = False

    try:
        benchmark_root = (
            build_root / "benchmarks" / args.benchmark_id
        )
        final_benchmark_root = (
            v2_root / "benchmarks" / args.benchmark_id
        )

        canonical_parent = (
            v1_root / "manifests" / "crystals_broad.csv"
        )
        canonical_output = (
            build_root / "manifests" / "crystals_broad.csv"
        )

        # V2 removes only the two identified redundant acquisition variants while
        # preserving the accepted V1 split membership of retained crystals.
        excluded = set(EXCLUDED_ACQUISITIONS)
        excluded_counts: Counter[str] = Counter()
        acquisition_counts: Counter[str] = Counter()
        locality_counts: Counter[str] = Counter()
        source_dataset_counts: Counter[str] = Counter()
        sample_ids: set[str] = set()
        physical_sample_ids: set[str] = set()
        crystal_ids: set[str] = set()
        image_locators: set[str] = set()
        image_hashes: set[str] = set()
        # Reindex retained crystals contiguously for V2 while recording the original
        # V1 manifest order in the lineage table.
        new_order_by_crystal_id: dict[str, int] = {}
        old_order_by_crystal_id: dict[str, int] = {}

        group_details: dict[str, dict[str, object]] = {}
        canonical_fingerprint_rows: list[tuple[str, str, str, str]] = []
        canonical_column_count = 0

        print("========== BUILD V2 CANONICAL MANIFEST ==========")

        with canonical_parent.open(
            "r", encoding="utf-8", newline=""
        ) as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise RuntimeError("Canonical v1 manifest has no header.")
            canonical_column_count = len(reader.fieldnames)

            required_columns = {
                "manifest_order",
                "crystal_id",
                "locality",
                "sample_id",
                "physical_sample_id",
                "acquisition_id",
                "split_group_id",
                "image_locator",
                "image_sha256",
                "source_dataset",
            }
            missing_columns = required_columns.difference(reader.fieldnames)
            if missing_columns:
                raise RuntimeError(
                    f"Canonical manifest missing columns: {missing_columns}"
                )

            output_handle, writer = csv_writer(
                canonical_output,
                reader.fieldnames,
            )

            try:
                for row in reader:
                    acquisition = row["acquisition_id"]
                    if acquisition in excluded:
                        excluded_counts[acquisition] += 1
                        continue

                    crystal_id = row["crystal_id"]
                    if crystal_id in new_order_by_crystal_id:
                        raise RuntimeError(
                            f"Duplicate crystal_id in v2: {crystal_id}"
                        )

                    old_order = int(row["manifest_order"])
                    new_order = len(new_order_by_crystal_id)
                    row["manifest_order"] = str(new_order)

                    new_order_by_crystal_id[crystal_id] = new_order
                    old_order_by_crystal_id[crystal_id] = old_order

                    acquisition_counts[acquisition] += 1
                    locality_counts[row["locality"]] += 1
                    source_dataset_counts[row["source_dataset"]] += 1
                    sample_ids.add(row["sample_id"])
                    physical_sample_ids.add(row["physical_sample_id"])

                    if crystal_id in crystal_ids:
                        raise RuntimeError(f"Duplicate crystal ID: {crystal_id}")
                    crystal_ids.add(crystal_id)

                    locator = row["image_locator"]
                    if locator in image_locators:
                        raise RuntimeError(f"Duplicate image locator: {locator}")
                    image_locators.add(locator)

                    image_hash = row["image_sha256"]
                    if image_hash in image_hashes:
                        raise RuntimeError(
                            f"Exact-content duplicate retained: {image_hash}"
                        )
                    image_hashes.add(image_hash)

                    group_id = row["split_group_id"]
                    if group_id not in group_details:
                        group_details[group_id] = {
                            "locality": row["locality"],
                            "physical_sample_id": row["physical_sample_id"],
                            "acquisitions": [],
                            "acquisition_set": set(),
                            "crystal_count": 0,
                        }

                    group = group_details[group_id]
                    if (
                        group["locality"] != row["locality"]
                        or group["physical_sample_id"]
                        != row["physical_sample_id"]
                    ):
                        raise RuntimeError(
                            f"Inconsistent metadata for split group {group_id}"
                        )

                    group["crystal_count"] = int(group["crystal_count"]) + 1
                    acquisition_set = group["acquisition_set"]
                    acquisition_list = group["acquisitions"]
                    assert isinstance(acquisition_set, set)
                    assert isinstance(acquisition_list, list)
                    if acquisition not in acquisition_set:
                        acquisition_set.add(acquisition)
                        acquisition_list.append(acquisition)

                    canonical_fingerprint_rows.append(
                        (
                            str(new_order),
                            crystal_id,
                            locator,
                            image_hash,
                        )
                    )
                    writer.writerow(row)
            finally:
                output_handle.close()

        if excluded_counts != Counter(EXPECTED_EXCLUDED_COUNTS):
            raise RuntimeError(
                "Unexpected excluded-acquisition counts: "
                f"{dict(excluded_counts)}"
            )

        if len(new_order_by_crystal_id) != EXPECTED_V2_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_V2_ROWS:,} v2 rows, found "
                f"{len(new_order_by_crystal_id):,}"
            )

        if dict(locality_counts) != EXPECTED_V2_LOCALITY_COUNTS:
            raise RuntimeError(
                f"Unexpected v2 locality counts: {dict(locality_counts)}"
            )

        if len(acquisition_counts) != EXPECTED_V2_ACQUISITIONS:
            raise RuntimeError("Unexpected v2 acquisition count.")
        if len(physical_sample_ids) != EXPECTED_V2_PHYSICAL_SAMPLES:
            raise RuntimeError("Unexpected v2 physical-sample count.")
        if len(group_details) != EXPECTED_V2_SPLIT_GROUPS:
            raise RuntimeError("Unexpected v2 split-group count.")

        canonical_fingerprint = stable_manifest_fingerprint(
            canonical_fingerprint_rows
        )

        split_groups_output = build_root / "metadata" / "split_groups.csv"
        split_group_fields = [
            "split_group_id",
            "locality",
            "physical_sample_id",
            "acquisitions",
            "acquisition_count",
            "crystal_count",
        ]
        group_handle, group_writer = csv_writer(
            split_groups_output,
            split_group_fields,
        )
        try:
            for group_id in sorted(group_details):
                group = group_details[group_id]
                acquisitions = group["acquisitions"]
                assert isinstance(acquisitions, list)
                group_writer.writerow(
                    {
                        "split_group_id": group_id,
                        "locality": group["locality"],
                        "physical_sample_id": group["physical_sample_id"],
                        "acquisitions": ";".join(acquisitions),
                        "acquisition_count": len(acquisitions),
                        "crystal_count": group["crystal_count"],
                    }
                )
        finally:
            group_handle.close()

        exclusions_output = build_root / "metadata" / "exclusions_v2.csv"
        exclusion_handle, exclusion_writer = csv_writer(
            exclusions_output,
            [
                "acquisition_id",
                "rows_removed",
                "parent_benchmark_split",
                "reason",
                "exact_content_duplicates_claimed",
            ],
        )
        try:
            for acquisition in EXCLUDED_ACQUISITIONS:
                exclusion_writer.writerow(
                    {
                        "acquisition_id": acquisition,
                        "rows_removed": excluded_counts[acquisition],
                        "parent_benchmark_split": "train",
                        "reason": (
                            "excluded_redundant_acquisition_variant_"
                            "by_project_decision"
                        ),
                        "exact_content_duplicates_claimed": "False",
                    }
                )
        finally:
            exclusion_handle.close()

        shutil.copy2(
            v1_root / "metadata" / "image_root_map.json",
            build_root / "metadata" / "image_root_map.json",
        )
        empty_file_with_parent_header(
            v1_root / "manifests" / "build_errors.tsv",
            build_root / "manifests" / "build_errors.tsv",
        )

        print("========== BUILD V2 BENCHMARK MANIFESTS ==========")

        parent_all = (
            parent_benchmark_root / "manifests" / "crystals_all.csv"
        )
        all_output = benchmark_root / "manifests" / "crystals_all.csv"
        split_outputs = {
            split: benchmark_root / "manifests" / f"crystals_{split}.csv"
            for split in ("train", "validation", "test")
        }
        lineage_output = (
            benchmark_root
            / "metadata"
            / "manifest_order_lineage_v1_to_v2.csv"
        )

        split_counts: Counter[str] = Counter()
        locality_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
        group_split_sets: dict[str, set[str]] = defaultdict(set)
        locality_group_sets: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        excluded_parent_splits: dict[str, Counter[str]] = defaultdict(Counter)
        benchmark_fingerprint_rows: list[tuple[str, str, str, str, str]] = []
        parent_membership_hashes = {
            "validation": hashlib.sha256(),
            "test": hashlib.sha256(),
        }
        v2_membership_hashes = {
            "validation": hashlib.sha256(),
            "test": hashlib.sha256(),
        }

        with parent_all.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError("Parent all-manifest has no header.")

            all_handle, all_writer = csv_writer(all_output, reader.fieldnames)
            split_handles: dict[str, object] = {}
            split_writers: dict[str, csv.DictWriter] = {}
            for split, output in split_outputs.items():
                split_handle, split_writer = csv_writer(output, reader.fieldnames)
                split_handles[split] = split_handle
                split_writers[split] = split_writer

            lineage_handle, lineage_writer = csv_writer(
                lineage_output,
                [
                    "crystal_id",
                    "v1_manifest_order",
                    "v2_manifest_order",
                    "benchmark_split",
                    "acquisition_id",
                ],
            )

            try:
                all_row_count = 0
                for row in reader:
                    split = row["benchmark_split"]
                    if split not in split_outputs:
                        raise RuntimeError(f"Invalid parent split label: {split}")

                    crystal_id = row["crystal_id"]
                    if split in parent_membership_hashes:
                        parent_membership_hashes[split].update(
                            (crystal_id + "\n").encode("utf-8")
                        )

                    acquisition = row["acquisition_id"]
                    if acquisition in excluded:
                        excluded_parent_splits[acquisition][split] += 1
                        continue

                    if crystal_id not in new_order_by_crystal_id:
                        raise RuntimeError(
                            f"Benchmark crystal missing from canonical v2: "
                            f"{crystal_id}"
                        )

                    old_order = int(row["manifest_order"])
                    if old_order != old_order_by_crystal_id[crystal_id]:
                        raise RuntimeError(
                            f"Parent manifest-order disagreement: {crystal_id}"
                        )

                    new_order = new_order_by_crystal_id[crystal_id]
                    row["manifest_order"] = str(new_order)
                    row["benchmark_id"] = args.benchmark_id

                    all_writer.writerow(row)
                    split_writers[split].writerow(row)
                    lineage_writer.writerow(
                        {
                            "crystal_id": crystal_id,
                            "v1_manifest_order": old_order,
                            "v2_manifest_order": new_order,
                            "benchmark_split": split,
                            "acquisition_id": acquisition,
                        }
                    )

                    split_counts[split] += 1
                    locality_split_counts[row["locality"]][split] += 1
                    group_split_sets[row["split_group_id"]].add(split)
                    locality_group_sets[row["locality"]][split].add(
                        row["split_group_id"]
                    )

                    if split in v2_membership_hashes:
                        v2_membership_hashes[split].update(
                            (crystal_id + "\n").encode("utf-8")
                        )

                    # Build the new fingerprint from V2 row identities after duplicate removal
                    # so downstream runs cannot silently mix V1 and V2 manifests.
                    benchmark_fingerprint_rows.append(
                        (
                            str(new_order),
                            crystal_id,
                            split,
                            row["image_locator"],
                            row["image_sha256"],
                        )
                    )
                    all_row_count += 1
            finally:
                all_handle.close()
                lineage_handle.close()
                for split_handle in split_handles.values():
                    split_handle.close()

        if all_row_count != EXPECTED_V2_ROWS:
            raise RuntimeError("Unexpected crystals_all.csv row count.")
        if dict(split_counts) != EXPECTED_V2_SPLIT_COUNTS:
            raise RuntimeError(
                f"Unexpected v2 split counts: {dict(split_counts)}"
            )
        for acquisition, expected_count in EXPECTED_EXCLUDED_COUNTS.items():
            if excluded_parent_splits[acquisition] != Counter(
                {"train": expected_count}
            ):
                raise RuntimeError(
                    f"Excluded acquisition was not train-only: {acquisition} "
                    f"{dict(excluded_parent_splits[acquisition])}"
                )
        leaking_groups = {
            group_id: sorted(splits)
            for group_id, splits in group_split_sets.items()
            if len(splits) != 1
        }
        if leaking_groups:
            raise RuntimeError(f"Split-group leakage detected: {leaking_groups}")
        for split in ("validation", "test"):
            if (
                parent_membership_hashes[split].hexdigest()
                != v2_membership_hashes[split].hexdigest()
            ):
                raise RuntimeError(
                    f"{split} membership changed unexpectedly."
                )

        benchmark_fingerprint = stable_manifest_fingerprint(
            benchmark_fingerprint_rows
        )

        parent_assignments = (
            parent_benchmark_root
            / "metadata"
            / "split_group_assignments.csv"
        )
        assignments_output = (
            benchmark_root
            / "metadata"
            / "split_group_assignments.csv"
        )

        with parent_assignments.open(
            "r", encoding="utf-8", newline=""
        ) as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise RuntimeError("Parent assignment table has no header.")
            assignment_handle, assignment_writer = csv_writer(
                assignments_output,
                reader.fieldnames,
            )
            assignment_count = 0
            try:
                for row in reader:
                    group_id = row["split_group_id"]
                    if group_id not in group_details:
                        continue
                    details = group_details[group_id]
                    acquisitions = details["acquisitions"]
                    assert isinstance(acquisitions, list)
                    row["benchmark_id"] = args.benchmark_id
                    row["acquisitions"] = ";".join(acquisitions)
                    row["crystal_count"] = str(details["crystal_count"])
                    actual_split = next(iter(group_split_sets[group_id]))
                    if row["benchmark_split"] != actual_split:
                        raise RuntimeError(
                            f"Assignment mismatch for group {group_id}"
                        )
                    assignment_writer.writerow(row)
                    assignment_count += 1
            finally:
                assignment_handle.close()

        if assignment_count != EXPECTED_V2_SPLIT_GROUPS:
            raise RuntimeError("Unexpected assignment-table row count.")

        locality_output = (
            benchmark_root
            / "metadata"
            / "split_counts_by_locality.csv"
        )
        locality_fields = [
            "locality",
            "total_crystals",
            "train_crystals",
            "validation_crystals",
            "test_crystals",
            "train_fraction",
            "validation_fraction",
            "test_fraction",
            "train_group_count",
            "validation_group_count",
            "test_group_count",
        ]
        locality_handle, locality_writer = csv_writer(
            locality_output,
            locality_fields,
        )
        try:
            for locality in sorted(locality_counts):
                total = locality_counts[locality]
                split_values = locality_split_counts[locality]
                locality_writer.writerow(
                    {
                        "locality": locality,
                        "total_crystals": total,
                        "train_crystals": split_values["train"],
                        "validation_crystals": split_values["validation"],
                        "test_crystals": split_values["test"],
                        "train_fraction": split_values["train"] / total,
                        "validation_fraction": (
                            split_values["validation"] / total
                        ),
                        "test_fraction": split_values["test"] / total,
                        "train_group_count": len(
                            locality_group_sets[locality]["train"]
                        ),
                        "validation_group_count": len(
                            locality_group_sets[locality]["validation"]
                        ),
                        "test_group_count": len(
                            locality_group_sets[locality]["test"]
                        ),
                    }
                )
        finally:
            locality_handle.close()

        empty_file_with_parent_header(
            parent_benchmark_root / "metadata" / "freeze_errors.tsv",
            benchmark_root / "metadata" / "freeze_errors.tsv",
        )

        created_utc = datetime.now(timezone.utc).isoformat()
        parent_canonical_hash = sha256_file(canonical_parent)
        v2_canonical_hash = sha256_file(canonical_output)
        image_root_map_hash = sha256_file(
            build_root / "metadata" / "image_root_map.json"
        )

        parent_summary = json.loads(
            (
                v1_root
                / "metadata"
                / "unified_ssl_manifest_summary_v1.json"
            ).read_text(encoding="utf-8")
        )
        parent_benchmark_summary = json.loads(
            (
                parent_benchmark_root
                / "metadata"
                / "pooled_broad_split_summary_v1.json"
            ).read_text(encoding="utf-8")
        )

        root_summary = {
            "status": "PASS",
            "schema_version": "plagnetplus_unified_ssl_manifest_v2",
            "dataset_id": "plagnetplus_ssl_unified",
            "dataset_version": "v2",
            "created_utc": created_utc,
            "derivation": (
                "Filtered immutable derivative of ssl_unified_v1. Two "
                "project-designated redundant acquisition variants were "
                "removed. No image files were copied, modified, or opened."
            ),
            "duplicate_interpretation": (
                "The removed acquisitions are treated as redundant physical-"
                "material/acquisition variants. The v1 audit found no exact "
                "image-content duplicate groups."
            ),
            "parent_dataset": {
                "path": str(v1_root),
                "dataset_version": parent_summary.get("dataset_version"),
                "canonical_manifest_sha256": parent_canonical_hash,
            },
            "excluded_acquisitions": [
                {
                    "acquisition_id": acquisition,
                    "rows_removed": excluded_counts[acquisition],
                    "parent_benchmark_split": "train",
                }
                for acquisition in EXCLUDED_ACQUISITIONS
            ],
            "rows_removed": sum(excluded_counts.values()),
            "rows": EXPECTED_V2_ROWS,
            "columns": canonical_column_count,
            "acquisition_count": len(acquisition_counts),
            "sample_count": len(sample_ids),
            "physical_sample_count": len(physical_sample_ids),
            "split_group_count": len(group_details),
            "unique_crystal_ids": len(crystal_ids),
            "unique_image_locators": len(image_locators),
            "exact_content_duplicate_groups": 0,
            "fct_6861_acquisitions": ["6861_FCT"],
            "fct_6861_crystal_count": acquisition_counts["6861_FCT"],
            "locality_counts": dict(sorted(locality_counts.items())),
            "source_dataset_counts": dict(
                sorted(source_dataset_counts.items())
            ),
            "representation_view": parent_summary.get(
                "representation_view", "broad"
            ),
            "ordering_policy": (
                "v1 canonical order after acquisition filtering, followed "
                "by contiguous v2 manifest_order reassignment"
            ),
            "manifest_fingerprint_policy": (
                "sha256 over tab-separated v2 manifest_order, crystal_id, "
                "image_locator, and image_sha256 records"
            ),
            "manifest_fingerprint_sha256": canonical_fingerprint,
            "manifest_file_sha256": v2_canonical_hash,
            "image_root_map_sha256": image_root_map_hash,
            "source_manifests": parent_summary.get("source_manifests", {}),
            "outputs": {
                "manifest": str(
                    v2_root / "manifests" / "crystals_broad.csv"
                ),
                "errors": str(
                    v2_root / "manifests" / "build_errors.tsv"
                ),
                "image_root_map": str(
                    v2_root / "metadata" / "image_root_map.json"
                ),
                "split_groups": str(
                    v2_root / "metadata" / "split_groups.csv"
                ),
                "exclusions": str(
                    v2_root / "metadata" / "exclusions_v2.csv"
                ),
            },
            "checks": {
                "parent_v1_hashes_match_audit": True,
                "rows_expected": True,
                "excluded_counts_expected": True,
                "manifest_order_contiguous": True,
                "crystal_ids_unique": True,
                "image_locators_unique": True,
                "no_exact_content_duplicates": True,
                "acquisition_count_expected": True,
                "physical_sample_count_expected": True,
                "split_group_count_expected": True,
                "all_paths_remain_index_only": True,
            },
        }

        root_summary_path = (
            build_root
            / "metadata"
            / "unified_ssl_manifest_summary_v2.json"
        )
        write_json(root_summary_path, root_summary)

        group_split_counts = Counter(
            next(iter(splits)) for splits in group_split_sets.values()
        )
        benchmark_summary = {
            "status": "PASS",
            "schema_version": "pooled_broad_split_v2",
            "benchmark_id": args.benchmark_id,
            "created_utc": created_utc,
            "parent_benchmark_id": args.parent_benchmark_id,
            "parent_benchmark_path": str(parent_benchmark_root),
            "derivation": (
                "Retains the frozen v1 split membership after removing "
                "6861_recoated and M83-7-1-II_no_overlap from training."
            ),
            "selected_candidate": parent_benchmark_summary.get(
                "selected_candidate"
            ),
            "interpretation": (
                "Corrected pooled broad benchmark. Validation and test "
                "crystal memberships are identical to v1. All remaining "
                "assignment is leakage-safe by split_group_id."
            ),
            "source_manifest": str(
                v2_root / "manifests" / "crystals_broad.csv"
            ),
            "source_manifest_sha256": v2_canonical_hash,
            "manifest_fingerprint_policy": (
                "sha256 over tab-separated v2 manifest_order, crystal_id, "
                "benchmark_split, image_locator, and image_sha256 records"
            ),
            "manifest_fingerprint_sha256": benchmark_fingerprint,
            "rows": EXPECTED_V2_ROWS,
            "split_row_counts": dict(split_counts),
            "split_row_fractions": {
                split: split_counts[split] / EXPECTED_V2_ROWS
                for split in ("train", "validation", "test")
            },
            "split_group_counts": dict(group_split_counts),
            "excluded_acquisitions": list(EXCLUDED_ACQUISITIONS),
            "outputs": {
                "all": str(
                    final_benchmark_root / "manifests" / "crystals_all.csv"
                ),
                "train": str(
                    final_benchmark_root / "manifests" / "crystals_train.csv"
                ),
                "validation": str(
                    final_benchmark_root
                    / "manifests"
                    / "crystals_validation.csv"
                ),
                "test": str(
                    final_benchmark_root / "manifests" / "crystals_test.csv"
                ),
                "errors": str(
                    final_benchmark_root / "metadata" / "freeze_errors.tsv"
                ),
                "group_assignments": str(
                    final_benchmark_root
                    / "metadata"
                    / "split_group_assignments.csv"
                ),
                "locality_counts": str(
                    final_benchmark_root
                    / "metadata"
                    / "split_counts_by_locality.csv"
                ),
                "manifest_order_lineage": str(
                    final_benchmark_root
                    / "metadata"
                    / "manifest_order_lineage_v1_to_v2.csv"
                ),
            },
            "checks": {
                "all_rows_assigned": True,
                "split_row_counts_expected": True,
                "manifest_order_preserved_from_v2_canonical": True,
                "canonical_manifest_order_contiguous": True,
                "no_invalid_split_labels": True,
                "no_split_group_leakage": True,
                "excluded_acquisitions_train_only_in_parent": True,
                "excluded_acquisitions_absent": True,
                "validation_membership_identical_to_v1": True,
                "test_membership_identical_to_v1": True,
            },
        }

        benchmark_summary_path = (
            benchmark_root
            / "metadata"
            / "pooled_broad_split_summary_v2.json"
        )
        write_json(benchmark_summary_path, benchmark_summary)

        root_readme = """# PlagNetPlus unified SSL dataset v2

This is an immutable, index-only derivative of `ssl_unified_v1`. Image files
remain in the validated source datasets and are not copied or modified.

Version 2 excludes the following project-designated redundant acquisition
variants from the training population:

- `6861_recoated`: 4,309 crystal rows;
- `M83-7-1-II_no_overlap`: 6,016 crystal rows.

These are treated as redundant acquisition or physical-material variants, not
as byte-identical images. The v1 audit found no exact-content duplicate groups.

The corrected dataset contains 123,264 rows from 27 acquisitions. Canonical v2
`manifest_order` values are contiguous. The benchmark metadata contains a
v1-to-v2 order-lineage table. Resolve images using `metadata/image_root_map.json`.
"""
        (build_root / "README.md").write_text(root_readme, encoding="utf-8")

        benchmark_readme = f"""# Pooled broad benchmark: 80/10/10 v2

This corrected benchmark retains the frozen v1 split membership while removing
`6861_recoated` and `M83-7-1-II_no_overlap` from training.

- Training: {EXPECTED_V2_SPLIT_COUNTS['train']:,} images
- Validation: {EXPECTED_V2_SPLIT_COUNTS['validation']:,} images
- Test: {EXPECTED_V2_SPLIT_COUNTS['test']:,} images
- Total: {EXPECTED_V2_ROWS:,} images

Validation and test crystal memberships are unchanged from v1. No images are
copied or modified. Normalisation statistics must be recalculated from the v2
training manifest before training any model.

The selected v1 candidate assignment remains
`{parent_benchmark_summary.get('selected_candidate')}`. Mt St Helens retains the
same v1 caveat: it has no independent validation sample and only 84 pooled-test
crystals.
"""
        (benchmark_root / "README.md").write_text(
            benchmark_readme,
            encoding="utf-8",
        )

        root_log = (
            "UNIFIED SSL MANIFEST V2 BUILD: PASS\n"
            f"created_utc={created_utc}\n"
            f"parent={v1_root}\n"
            f"rows={EXPECTED_V2_ROWS}\n"
            f"rows_removed={sum(excluded_counts.values())}\n"
            "image_files_accessed=NO\n"
            "test_image_data_accessed=NO\n"
        )
        (build_root / "unified_ssl_manifest_build_v2.log").write_text(
            root_log,
            encoding="utf-8",
        )

        benchmark_log = (
            "POOLED BROAD SPLIT FREEZE V2: PASS\n"
            f"created_utc={created_utc}\n"
            f"benchmark_id={args.benchmark_id}\n"
            f"train={split_counts['train']}\n"
            f"validation={split_counts['validation']}\n"
            f"test={split_counts['test']}\n"
            "validation_membership_identical_to_v1=YES\n"
            "test_membership_identical_to_v1=YES\n"
            "image_files_accessed=NO\n"
            "test_image_data_accessed=NO\n"
        )
        (
            benchmark_root / "pooled_broad_split_freeze_v2.log"
        ).write_text(benchmark_log, encoding="utf-8")

        root_checksum_paths = [
            Path("README.md"),
            Path("manifests/build_errors.tsv"),
            Path("manifests/crystals_broad.csv"),
            Path("metadata/exclusions_v2.csv"),
            Path("metadata/image_root_map.json"),
            Path("metadata/split_groups.csv"),
            Path("metadata/unified_ssl_manifest_summary_v2.json"),
        ]
        write_checksum_file(build_root, root_checksum_paths)

        benchmark_checksum_paths = [
            Path("README.md"),
            Path("manifests/crystals_all.csv"),
            Path("manifests/crystals_train.csv"),
            Path("manifests/crystals_validation.csv"),
            Path("manifests/crystals_test.csv"),
            Path("metadata/freeze_errors.tsv"),
            Path("metadata/manifest_order_lineage_v1_to_v2.csv"),
            Path("metadata/pooled_broad_split_summary_v2.json"),
            Path("metadata/split_counts_by_locality.csv"),
            Path("metadata/split_group_assignments.csv"),
        ]
        write_checksum_file(benchmark_root, benchmark_checksum_paths)

        print("========== FINAL PRE-INSTALLATION CHECKS ==========")
        if sha256_file(canonical_parent) != parent_canonical_hash:
            raise RuntimeError("Parent canonical manifest changed during build.")
        check_parent_hashes(v1_root)

        expected_outputs = [
            canonical_output,
            all_output,
            *split_outputs.values(),
            root_summary_path,
            benchmark_summary_path,
            lineage_output,
            build_root / "SHA256SUMS",
            benchmark_root / "SHA256SUMS",
        ]
        for path in expected_outputs:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty v2 output: {path}")

        os.rename(build_root, v2_root)
        installed = True

        print("========== UNIFIED SSL V2 CREATED ==========")
        print("V2 root:", v2_root)
        print("Benchmark:", v2_root / "benchmarks" / args.benchmark_id)
        print("Rows:", f"{EXPECTED_V2_ROWS:,}")
        print("Training:", f"{split_counts['train']:,}")
        print("Validation:", f"{split_counts['validation']:,}")
        print("Test:", f"{split_counts['test']:,}")
        print("Removed:", dict(excluded_counts))
        print("Validation membership unchanged: YES")
        print("Test membership unchanged: YES")
        print("V1 modified: NO")
        print("IMAGE FILES ACCESSED: NO")
        print("TEST IMAGE DATA ACCESSED: NO")
        print("NORMALISATION CALCULATED: NO")

    finally:
        if not installed and build_root.exists():
            shutil.rmtree(build_root)


if __name__ == "__main__":
    main()
