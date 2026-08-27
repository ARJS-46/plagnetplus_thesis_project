#!/usr/bin/env python3

from __future__ import annotations

import os
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IRP_ROOT = Path(
    os.environ.get(
        "PNP_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

UNIFIED_ROOT = (
    IRP_ROOT
    / "data/processed/plagnetplus"
    / "ssl_unified_v1"
)

CANONICAL_MANIFEST = (
    UNIFIED_ROOT
    / "manifests/crystals_broad.csv"
)

CANDIDATE_ASSIGNMENTS = (
    IRP_ROOT
    / "data/interim/plagnetplus"
    / "candidate_benchmark_splits_v1"
    / "candidate_split_assignments_v1.tsv"
)

OUTPUT_ROOT = (
    UNIFIED_ROOT
    / "benchmarks"
    / "pooled_broad_p80_10_10_v1"
)

MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
METADATA_ROOT = OUTPUT_ROOT / "metadata"

SELECTED_CANDIDATE = (
    "p80_10_10"
    "__mtsthelens_large_train_small_test"
)

BENCHMARK_ID = (
    "pooled_broad_p80_10_10_v1"
)

EXPECTED_ROWS = 133_589
EXPECTED_GROUPS = 28

EXPECTED_SPLIT_ROWS = {
    "train": 101_682,
    "validation": 14_919,
    "test": 16_988,
}

EXPECTED_SPLIT_GROUPS = {
    "train": 16,
    "validation": 5,
    "test": 7,
}

SPLIT_ORDER = (
    "train",
    "validation",
    "test",
)


def read_table(
    path: Path,
    *,
    delimiter: str = ",",
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
        )

        fields = reader.fieldnames or []
        rows = list(reader)

    return fields, rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(path)


def write_json(
    path: Path,
    value: Any,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_table(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


for directory in (
    OUTPUT_ROOT,
    MANIFEST_ROOT,
    METADATA_ROOT,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


canonical_fields, canonical_rows = read_table(
    CANONICAL_MANIFEST
)

_, candidate_rows = read_table(
    CANDIDATE_ASSIGNMENTS,
    delimiter="\t",
)

selected_assignments = [
    row
    for row in candidate_rows
    if row["candidate_id"]
    == SELECTED_CANDIDATE
]

# Validate the candidate split assignment at the split-group level before
# applying it to individual crystal rows.
assignment_counts = Counter(
    row["split_group_id"]
    for row in selected_assignments
)

duplicate_assignment_groups = {
    split_group_id: count
    for split_group_id, count
    in assignment_counts.items()
    if count > 1
}

group_to_split = {
    row["split_group_id"]:
    row["assigned_split"]
    for row in selected_assignments
}

invalid_split_labels = sorted({
    split
    for split in group_to_split.values()
    if split not in SPLIT_ORDER
})

canonical_group_ids = {
    row["split_group_id"]
    for row in canonical_rows
}

assignment_group_ids = set(
    group_to_split
)

missing_assignments = sorted(
    canonical_group_ids
    - assignment_group_ids
)

extra_assignments = sorted(
    assignment_group_ids
    - canonical_group_ids
)

benchmark_fields = list(
    canonical_fields
)

if "benchmark_id" not in benchmark_fields:
    insertion_index = (
        benchmark_fields.index(
            "benchmark_split"
        )
        if "benchmark_split"
        in benchmark_fields
        else 0
    )

    benchmark_fields.insert(
        insertion_index,
        "benchmark_id",
    )

benchmark_rows: list[
    dict[str, Any]
] = []

unassigned_rows = []

# Propagate each physical sample group to one frozen benchmark split; no
# individual crystal is assigned independently.
for row in canonical_rows:
    split_group_id = (
        row["split_group_id"]
    )

    assigned_split = (
        group_to_split.get(
            split_group_id
        )
    )

    if assigned_split is None:
        unassigned_rows.append({
            "crystal_id": (
                row["crystal_id"]
            ),
            "split_group_id": (
                split_group_id
            ),
            "error": (
                "missing_group_assignment"
            ),
        })

        continue

    benchmark_row = dict(row)

    benchmark_row[
        "benchmark_id"
    ] = BENCHMARK_ID

    benchmark_row[
        "benchmark_split"
    ] = assigned_split

    benchmark_rows.append(
        benchmark_row
    )


rows_by_split: defaultdict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)

for row in benchmark_rows:
    rows_by_split[
        row["benchmark_split"]
    ].append(row)


group_split_sets: defaultdict[
    str,
    set[str],
] = defaultdict(set)

for row in benchmark_rows:
    group_split_sets[
        row["split_group_id"]
    ].add(
        row["benchmark_split"]
    )

leaking_groups = {
    split_group_id: sorted(
        split_names
    )
    for split_group_id, split_names
    in group_split_sets.items()
    if len(split_names) != 1
}


split_group_counts = Counter(
    row["assigned_split"]
    for row in selected_assignments
)

split_row_counts = {
    split_name: len(
        rows_by_split[split_name]
    )
    for split_name in SPLIT_ORDER
}


locality_split_counts: defaultdict[
    str,
    Counter[str],
] = defaultdict(Counter)

locality_split_group_sets: defaultdict[
    tuple[str, str],
    set[str],
] = defaultdict(set)

for row in benchmark_rows:
    locality = row["locality"]
    split_name = row[
        "benchmark_split"
    ]

    locality_split_counts[
        locality
    ][split_name] += 1

    locality_split_group_sets[
        (
            locality,
            split_name,
        )
    ].add(
        row["split_group_id"]
    )


locality_summary_rows = []

for locality in sorted(
    locality_split_counts
):
    total = sum(
        locality_split_counts[
            locality
        ].values()
    )

    locality_summary_rows.append({
        "locality": locality,
        "total_crystals": total,
        "train_crystals": (
            locality_split_counts[
                locality
            ]["train"]
        ),
        "validation_crystals": (
            locality_split_counts[
                locality
            ]["validation"]
        ),
        "test_crystals": (
            locality_split_counts[
                locality
            ]["test"]
        ),
        "train_fraction": (
            locality_split_counts[
                locality
            ]["train"]
            / total
        ),
        "validation_fraction": (
            locality_split_counts[
                locality
            ]["validation"]
            / total
        ),
        "test_fraction": (
            locality_split_counts[
                locality
            ]["test"]
            / total
        ),
        "train_group_count": len(
            locality_split_group_sets[
                (
                    locality,
                    "train",
                )
            ]
        ),
        "validation_group_count": len(
            locality_split_group_sets[
                (
                    locality,
                    "validation",
                )
            ]
        ),
        "test_group_count": len(
            locality_split_group_sets[
                (
                    locality,
                    "test",
                )
            ]
        ),
    })


mt_st_helens_assignments = {
    row["split_group_id"]:
    row["assigned_split"]
    for row in selected_assignments
    if row["locality"]
    == "MtStHelens"
}


assignment_output_rows = []

for row in selected_assignments:
    assignment_output_rows.append({
        "benchmark_id": BENCHMARK_ID,
        "candidate_id": (
            row["candidate_id"]
        ),
        "split_group_id": (
            row["split_group_id"]
        ),
        "locality": (
            row["locality"]
        ),
        "physical_sample_id": (
            row["physical_sample_id"]
        ),
        "acquisitions": (
            row["acquisitions"]
        ),
        "crystal_count": int(
            row["crystal_count"]
        ),
        "benchmark_split": (
            row["assigned_split"]
        ),
        "constraint_note": (
            row["constraint_note"]
        ),
    })


# Hash the frozen membership/order fields so model runs can verify that the
# benchmark has not changed after selection.
manifest_fingerprint = hashlib.sha256()

for row in benchmark_rows:
    fingerprint_line = "\t".join([
        row["manifest_order"],
        row["crystal_id"],
        row["split_group_id"],
        row["benchmark_split"],
        row["image_locator"],
        row["image_sha256"],
    ])

    manifest_fingerprint.update(
        (
            fingerprint_line
            + "\n"
        ).encode("utf-8")
    )


full_manifest_path = (
    MANIFEST_ROOT
    / "crystals_all.csv"
)

train_manifest_path = (
    MANIFEST_ROOT
    / "crystals_train.csv"
)

validation_manifest_path = (
    MANIFEST_ROOT
    / "crystals_validation.csv"
)

test_manifest_path = (
    MANIFEST_ROOT
    / "crystals_test.csv"
)

assignment_path = (
    METADATA_ROOT
    / "split_group_assignments.csv"
)

locality_summary_path = (
    METADATA_ROOT
    / "split_counts_by_locality.csv"
)

error_path = (
    METADATA_ROOT
    / "freeze_errors.tsv"
)

summary_path = (
    METADATA_ROOT
    / "pooled_broad_split_summary_v1.json"
)


write_table(
    full_manifest_path,
    benchmark_rows,
    benchmark_fields,
)

write_table(
    train_manifest_path,
    rows_by_split["train"],
    benchmark_fields,
)

write_table(
    validation_manifest_path,
    rows_by_split["validation"],
    benchmark_fields,
)

write_table(
    test_manifest_path,
    rows_by_split["test"],
    benchmark_fields,
)

write_table(
    assignment_path,
    assignment_output_rows,
    [
        "benchmark_id",
        "candidate_id",
        "split_group_id",
        "locality",
        "physical_sample_id",
        "acquisitions",
        "crystal_count",
        "benchmark_split",
        "constraint_note",
    ],
)

write_table(
    locality_summary_path,
    locality_summary_rows,
    [
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
    ],
)

write_table(
    error_path,
    unassigned_rows,
    [
        "crystal_id",
        "split_group_id",
        "error",
    ],
    delimiter="\t",
)


checks = {
    "canonical_rows_expected": (
        len(canonical_rows)
        == EXPECTED_ROWS
    ),
    "selected_assignment_count_expected": (
        len(selected_assignments)
        == EXPECTED_GROUPS
    ),
    "assignment_groups_unique": (
        not duplicate_assignment_groups
    ),
    "no_invalid_split_labels": (
        not invalid_split_labels
    ),
    "assignment_groups_match_manifest": (
        not missing_assignments
        and not extra_assignments
    ),
    "all_rows_assigned": (
        len(benchmark_rows)
        == len(canonical_rows)
        and not unassigned_rows
    ),
    "split_row_counts_expected": (
        split_row_counts
        == EXPECTED_SPLIT_ROWS
    ),
    "split_group_counts_expected": (
        dict(split_group_counts)
        == EXPECTED_SPLIT_GROUPS
    ),
    "no_split_group_leakage": (
        not leaking_groups
    ),
    "manifest_order_preserved": (
        [
            row["manifest_order"]
            for row in benchmark_rows
        ]
        == [
            row["manifest_order"]
            for row in canonical_rows
        ]
    ),
    "mt_st_helens_orientation_correct": (
        mt_st_helens_assignments
        == {
            "MtStHelens__71512":
            "train",
            "MtStHelens__18_1a":
            "test",
        }
    ),
    "all_non_mtsthelens_localities_in_all_splits": all(
        (
            row["train_group_count"] > 0
            and row[
                "validation_group_count"
            ] > 0
            and row[
                "test_group_count"
            ] > 0
        )
        for row in locality_summary_rows
        if row["locality"]
        != "MtStHelens"
    ),
    "mtsthelens_validation_intentionally_empty": (
        next(
            row
            for row in locality_summary_rows
            if row["locality"]
            == "MtStHelens"
        )["validation_crystals"]
        == 0
    ),
}

status = (
    "PASS"
    if all(checks.values())
    else "FAIL"
)


summary = {
    "schema_version": (
        "pooled_broad_split_v1"
    ),
    "status": status,
    "benchmark_id": BENCHMARK_ID,
    "selected_candidate": (
        SELECTED_CANDIDATE
    ),
    "source_manifest": str(
        CANONICAL_MANIFEST
    ),
    "source_manifest_sha256": (
        sha256_file(
            CANONICAL_MANIFEST
        )
    ),
    "candidate_assignment_source": str(
        CANDIDATE_ASSIGNMENTS
    ),
    "candidate_assignment_source_sha256": (
        sha256_file(
            CANDIDATE_ASSIGNMENTS
        )
    ),
    "rows": len(
        benchmark_rows
    ),
    "split_row_counts": (
        split_row_counts
    ),
    "split_row_fractions": {
        split_name: (
            split_row_counts[
                split_name
            ]
            / len(benchmark_rows)
        )
        for split_name in SPLIT_ORDER
    },
    "split_group_counts": {
        split_name: (
            split_group_counts[
                split_name
            ]
        )
        for split_name in SPLIT_ORDER
    },
    "mt_st_helens_assignments": (
        mt_st_helens_assignments
    ),
    "manifest_fingerprint_sha256": (
        manifest_fingerprint.hexdigest()
    ),
    "checks": checks,
    "outputs": {
        "all": str(
            full_manifest_path
        ),
        "train": str(
            train_manifest_path
        ),
        "validation": str(
            validation_manifest_path
        ),
        "test": str(
            test_manifest_path
        ),
        "group_assignments": str(
            assignment_path
        ),
        "locality_counts": str(
            locality_summary_path
        ),
        "errors": str(
            error_path
        ),
    },
    "interpretation": (
        "Frozen pooled broad benchmark. "
        "All assignment is by leakage-safe "
        "split_group_id. Mt St Helens has no "
        "validation group because only two "
        "independent physical samples exist."
    ),
    "mt_st_helens_caveat": (
        "The pooled test split contains only "
        "84 Mt St Helens crystals. Use the "
        "leave-Mt-St-Helens-out benchmark and "
        "two-group orientation sensitivity for "
        "locality-specific conclusions."
    ),
}

write_json(
    summary_path,
    summary,
)


readme = f"""# Pooled broad benchmark: 80/10/10 v1

Selected candidate:

    {SELECTED_CANDIDATE}

This is a benchmark-specific index layer. No images are copied or modified.

Assignments are made exclusively through `split_group_id`; crystals from the
same physical sample cannot cross train, validation, and test.

Mt St Helens contains only two physical samples. The larger sample is assigned
to training and the 84-crystal sample to testing. There is no independent
Mt St Helens validation sample in this benchmark.

Do not use the 84-crystal pooled test subset for strong Mt St Helens-specific
claims. Use the leave-Mt-St-Helens-out and two-group orientation sensitivity
benchmarks for that purpose.
"""

atomic_write_text(
    OUTPUT_ROOT / "README.md",
    readme,
)


checksum_targets = [
    full_manifest_path,
    train_manifest_path,
    validation_manifest_path,
    test_manifest_path,
    assignment_path,
    locality_summary_path,
    error_path,
    summary_path,
    OUTPUT_ROOT / "README.md",
]

checksum_lines = []

for path in sorted(
    checksum_targets
):
    checksum_lines.append(
        f"{sha256_file(path)}  "
        f"{path.relative_to(OUTPUT_ROOT).as_posix()}\n"
    )

atomic_write_text(
    OUTPUT_ROOT / "SHA256SUMS",
    "".join(checksum_lines),
)


print("=" * 72)
print(json.dumps(
    summary,
    indent=2,
    sort_keys=True,
))
print("=" * 72)
print(
    "POOLED BROAD SPLIT FREEZE:",
    status,
)

if status != "PASS":
    raise SystemExit(1)
