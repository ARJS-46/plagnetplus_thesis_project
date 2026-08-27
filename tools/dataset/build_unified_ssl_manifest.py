#!/usr/bin/env python3

from __future__ import annotations

import os
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IRP_ROOT = Path(
    os.environ.get(
        "PNP_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

NON_AEGINA_ROOT = (
    IRP_ROOT
    / "data/processed/plagnetplus"
    / "ssl_non_aegina_v1"
)

AEGINA_ROOT = (
    IRP_ROOT
    / "data/processed/plagnetplus"
    / "ssl_aegina_compat_v1"
)

NON_AEGINA_MANIFEST = (
    NON_AEGINA_ROOT
    / "manifests/crystals_broad.csv"
)

AEGINA_MANIFEST = (
    AEGINA_ROOT
    / "manifests/crystals_broad.csv"
)

OUTPUT_ROOT = (
    IRP_ROOT
    / "data/processed/plagnetplus"
    / "ssl_unified_v1"
)

MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
METADATA_ROOT = OUTPUT_ROOT / "metadata"

OUTPUT_MANIFEST = (
    MANIFEST_ROOT
    / "crystals_broad.csv"
)

ERROR_REPORT = (
    MANIFEST_ROOT
    / "build_errors.tsv"
)

SPLIT_GROUP_REPORT = (
    METADATA_ROOT
    / "split_groups.csv"
)

ROOT_MAP_PATH = (
    METADATA_ROOT
    / "image_root_map.json"
)

SUMMARY_PATH = (
    METADATA_ROOT
    / "unified_ssl_manifest_summary_v1.json"
)

README_PATH = OUTPUT_ROOT / "README.md"

EXPECTED_NON_AEGINA = 121_245
EXPECTED_AEGINA = 12_344
EXPECTED_TOTAL = 133_589
EXPECTED_SPLIT_GROUPS = 28
EXPECTED_ACQUISITIONS = 29

SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

CANONICAL_FIELDS = [
    "manifest_order",
    "crystal_id",
    "source_dataset",
    "source_dataset_id",
    "source_dataset_version",
    "source_schema_version",
    "source_identifier_field",
    "source_identifier",
    "source_manifest",
    "source_manifest_order",
    "source_manifest_row_1_based",
    "locality",
    "sample_id",
    "physical_sample_id",
    "physical_sample_id_method",
    "acquisition_id",
    "scan_id",
    "scan_id_method",
    "split_group_id",
    "benchmark_split",
    "image_root_key",
    "image_relative_path",
    "image_locator",
    "image_sha256",
    "image_width_px",
    "image_height_px",
    "image_side_px",
    "image_representation_contract",
    "margin_fraction",
    "minimum_margin_px",
    "margin_px",
    "padding_x_px",
    "padding_y_px",
    "resize_applied",
    "source_tile_context_available",
    "source_exact_mask_available",
    "pixel_provenance_note",
    "source_image_relative_path",
    "source_image_sha256",
    "source_tile_id",
    "source_tile_index_1",
    "source_tile_index_2",
    "source_instance_index",
    "segmentation_score",
    "mask_area_px",
    "bbox_width_px",
    "bbox_height_px",
    "touches_source_edge",
    "border_distance_px",
    "geometry_source",
    "ownership_policy_version",
    "qc_status",
    "source_exporter_version",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def parse_int(value: Any) -> int:
    return int(float(str(value).strip()))


def bool_text(value: Any) -> str:
    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return "true"

    if text in {"false", "0", "no", "n"}:
        return "false"

    return ""


def portable_relative_path(value: Any) -> str:
    return Path(
        str(value)
        .strip()
        .replace("\\", "/")
    ).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


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


def write_csv(
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


def normalise_non_aegina(
    row: dict[str, str],
    source_order: int,
) -> dict[str, Any]:
    locality = row["locality"].strip()
    sample_id = row["sample_id"].strip()

    acquisition_id = sample_id

    if (
        locality == "FCT"
        and sample_id
        in {
            "6861_FCT",
            "6861_recoated",
        }
    ):
        physical_sample_id = "6861"
        physical_sample_id_method = (
            "known_multi_acquisition_mapping"
        )
    else:
        physical_sample_id = sample_id
        physical_sample_id_method = (
            "sample_id_proxy"
        )

    split_group_id = (
        f"{locality}__{physical_sample_id}"
    )

    image_relative_path = (
        portable_relative_path(
            row["masked_relative_path"]
        )
    )

    image_width = parse_int(
        row["output_width_px"]
    )

    image_height = parse_int(
        row["output_height_px"]
    )

    source_instance = (
        row.get(
            "instance_label",
            "",
        ).strip()
        or row.get(
            "detection_index",
            "",
        ).strip()
    )

    exact_mask_area = (
        row.get(
            "exact_mask_area_px",
            "",
        ).strip()
        or row.get(
            "mask_area_px",
            "",
        ).strip()
    )

    bbox_width = (
        row.get(
            "exact_bbox_width_px",
            "",
        ).strip()
        or row.get(
            "bbox_width_px",
            "",
        ).strip()
    )

    bbox_height = (
        row.get(
            "exact_bbox_height_px",
            "",
        ).strip()
        or row.get(
            "bbox_height_px",
            "",
        ).strip()
    )

    touches_edge = (
        row.get(
            "exact_touches_tile_border",
            "",
        ).strip()
        or row.get(
            "touches_tile_border",
            "",
        ).strip()
    )

    border_distance = (
        row.get(
            "exact_border_distance_px",
            "",
        ).strip()
        or row.get(
            "border_distance_px",
            "",
        ).strip()
    )

    return {
        "crystal_id": (
            row["candidate_id"].strip()
        ),
        "source_dataset": (
            "non_aegina"
        ),
        "source_dataset_id": (
            "plagnetplus_ssl_non_aegina"
        ),
        "source_dataset_version": "v1",
        "source_schema_version": (
            row.get(
                "exporter_version",
                "",
            ).strip()
        ),
        "source_identifier_field": (
            "candidate_id"
        ),
        "source_identifier": (
            row["candidate_id"].strip()
        ),
        "source_manifest": (
            NON_AEGINA_MANIFEST
            .relative_to(IRP_ROOT)
            .as_posix()
        ),
        "source_manifest_order": (
            source_order
        ),
        "source_manifest_row_1_based": (
            source_order + 2
        ),
        "locality": locality,
        "sample_id": sample_id,
        "physical_sample_id": (
            physical_sample_id
        ),
        "physical_sample_id_method": (
            physical_sample_id_method
        ),
        "acquisition_id": acquisition_id,
        "scan_id": (
            f"{locality}/{acquisition_id}"
        ),
        "scan_id_method": (
            "locality_and_acquisition_proxy"
        ),
        "split_group_id": (
            split_group_id
        ),
        "benchmark_split": "",
        "image_root_key": (
            "non_aegina"
        ),
        "image_relative_path": (
            image_relative_path
        ),
        "image_locator": (
            "non_aegina::"
            + image_relative_path
        ),
        "image_sha256": (
            row["masked_sha256"]
            .strip()
            .lower()
        ),
        "image_width_px": image_width,
        "image_height_px": image_height,
        "image_side_px": max(
            image_width,
            image_height,
        ),
        "image_representation_contract": (
            "original_source_image"
            "__exact_clean_mask"
            "__margin_fraction_0.10"
            "__minimum_margin_8px"
            "__square_pad"
            "__no_resize"
        ),
        "margin_fraction": 0.10,
        "minimum_margin_px": 8,
        "margin_px": (
            row.get(
                "crop_margin_px",
                "",
            ).strip()
        ),
        "padding_x_px": (
            row.get(
                "padding_x_px",
                "",
            ).strip()
        ),
        "padding_y_px": (
            row.get(
                "padding_y_px",
                "",
            ).strip()
        ),
        "resize_applied": "false",
        "source_tile_context_available": (
            "true"
        ),
        "source_exact_mask_available": (
            "true"
        ),
        "pixel_provenance_note": (
            "Pixels sampled from the original "
            "source image; exact clean instance "
            "mask applied; background set to black."
        ),
        "source_image_relative_path": (
            portable_relative_path(
                row.get(
                    "source_relative_path",
                    "",
                )
            )
        ),
        "source_image_sha256": (
            row.get(
                "source_sha256",
                "",
            ).strip().lower()
        ),
        "source_tile_id": (
            row.get(
                "tile_id",
                "",
            ).strip()
        ),
        "source_tile_index_1": (
            row.get(
                "grid_index_1",
                "",
            ).strip()
        ),
        "source_tile_index_2": (
            row.get(
                "grid_index_2",
                "",
            ).strip()
        ),
        "source_instance_index": (
            source_instance
        ),
        "segmentation_score": (
            row.get(
                "segmentation_score",
                "",
            ).strip()
        ),
        "mask_area_px": exact_mask_area,
        "bbox_width_px": bbox_width,
        "bbox_height_px": bbox_height,
        "touches_source_edge": (
            bool_text(touches_edge)
        ),
        "border_distance_px": (
            border_distance
        ),
        "geometry_source": (
            row.get(
                "exact_mask_mode",
                "",
            ).strip()
            or row.get(
                "mask_geometry_source",
                "",
            ).strip()
        ),
        "ownership_policy_version": (
            row.get(
                "ownership_policy_version",
                "",
            ).strip()
        ),
        "qc_status": (
            "pass_broad"
            if bool_text(
                row.get(
                    "valid_broad",
                    "",
                )
            )
            == "true"
            else "fail"
        ),
        "source_exporter_version": (
            row.get(
                "exporter_version",
                "",
            ).strip()
        ),
    }


def normalise_aegina(
    row: dict[str, str],
    source_order: int,
) -> dict[str, Any]:
    image_relative_path = (
        portable_relative_path(
            row["masked_relative_path"]
        )
    )

    image_width = parse_int(
        row["output_width_px"]
    )

    image_height = parse_int(
        row["output_height_px"]
    )

    return {
        "crystal_id": (
            row["crystal_id"].strip()
        ),
        "source_dataset": "aegina",
        "source_dataset_id": (
            row["dataset_id"].strip()
        ),
        "source_dataset_version": (
            row["dataset_version"].strip()
        ),
        "source_schema_version": (
            row["schema_version"].strip()
        ),
        "source_identifier_field": (
            "crystal_id"
        ),
        "source_identifier": (
            row["crystal_id"].strip()
        ),
        "source_manifest": (
            AEGINA_MANIFEST
            .relative_to(IRP_ROOT)
            .as_posix()
        ),
        "source_manifest_order": (
            source_order
        ),
        "source_manifest_row_1_based": (
            source_order + 2
        ),
        "locality": (
            row["locality"].strip()
        ),
        "sample_id": (
            row["sample_id"].strip()
        ),
        "physical_sample_id": (
            row["physical_sample_id"]
            .strip()
        ),
        "physical_sample_id_method": (
            "source_manifest"
        ),
        "acquisition_id": (
            row["acquisition_id"].strip()
        ),
        "scan_id": (
            row["scan_id"].strip()
        ),
        "scan_id_method": (
            row["scan_id_method"].strip()
        ),
        "split_group_id": (
            row["split_group_id"].strip()
        ),
        "benchmark_split": "",
        "image_root_key": "aegina",
        "image_relative_path": (
            image_relative_path
        ),
        "image_locator": (
            "aegina::"
            + image_relative_path
        ),
        "image_sha256": (
            row["masked_sha256"]
            .strip()
            .lower()
        ),
        "image_width_px": image_width,
        "image_height_px": image_height,
        "image_side_px": max(
            image_width,
            image_height,
        ),
        "image_representation_contract": (
            "legacy_masked_bbox_crop"
            "__black_margin_fraction_0.10"
            "__minimum_margin_8px"
            "__square_pad"
            "__no_resize"
        ),
        "margin_fraction": (
            row["margin_fraction"].strip()
        ),
        "minimum_margin_px": (
            row["minimum_margin_px"]
            .strip()
        ),
        "margin_px": (
            row["margin_px"].strip()
        ),
        "padding_x_px": (
            row["padding_x_px"].strip()
        ),
        "padding_y_px": (
            row["padding_y_px"].strip()
        ),
        "resize_applied": (
            bool_text(
                row["resize_applied"]
            )
        ),
        "source_tile_context_available": (
            bool_text(
                row[
                    "source_tile_context_available"
                ]
            )
        ),
        "source_exact_mask_available": (
            bool_text(
                row[
                    "source_exact_mask_available"
                ]
            )
        ),
        "pixel_provenance_note": (
            "Every pixel from the legacy masked "
            "bbox PNG is preserved exactly; black "
            "compatibility margin and square padding "
            "added without resizing."
        ),
        "source_image_relative_path": (
            portable_relative_path(
                row["source_relative_path"]
            )
        ),
        "source_image_sha256": (
            row["source_sha256"]
            .strip()
            .lower()
        ),
        "source_tile_id": "",
        "source_tile_index_1": (
            row["tile_index_1"].strip()
        ),
        "source_tile_index_2": (
            row["tile_index_2"].strip()
        ),
        "source_instance_index": (
            row["instance_index"].strip()
        ),
        "segmentation_score": "",
        "mask_area_px": "",
        "bbox_width_px": (
            row["source_width_px"].strip()
        ),
        "bbox_height_px": (
            row["source_height_px"].strip()
        ),
        "touches_source_edge": (
            bool_text(
                row[
                    "audit_nonblack_bbox_touches_source_edge"
                ]
            )
        ),
        "border_distance_px": "",
        "geometry_source": (
            "legacy_bbox_crop"
        ),
        "ownership_policy_version": "",
        "qc_status": (
            row[
                "compatibility_qc_status"
            ].strip()
        ),
        "source_exporter_version": (
            row["schema_version"].strip()
        ),
    }


for directory in (
    OUTPUT_ROOT,
    MANIFEST_ROOT,
    METADATA_ROOT,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# Load the two source datasets independently, normalise their schemas, then
# combine them under one portable unified-manifest contract.
non_aegina_rows = read_csv(
    NON_AEGINA_MANIFEST
)

aegina_rows = read_csv(
    AEGINA_MANIFEST
)

normalised_rows: list[
    dict[str, Any]
] = []

errors: list[dict[str, Any]] = []

for source_order, row in enumerate(
    non_aegina_rows
):
    try:
        normalised_rows.append(
            normalise_non_aegina(
                row,
                source_order,
            )
        )
    except Exception as exc:
        errors.append({
            "source_dataset": (
                "non_aegina"
            ),
            "source_manifest_order": (
                source_order
            ),
            "source_identifier": (
                row.get(
                    "candidate_id",
                    "",
                )
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error": repr(exc),
        })

for source_order, row in enumerate(
    aegina_rows
):
    try:
        normalised_rows.append(
            normalise_aegina(
                row,
                source_order,
            )
        )
    except Exception as exc:
        errors.append({
            "source_dataset": "aegina",
            "source_manifest_order": (
                source_order
            ),
            "source_identifier": (
                row.get(
                    "crystal_id",
                    "",
                )
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error": repr(exc),
        })

# Assign a deterministic global manifest order only after both source
# collections have been normalised and sorted.
for manifest_order, row in enumerate(
    normalised_rows
):
    row["manifest_order"] = (
        manifest_order
    )

root_paths = {
    "non_aegina": NON_AEGINA_ROOT,
    "aegina": AEGINA_ROOT,
}

missing_images = []
invalid_hashes = []
nonsquare_images = []
nonportable_paths = []

for row in normalised_rows:
    relative = Path(
        row["image_relative_path"]
    )

    if (
        relative.is_absolute()
        or ".." in relative.parts
    ):
        nonportable_paths.append(
            row["crystal_id"]
        )
        continue

    image_path = (
        root_paths[
            row["image_root_key"]
        ]
        / relative
    )

    if not image_path.is_file():
        missing_images.append({
            "source_dataset": (
                row["source_dataset"]
            ),
            "source_manifest_order": (
                row[
                    "source_manifest_order"
                ]
            ),
            "source_identifier": (
                row["source_identifier"]
            ),
            "error_type": (
                "missing_image"
            ),
            "error": str(image_path),
        })

    if not SHA256_PATTERN.fullmatch(
        row["image_sha256"]
    ):
        invalid_hashes.append(
            row["crystal_id"]
        )

    if (
        row["image_width_px"]
        != row["image_height_px"]
    ):
        nonsquare_images.append(
            row["crystal_id"]
        )

errors.extend(missing_images)

crystal_id_counts = Counter(
    row["crystal_id"]
    for row in normalised_rows
)

locator_counts = Counter(
    row["image_locator"]
    for row in normalised_rows
)

hash_counts = Counter(
    row["image_sha256"]
    for row in normalised_rows
)

locality_counts = Counter(
    row["locality"]
    for row in normalised_rows
)

source_dataset_counts = Counter(
    row["source_dataset"]
    for row in normalised_rows
)

sample_counts = Counter(
    (
        row["locality"],
        row["sample_id"],
    )
    for row in normalised_rows
)

physical_sample_counts = Counter(
    (
        row["locality"],
        row[
            "physical_sample_id"
        ],
    )
    for row in normalised_rows
)

acquisition_counts = Counter(
    (
        row["locality"],
        row["acquisition_id"],
    )
    for row in normalised_rows
)

split_group_rows: defaultdict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)

for row in normalised_rows:
    split_group_rows[
        row["split_group_id"]
    ].append(row)

# Summarise split groups explicitly because physical-sample grouping is the
# unit later used to prevent leakage between benchmark splits.
split_group_report = []

for split_group_id, rows in sorted(
    split_group_rows.items()
):
    split_group_report.append({
        "split_group_id": (
            split_group_id
        ),
        "locality": (
            sorted({
                row["locality"]
                for row in rows
            })[0]
        ),
        "physical_sample_id": (
            sorted({
                row[
                    "physical_sample_id"
                ]
                for row in rows
            })[0]
        ),
        "acquisitions": ";".join(
            sorted({
                row["acquisition_id"]
                for row in rows
            })
        ),
        "acquisition_count": len({
            row["acquisition_id"]
            for row in rows
        }),
        "crystal_count": len(rows),
    })

fct_6861_rows = [
    row
    for row in normalised_rows
    if row["split_group_id"]
    == "FCT__6861"
]

fct_6861_acquisitions = sorted({
    row["acquisition_id"]
    for row in fct_6861_rows
})

# Fingerprint stable row identity fields so later training runs can verify
# that they use this exact unified dataset definition.
manifest_fingerprint = (
    hashlib.sha256()
)

for row in normalised_rows:
    fingerprint_line = "\t".join([
        str(row["manifest_order"]),
        str(row["crystal_id"]),
        str(row["image_locator"]),
        str(row["image_sha256"]),
        str(row["split_group_id"]),
    ])

    manifest_fingerprint.update(
        (
            fingerprint_line
            + "\n"
        ).encode("utf-8")
    )

write_csv(
    OUTPUT_MANIFEST,
    normalised_rows,
    CANONICAL_FIELDS,
)

write_csv(
    ERROR_REPORT,
    errors,
    [
        "source_dataset",
        "source_manifest_order",
        "source_identifier",
        "error_type",
        "error",
    ],
    delimiter="\t",
)

write_csv(
    SPLIT_GROUP_REPORT,
    split_group_report,
    [
        "split_group_id",
        "locality",
        "physical_sample_id",
        "acquisitions",
        "acquisition_count",
        "crystal_count",
    ],
)

root_map = {
    "schema_version": (
        "plagnetplus_image_root_map_v1"
    ),
    "environment_variable": (
        "PNP_DATA_ROOT"
    ),
    "resolution_rule": (
        "roots[image_root_key] / "
        "image_relative_path"
    ),
    "portable_roots": {
        "non_aegina": (
            "${PNP_DATA_ROOT}/processed/"
            "plagnetplus/ssl_non_aegina_v1"
        ),
        "aegina": (
            "${PNP_DATA_ROOT}/processed/"
            "plagnetplus/ssl_aegina_compat_v1"
        ),
    },
    "current_resolved_roots": {
        "non_aegina": str(
            NON_AEGINA_ROOT
        ),
        "aegina": str(
            AEGINA_ROOT
        ),
    },
}

write_json(
    ROOT_MAP_PATH,
    root_map,
)

checks = {
    "non_aegina_rows_expected": (
        len(non_aegina_rows)
        == EXPECTED_NON_AEGINA
    ),
    "aegina_rows_expected": (
        len(aegina_rows)
        == EXPECTED_AEGINA
    ),
    "combined_rows_expected": (
        len(normalised_rows)
        == EXPECTED_TOTAL
    ),
    "no_build_errors": (
        not errors
    ),
    "crystal_ids_unique": all(
        count == 1
        for count
        in crystal_id_counts.values()
    ),
    "image_locators_unique": all(
        count == 1
        for count
        in locator_counts.values()
    ),
    "no_exact_content_duplicates": all(
        count == 1
        for count
        in hash_counts.values()
    ),
    "all_images_exist": (
        not missing_images
    ),
    "all_paths_portable": (
        not nonportable_paths
    ),
    "all_hashes_valid": (
        not invalid_hashes
    ),
    "all_images_square": (
        not nonsquare_images
    ),
    "manifest_order_contiguous": (
        [
            row["manifest_order"]
            for row in normalised_rows
        ]
        == list(
            range(
                len(normalised_rows)
            )
        )
    ),
    "split_group_count_expected": (
        len(split_group_rows)
        == EXPECTED_SPLIT_GROUPS
    ),
    "acquisition_count_expected": (
        len(acquisition_counts)
        == EXPECTED_ACQUISITIONS
    ),
    "fct_6861_grouping_correct": (
        fct_6861_acquisitions
        == [
            "6861_FCT",
            "6861_recoated",
        ]
        and len(fct_6861_rows)
        == 8_871
    ),
}

status = (
    "PASS"
    if all(checks.values())
    else "FAIL"
)

summary = {
    "schema_version": (
        "plagnetplus_unified_ssl_manifest_v1"
    ),
    "status": status,
    "dataset_id": (
        "plagnetplus_ssl_unified"
    ),
    "dataset_version": "v1",
    "representation_view": "broad",
    "ordering_policy": (
        "non_aegina_source_order"
        "__then__"
        "aegina_source_order"
    ),
    "rows": len(
        normalised_rows
    ),
    "columns": len(
        CANONICAL_FIELDS
    ),
    "source_dataset_counts": dict(
        sorted(
            source_dataset_counts.items()
        )
    ),
    "locality_counts": dict(
        sorted(
            locality_counts.items()
        )
    ),
    "sample_count": len(
        sample_counts
    ),
    "physical_sample_count": len(
        physical_sample_counts
    ),
    "acquisition_count": len(
        acquisition_counts
    ),
    "split_group_count": len(
        split_group_rows
    ),
    "unique_crystal_ids": len(
        crystal_id_counts
    ),
    "unique_image_locators": len(
        locator_counts
    ),
    "exact_content_duplicate_groups": sum(
        count > 1
        for count
        in hash_counts.values()
    ),
    "fct_6861_crystal_count": len(
        fct_6861_rows
    ),
    "fct_6861_acquisitions": (
        fct_6861_acquisitions
    ),
    "manifest_fingerprint_sha256": (
        manifest_fingerprint.hexdigest()
    ),
    "source_manifests": {
        "non_aegina": {
            "path": str(
                NON_AEGINA_MANIFEST
            ),
            "sha256": sha256_file(
                NON_AEGINA_MANIFEST
            ),
        },
        "aegina": {
            "path": str(
                AEGINA_MANIFEST
            ),
            "sha256": sha256_file(
                AEGINA_MANIFEST
            ),
        },
    },
    "checks": checks,
    "outputs": {
        "manifest": str(
            OUTPUT_MANIFEST
        ),
        "errors": str(
            ERROR_REPORT
        ),
        "split_groups": str(
            SPLIT_GROUP_REPORT
        ),
        "image_root_map": str(
            ROOT_MAP_PATH
        ),
    },
}

write_json(
    SUMMARY_PATH,
    summary,
)

readme = """# PlagNetPlus unified SSL dataset v1

This is an index-only dataset. Images remain in the two validated source
datasets and are not copied or modified.

Resolve each image as:

    root_map[row["image_root_key"]] / row["image_relative_path"]

Canonical ordering is:

1. non-Aegina broad manifest in exact source order;
2. Aegina compatibility broad manifest in exact source order.

`benchmark_split` is intentionally blank. Benchmark-specific manifests and
leakage-safe split assignments are created separately from `split_group_id`.
"""

atomic_write_text(
    README_PATH,
    readme,
)

checksum_targets = [
    OUTPUT_MANIFEST,
    ERROR_REPORT,
    SPLIT_GROUP_REPORT,
    ROOT_MAP_PATH,
    SUMMARY_PATH,
    README_PATH,
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
    "UNIFIED SSL MANIFEST BUILD:",
    status,
)

if status != "PASS":
    raise SystemExit(1)
