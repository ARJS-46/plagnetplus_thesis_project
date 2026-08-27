#!/usr/bin/env python3

from __future__ import annotations

import os
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IRP_ROOT = Path(
    os.environ.get(
        "PNP_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

SOURCE_MANIFEST = (
    IRP_ROOT
    / "data/manifests/plagnetplus"
    / "aegina_plagnetplus_manifest_v1.csv"
)

SOURCE_IMAGE_ROOT = (
    IRP_ROOT
    / "data/segmented/aegina"
)

SOURCE_AUDIT = (
    IRP_ROOT
    / "data/interim/plagnetplus"
    / "aegina_image_contract_audit_v1"
    / "aegina_per_image_audit_v1.csv"
)

OUTPUT_ROOT = (
    IRP_ROOT
    / "data/processed/plagnetplus"
    / "ssl_aegina_compat_v1"
)

OUTPUT_IMAGE_ROOT = (
    OUTPUT_ROOT
    / "images_masked"
    / "Aegina"
)

MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
PER_SAMPLE_ROOT = MANIFEST_ROOT / "per_sample"
COMPLETION_ROOT = OUTPUT_ROOT / "completion"
METADATA_ROOT = OUTPUT_ROOT / "metadata"

EXPECTED_ROWS = 12_344
EXPECTED_SAMPLES = 8

MARGIN_FRACTION = 0.10
MINIMUM_MARGIN_PX = 8

DATASET_ID = "plagnetplus_ssl_aegina_compat"
DATASET_VERSION = "v1"
SCHEMA_VERSION = "aegina_ssl_compatibility_export_v1"

TRANSFORM_NAME = (
    "legacy_bbox_masked_crop"
    "__black_margin"
    "__square_pad"
)

SOURCE_CONTEXT = "legacy_masked_bbox_crop"
BACKGROUND_RGB = (0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_int(value: Any) -> int:
    return int(float(str(value).strip()))


def parse_float(value: Any) -> float:
    result = float(str(value).strip())

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value: {value!r}"
        )

    return result


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


def save_png_atomic(
    image: Image.Image,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".png.partial"
    )

    image.save(
        temporary,
        format="PNG",
        optimize=False,
    )

    temporary.replace(path)


def compatibility_transform(
    source: Image.Image,
) -> tuple[Image.Image, dict[str, int]]:
    source = source.convert("RGB")

    source_width, source_height = (
        source.size
    )

    source_longest_side = max(
        source_width,
        source_height,
    )

    margin_px = max(
        MINIMUM_MARGIN_PX,
        math.ceil(
            source_longest_side
            * MARGIN_FRACTION
        ),
    )

    bordered_width = (
        source_width + 2 * margin_px
    )

    bordered_height = (
        source_height + 2 * margin_px
    )

    bordered = Image.new(
        "RGB",
        (
            bordered_width,
            bordered_height,
        ),
        BACKGROUND_RGB,
    )

    bordered.paste(
        source,
        (margin_px, margin_px),
    )

    output_side = max(
        bordered_width,
        bordered_height,
    )

    padding_x = (
        output_side - bordered_width
    ) // 2

    padding_y = (
        output_side - bordered_height
    ) // 2

    output = Image.new(
        "RGB",
        (output_side, output_side),
        BACKGROUND_RGB,
    )

    output.paste(
        bordered,
        (padding_x, padding_y),
    )

    return output, {
        "margin_px": margin_px,
        "bordered_width_px": (
            bordered_width
        ),
        "bordered_height_px": (
            bordered_height
        ),
        "padding_x_px": padding_x,
        "padding_y_px": padding_y,
        "source_offset_x_px": (
            padding_x + margin_px
        ),
        "source_offset_y_px": (
            padding_y + margin_px
        ),
        "output_side_px": output_side,
    }


for directory in (
    OUTPUT_IMAGE_ROOT,
    MANIFEST_ROOT,
    PER_SAMPLE_ROOT,
    COMPLETION_ROOT,
    METADATA_ROOT,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

with SOURCE_MANIFEST.open(
    newline="",
    encoding="utf-8",
) as handle:
    source_reader = csv.DictReader(handle)
    source_rows = list(source_reader)

with SOURCE_AUDIT.open(
    newline="",
    encoding="utf-8",
) as handle:
    audit_reader = csv.DictReader(handle)
    audit_rows = list(audit_reader)

if len(source_rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} source rows; "
        f"found {len(source_rows):,}"
    )

if len(audit_rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS:,} audit rows; "
        f"found {len(audit_rows):,}"
    )

audit_by_path = {
    Path(
        row["relative_path"]
        .replace("\\", "/")
    ).as_posix(): row
    for row in audit_rows
}

if len(audit_by_path) != len(audit_rows):
    raise RuntimeError(
        "Duplicate relative paths in source audit."
    )

output_rows: list[dict[str, Any]] = []
error_rows: list[dict[str, Any]] = []

sample_counts: Counter[str] = Counter()
source_hash_counts: Counter[str] = Counter()
output_hash_counts: Counter[str] = Counter()

output_side_values: list[int] = []
margin_values: list[int] = []

source_pixel_embedding_errors = 0
outer_border_errors = 0
dimension_errors = 0
output_validation_errors = 0

# Re-export inherited Aegina crops through the same square-padding/image
# contract used by the project without altering their underlying segmentation.
for row_index, source_row in enumerate(
    source_rows,
    start=1,
):
    legacy_crystal_id = (
        source_row["crystal_id"].strip()
    )

    sample_id = (
        source_row["sample_id"].strip()
    )

    source_relative_path = Path(
        source_row["relative_path"]
        .replace("\\", "/")
    ).as_posix()

    source_path = (
        SOURCE_IMAGE_ROOT
        / source_relative_path
    )

    audit = audit_by_path.get(
        source_relative_path
    )

    try:
        if audit is None:
            raise RuntimeError(
                "Missing corresponding audit row."
            )

        if not source_path.is_file():
            raise FileNotFoundError(
                source_path
            )

        with Image.open(
            source_path
        ) as image:
            image.load()

            source_format = str(
                image.format or ""
            )

            source = image.convert("RGB")

        source_width = source.width
        source_height = source.height

        manifest_width = parse_int(
            source_row["width_px"]
        )

        manifest_height = parse_int(
            source_row["height_px"]
        )

        dimensions_match = (
            source_width == manifest_width
            and source_height
            == manifest_height
        )

        if not dimensions_match:
            dimension_errors += 1

            raise RuntimeError(
                "Source dimension mismatch: "
                f"manifest="
                f"{manifest_width}x"
                f"{manifest_height}; "
                f"disk="
                f"{source_width}x"
                f"{source_height}"
            )

        transformed, geometry = (
            compatibility_transform(source)
        )

        output_relative_path = (
            Path("images_masked")
            / "Aegina"
            / sample_id
            / source_path.name
        )

        output_path = (
            OUTPUT_ROOT
            / output_relative_path
        )

        save_png_atomic(
            transformed,
            output_path,
        )

        source_array = np.asarray(
            source,
            dtype=np.uint8,
        )

        with Image.open(
            output_path
        ) as output_image:
            output_image.load()

            output_mode = (
                output_image.mode
            )

            output_format = str(
                output_image.format or ""
            )

            output_width = (
                output_image.width
            )

            output_height = (
                output_image.height
            )

            output_array = np.asarray(
                output_image.convert("RGB"),
                dtype=np.uint8,
            )

        source_x = geometry[
            "source_offset_x_px"
        ]

        source_y = geometry[
            "source_offset_y_px"
        ]

        embedded_source = output_array[
            source_y:
            source_y + source_height,
            source_x:
            source_x + source_width,
        ]

        source_pixels_preserved = (
            embedded_source.shape
            == source_array.shape
            and np.array_equal(
                embedded_source,
                source_array,
            )
        )

        if not source_pixels_preserved:
            source_pixel_embedding_errors += 1

        outer_border_is_black = bool(
            np.all(
                output_array[0, :, :]
                == 0
            )
            and np.all(
                output_array[-1, :, :]
                == 0
            )
            and np.all(
                output_array[:, 0, :]
                == 0
            )
            and np.all(
                output_array[:, -1, :]
                == 0
            )
        )

        if not outer_border_is_black:
            outer_border_errors += 1

        output_is_square = (
            output_width
            == output_height
            == geometry[
                "output_side_px"
            ]
        )

        output_is_valid = (
            output_mode == "RGB"
            and output_format == "PNG"
            and output_is_square
            and source_pixels_preserved
            and outer_border_is_black
        )

        if not output_is_valid:
            output_validation_errors += 1

        source_hash = sha256_file(
            source_path
        )

        output_hash = sha256_file(
            output_path
        )

        source_hash_counts[
            source_hash
        ] += 1

        output_hash_counts[
            output_hash
        ] += 1

        global_crystal_id = (
            "aegina__"
            + legacy_crystal_id
        )

        split_group_id = (
            "Aegina__" + sample_id
        )

        result = {
            "crystal_id": (
                global_crystal_id
            ),
            "legacy_crystal_id": (
                legacy_crystal_id
            ),
            "locality": "Aegina",
            "legacy_locality": (
                source_row["locality"]
            ),
            "sample_id": sample_id,
            "physical_sample_id": (
                sample_id
            ),
            "acquisition_id": sample_id,
            "scan_id": (
                source_row["scan_id"]
            ),
            "split_group_id": (
                split_group_id
            ),
            "split": (
                source_row["split"]
            ),
            "dataset_id": DATASET_ID,
            "dataset_version": (
                DATASET_VERSION
            ),
            "schema_version": (
                SCHEMA_VERSION
            ),
            "segmentation_pipeline": (
                source_row[
                    "segmentation_pipeline"
                ]
            ),
            "source_context": (
                SOURCE_CONTEXT
            ),
            "source_tile_context_available": (
                False
            ),
            "source_exact_mask_available": (
                False
            ),
            "transform_class": (
                "compatibility_transform"
            ),
            "transform_name": (
                TRANSFORM_NAME
            ),
            "source_pixels_preserved_exactly": (
                source_pixels_preserved
            ),
            "resize_applied": False,
            "background_r": 0,
            "background_g": 0,
            "background_b": 0,
            "margin_fraction": (
                MARGIN_FRACTION
            ),
            "minimum_margin_px": (
                MINIMUM_MARGIN_PX
            ),
            "margin_px": (
                geometry["margin_px"]
            ),
            "bordered_width_px": (
                geometry[
                    "bordered_width_px"
                ]
            ),
            "bordered_height_px": (
                geometry[
                    "bordered_height_px"
                ]
            ),
            "padding_x_px": (
                geometry[
                    "padding_x_px"
                ]
            ),
            "padding_y_px": (
                geometry[
                    "padding_y_px"
                ]
            ),
            "source_offset_x_px": (
                geometry[
                    "source_offset_x_px"
                ]
            ),
            "source_offset_y_px": (
                geometry[
                    "source_offset_y_px"
                ]
            ),
            "source_width_px": (
                source_width
            ),
            "source_height_px": (
                source_height
            ),
            "source_minimum_dimension_px": (
                min(
                    source_width,
                    source_height,
                )
            ),
            "source_maximum_dimension_px": (
                max(
                    source_width,
                    source_height,
                )
            ),
            "source_aspect_ratio": (
                max(
                    source_width,
                    source_height,
                )
                / min(
                    source_width,
                    source_height,
                )
            ),
            "output_width_px": (
                output_width
            ),
            "output_height_px": (
                output_height
            ),
            "output_side_px": (
                geometry[
                    "output_side_px"
                ]
            ),
            "output_is_square": (
                output_is_square
            ),
            "outer_border_is_black": (
                outer_border_is_black
            ),
            "source_mode": "RGB",
            "source_format": (
                source_format
            ),
            "output_mode": (
                output_mode
            ),
            "output_format": (
                output_format
            ),
            "source_relative_path": (
                source_relative_path
            ),
            "masked_relative_path": (
                output_relative_path
                .as_posix()
            ),
            "source_sha256": (
                source_hash
            ),
            "masked_sha256": (
                output_hash
            ),
            "source_file_size_bytes": (
                source_path.stat().st_size
            ),
            "masked_file_size_bytes": (
                output_path.stat().st_size
            ),
            "legacy_filename": (
                source_row["filename"]
            ),
            "filename_pattern_valid": (
                source_row[
                    "filename_pattern_valid"
                ]
            ),
            "filename_sample": (
                source_row[
                    "filename_sample"
                ]
            ),
            "tile_index_1": (
                source_row[
                    "tile_index_1"
                ]
            ),
            "tile_index_2": (
                source_row[
                    "tile_index_2"
                ]
            ),
            "instance_index": (
                source_row[
                    "instance_index"
                ]
            ),
            "sample_matches_filename": (
                source_row[
                    "sample_matches_filename"
                ]
            ),
            "cutoff_px": (
                source_row["cutoff_px"]
            ),
            "included_at_cutoff": (
                source_row[
                    "included_at_cutoff"
                ]
            ),
            "exclusion_reason": (
                source_row[
                    "exclusion_reason"
                ]
            ),
            "geometry_relative_path": (
                source_row[
                    "geometry_relative_path"
                ]
            ),
            "geometry_file_exists": (
                source_row[
                    "geometry_file_exists"
                ]
            ),
            "scan_id_inferred": (
                source_row[
                    "scan_id_inferred"
                ]
            ),
            "scan_id_method": (
                source_row[
                    "scan_id_method"
                ]
            ),
            "manual_qc_status": (
                source_row[
                    "manual_qc_status"
                ]
            ),
            "legacy_source_manifest_row": (
                source_row[
                    "source_manifest_row"
                ]
            ),
            "legacy_modified_time_utc": (
                source_row[
                    "modified_time_utc"
                ]
            ),
            "audit_black_fraction": (
                parse_float(
                    audit[
                        "black_fraction"
                    ]
                )
            ),
            "audit_border_black_fraction": (
                parse_float(
                    audit[
                        "border_black_fraction"
                    ]
                )
            ),
            "audit_all_four_corners_black": (
                parse_bool(
                    audit[
                        "all_four_corners_black"
                    ]
                )
            ),
            "audit_nonblack_bbox_touches_source_edge": (
                parse_bool(
                    audit[
                        "nonblack_bbox_touches_image_edge"
                    ]
                )
            ),
            "compatibility_qc_status": (
                "pass"
                if output_is_valid
                else "fail"
            ),
            "manifest_order": (
                row_index - 1
            ),
        }

        output_rows.append(result)
        sample_counts[sample_id] += 1

        output_side_values.append(
            geometry["output_side_px"]
        )

        margin_values.append(
            geometry["margin_px"]
        )

    except Exception as exc:
        error_rows.append({
            "manifest_row_1_based": (
                row_index
            ),
            "crystal_id": (
                legacy_crystal_id
            ),
            "sample_id": sample_id,
            "source_relative_path": (
                source_relative_path
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error": repr(exc),
        })

    if row_index % 1000 == 0:
        print(
            f"Processed {row_index:,}/"
            f"{len(source_rows):,}",
            flush=True,
        )

if not output_rows:
    raise RuntimeError(
        "No output rows were created."
    )

manifest_fields = list(
    output_rows[0].keys()
)

main_manifest = (
    MANIFEST_ROOT
    / "crystals_broad.csv"
)

error_manifest = (
    MANIFEST_ROOT
    / "export_errors.tsv"
)

write_table(
    main_manifest,
    output_rows,
    manifest_fields,
)

write_table(
    error_manifest,
    error_rows,
    [
        "manifest_row_1_based",
        "crystal_id",
        "sample_id",
        "source_relative_path",
        "error_type",
        "error",
    ],
    delimiter="\t",
)

rows_by_sample: defaultdict[
    str,
    list[dict[str, Any]],
] = defaultdict(list)

for row in output_rows:
    rows_by_sample[
        str(row["sample_id"])
    ].append(row)

per_sample_manifests = []

for sample_id in sorted(
    rows_by_sample
):
    sample_manifest = (
        PER_SAMPLE_ROOT
        / f"{sample_id}__retained.csv"
    )

    write_table(
        sample_manifest,
        rows_by_sample[sample_id],
        manifest_fields,
    )

    per_sample_manifests.append(
        sample_manifest
    )

    completion_path = (
        COMPLETION_ROOT
        / f"{sample_id}.complete.json"
    )

    write_json(
        completion_path,
        {
            "status": "PASS",
            "sample_id": sample_id,
            "physical_sample_id": (
                sample_id
            ),
            "acquisition_id": sample_id,
            "split_group_id": (
                "Aegina__" + sample_id
            ),
            "retained_images": len(
                rows_by_sample[sample_id]
            ),
            "manifest": str(
                sample_manifest
            ),
            "dataset_id": DATASET_ID,
            "dataset_version": (
                DATASET_VERSION
            ),
        },
    )

disk_output_paths = sorted(
    OUTPUT_IMAGE_ROOT.rglob(
        "*.png"
    )
)

unique_crystal_ids = len({
    row["crystal_id"]
    for row in output_rows
})

unique_output_paths = len({
    row["masked_relative_path"]
    for row in output_rows
})

# Audit source and output hashes separately so accidental duplicate creation
# during compatibility export is distinguishable from inherited duplicates.
source_duplicate_hash_groups = sum(
    count > 1
    for count in source_hash_counts.values()
)

output_duplicate_hash_groups = sum(
    count > 1
    for count in output_hash_counts.values()
)

checks = {
    "source_rows_expected": (
        len(source_rows)
        == EXPECTED_ROWS
    ),
    "output_rows_expected": (
        len(output_rows)
        == EXPECTED_ROWS
    ),
    "sample_count_expected": (
        len(sample_counts)
        == EXPECTED_SAMPLES
    ),
    "no_export_errors": (
        not error_rows
    ),
    "all_manifest_crystal_ids_unique": (
        unique_crystal_ids
        == len(output_rows)
    ),
    "all_output_paths_unique": (
        unique_output_paths
        == len(output_rows)
    ),
    "disk_png_count_matches_manifest": (
        len(disk_output_paths)
        == len(output_rows)
    ),
    "all_source_dimensions_match": (
        dimension_errors == 0
    ),
    "all_source_pixels_preserved": (
        source_pixel_embedding_errors
        == 0
    ),
    "all_outer_borders_black": (
        outer_border_errors == 0
    ),
    "all_outputs_valid": (
        output_validation_errors == 0
    ),
    "completion_marker_count_expected": (
        len(list(
            COMPLETION_ROOT.glob(
                "*.complete.json"
            )
        ))
        == EXPECTED_SAMPLES
    ),
}

status = (
    "PASS"
    if all(checks.values())
    else "FAIL"
)

summary = {
    "schema_version": SCHEMA_VERSION,
    "status": status,
    "dataset_id": DATASET_ID,
    "dataset_version": DATASET_VERSION,
    "source_manifest": str(
        SOURCE_MANIFEST
    ),
    "source_manifest_sha256": (
        sha256_file(
            SOURCE_MANIFEST
        )
    ),
    "source_image_root": str(
        SOURCE_IMAGE_ROOT
    ),
    "output_root": str(
        OUTPUT_ROOT
    ),
    "source_rows": len(
        source_rows
    ),
    "output_rows": len(
        output_rows
    ),
    "disk_png_count": len(
        disk_output_paths
    ),
    "sample_count": len(
        sample_counts
    ),
    "sample_counts": dict(
        sorted(sample_counts.items())
    ),
    "unique_crystal_ids": (
        unique_crystal_ids
    ),
    "unique_output_paths": (
        unique_output_paths
    ),
    "error_count": len(
        error_rows
    ),
    "dimension_errors": (
        dimension_errors
    ),
    "source_pixel_embedding_errors": (
        source_pixel_embedding_errors
    ),
    "outer_border_errors": (
        outer_border_errors
    ),
    "output_validation_errors": (
        output_validation_errors
    ),
    "source_duplicate_hash_groups": (
        source_duplicate_hash_groups
    ),
    "output_duplicate_hash_groups": (
        output_duplicate_hash_groups
    ),
    "margin_fraction": (
        MARGIN_FRACTION
    ),
    "minimum_margin_px": (
        MINIMUM_MARGIN_PX
    ),
    "margin_px": {
        "minimum": min(
            margin_values
        ),
        "median": float(
            statistics.median(
                margin_values
            )
        ),
        "maximum": max(
            margin_values
        ),
    },
    "output_side_px": {
        "minimum": min(
            output_side_values
        ),
        "median": float(
            statistics.median(
                output_side_values
            )
        ),
        "maximum": max(
            output_side_values
        ),
    },
    "transform": {
        "class": (
            "compatibility_transform"
        ),
        "name": TRANSFORM_NAME,
        "source_context": (
            SOURCE_CONTEXT
        ),
        "source_tile_context_available": (
            False
        ),
        "source_exact_mask_available": (
            False
        ),
        "source_pixels_preserved": (
            True
        ),
        "resize_applied": False,
        "background_rgb": [
            0,
            0,
            0,
        ],
    },
    "checks": checks,
    "outputs": {
        "manifest": str(
            main_manifest
        ),
        "errors": str(
            error_manifest
        ),
        "per_sample_manifests": str(
            PER_SAMPLE_ROOT
        ),
        "completion_markers": str(
            COMPLETION_ROOT
        ),
        "images": str(
            OUTPUT_IMAGE_ROOT
        ),
    },
    "interpretation": (
        "Compatibility transform only. "
        "Each legacy masked bbox PNG is "
        "preserved pixel-for-pixel, surrounded "
        "with black compatibility margin, and "
        "centre-padded to square without resize."
    ),
}

summary_path = (
    METADATA_ROOT
    / "aegina_ssl_compatibility_export_summary_v1.json"
)

write_json(
    summary_path,
    summary,
)

checksum_targets = [
    main_manifest,
    error_manifest,
    summary_path,
    *per_sample_manifests,
    *sorted(
        COMPLETION_ROOT.glob(
            "*.complete.json"
        )
    ),
]

checksum_lines = []

for path in sorted(
    checksum_targets
):
    relative_path = path.relative_to(
        OUTPUT_ROOT
    )

    checksum_lines.append(
        f"{sha256_file(path)}  "
        f"{relative_path.as_posix()}\n"
    )

checksum_path = (
    OUTPUT_ROOT
    / "SHA256SUMS"
)

atomic_write_text(
    checksum_path,
    "".join(checksum_lines),
)

print()
print("=" * 72)
print(json.dumps(
    summary,
    indent=2,
    sort_keys=True,
))
print("=" * 72)
print(
    "AEGINA FULL COMPATIBILITY EXPORT:",
    status,
)

if status != "PASS":
    raise SystemExit(1)
