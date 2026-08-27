from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import os
import argparse
import csv
import json
import shutil
import sys
import tarfile

import numpy as np
from PIL import Image

from _ssl_export_helpers import (
    build_member_indices,
    calculate_exact_geometry,
    expanded_crop_box,
    normalise_member_name,
    read_mask,
    resolve_mask_member,
    sanitise_path_component,
    save_png,
    sha256_file,
    square_pad,
)


IRP_ROOT = Path(
    os.environ.get(
        "PNP_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

SEG_ROOT = (
    IRP_ROOT
    / "runs/hpc/segmentation"
    / "plagnet_non_aegina_all_v1"
)

RAW_ROOT = (
    IRP_ROOT
    / "data/raw/non_aeg_raw_plag"
)

MANIFEST_ROOT = (
    IRP_ROOT
    / "data/manifests/plagnetplus"
)

CANDIDATE_MANIFEST = (
    MANIFEST_ROOT
    / "non_aegina_candidate_instances_v1.csv"
)

TILE_MANIFEST = (
    MANIFEST_ROOT
    / "non_aegina_tiles_v1.csv"
)

POLICY_PATH = (
    IRP_ROOT
    / "data/interim/plagnetplus"
    / "tile_overlap_all_v1"
    / "tile_ownership_policy_v1.csv"
)

ARCHIVE_ROOT = (
    SEG_ROOT
    / "noncore_archives"
)

DEFAULT_OUTPUT_ROOT = (
    IRP_ROOT
    / "data/processed/plagnetplus"
    / "ssl_non_aegina_v1"
)

EXPORTER_VERSION = (
    "plagnet_ssl_production_export_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--safe-id",
        action="append",
        help=(
            "Process only the named safe ID. "
            "May be supplied multiple times."
        ),
    )

    parser.add_argument(
        "--margin-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--minimum-margin-px",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--write-unmasked",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite-sample",
        action="store_true",
    )

    return parser.parse_args()


def read_csv_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        fields = reader.fieldnames or []
        rows = list(reader)

    return fields, rows


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {
        "true",
        "1",
        "yes",
    }


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(path)


def determine_ownership(
    *,
    geometry: dict[str, Any],
    image_width: int,
    image_height: int,
    grid_index_1: int,
    grid_index_2: int,
    sample_grid: set[tuple[int, int]],
    ownership_mode: str,
    half_overlap_x: float,
    half_overlap_y: float,
) -> dict[str, Any]:
    has_left = (
        grid_index_1,
        grid_index_2 - 1,
    ) in sample_grid

    has_right = (
        grid_index_1,
        grid_index_2 + 1,
    ) in sample_grid

    has_top = (
        grid_index_1 - 1,
        grid_index_2,
    ) in sample_grid

    has_bottom = (
        grid_index_1 + 1,
        grid_index_2,
    ) in sample_grid

    if ownership_mode == "no_overlap":
        left = 0.0
        right = float(image_width)
        top = 0.0
        bottom = float(image_height)
    elif ownership_mode == "overlap":
        left = (
            half_overlap_x
            if has_left
            else 0.0
        )

        right = (
            image_width - half_overlap_x
            if has_right
            else float(image_width)
        )

        top = (
            half_overlap_y
            if has_top
            else 0.0
        )

        bottom = (
            image_height - half_overlap_y
            if has_bottom
            else float(image_height)
        )
    else:
        raise RuntimeError(
            f"Unsupported ownership mode: "
            f"{ownership_mode!r}"
        )

    centroid_x = float(
        geometry["centroid_x_px"]
    )

    centroid_y = float(
        geometry["centroid_y_px"]
    )

    reasons = []

    if centroid_x < left:
        reasons.append(
            "owned_by_left_neighbour"
        )

    if centroid_x >= right:
        reasons.append(
            "owned_by_right_neighbour"
        )

    if centroid_y < top:
        reasons.append(
            "owned_by_top_neighbour"
        )

    if centroid_y >= bottom:
        reasons.append(
            "owned_by_bottom_neighbour"
        )

    retained = not reasons

    return {
        "has_left_neighbour": has_left,
        "has_right_neighbour": has_right,
        "has_top_neighbour": has_top,
        "has_bottom_neighbour": has_bottom,
        "ownership_left_px": left,
        "ownership_right_px": right,
        "ownership_top_px": top,
        "ownership_bottom_px": bottom,
        "retained_by_ownership": retained,
        "ownership_exclusion_reason": (
            ""
            if retained
            else ";".join(reasons)
        ),
    }


def clean_sample_outputs(
    *,
    output_root: Path,
    locality: str,
    sample_id: str,
    safe_id: str,
) -> None:
    locality_component = (
        sanitise_path_component(locality)
    )

    sample_component = (
        sanitise_path_component(sample_id)
    )

    for image_group in (
        "images_masked",
        "images_unmasked",
    ):
        target = (
            output_root
            / image_group
            / locality_component
            / sample_component
        )

        if target.exists():
            shutil.rmtree(target)

    per_sample_root = (
        output_root / "manifests/per_sample"
    )

    provenance_root = (
        output_root / "provenance/per_sample"
    )

    for path in (
        per_sample_root
        / f"{safe_id}__retained.csv",
        per_sample_root
        / f"{safe_id}__excluded.csv",
        provenance_root
        / f"{safe_id}__summary.json",
        provenance_root
        / f"{safe_id}__errors.tsv",
        provenance_root
        / f"{safe_id}__EXPORT_COMPLETE",
    ):
        path.unlink(missing_ok=True)


def process_sample(
    *,
    safe_id: str,
    rows: list[dict[str, str]],
    input_fields: list[str],
    policy: dict[str, str],
    sample_grid: set[tuple[int, int]],
    output_root: Path,
    margin_fraction: float,
    minimum_margin_px: int,
    write_unmasked: bool,
    overwrite_sample: bool,
) -> dict[str, Any]:
    locality = rows[0]["locality"]
    sample_id = rows[0]["sample_id"]

    per_sample_root = (
        output_root / "manifests/per_sample"
    )

    provenance_root = (
        output_root / "provenance/per_sample"
    )

    per_sample_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    provenance_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker_path = (
        provenance_root
        / f"{safe_id}__EXPORT_COMPLETE"
    )

    summary_path = (
        provenance_root
        / f"{safe_id}__summary.json"
    )

    if marker_path.is_file() and not overwrite_sample:
        print(
            f"Skipping completed sample: {safe_id}"
        )

        return json.loads(
            summary_path.read_text()
        )

    if overwrite_sample:
        clean_sample_outputs(
            output_root=output_root,
            locality=locality,
            sample_id=sample_id,
            safe_id=safe_id,
        )

    marker_path.unlink(missing_ok=True)

    retained_path = (
        per_sample_root
        / f"{safe_id}__retained.csv"
    )

    excluded_path = (
        per_sample_root
        / f"{safe_id}__excluded.csv"
    )

    retained_partial = (
        retained_path.with_suffix(
            ".csv.partial"
        )
    )

    excluded_partial = (
        excluded_path.with_suffix(
            ".csv.partial"
        )
    )

    error_path = (
        provenance_root
        / f"{safe_id}__errors.tsv"
    )

    appended_fields = [
        "exact_mask_archive",
        "exact_mask_archive_member",
        "exact_mask_mode",
        "exact_mask_area_px",
        "exact_bbox_x_min",
        "exact_bbox_y_min",
        "exact_bbox_x_max_exclusive",
        "exact_bbox_y_max_exclusive",
        "exact_bbox_width_px",
        "exact_bbox_height_px",
        "exact_mask_bbox_occupancy",
        "exact_centroid_x_px",
        "exact_centroid_y_px",
        "exact_touches_tile_border",
        "exact_border_distance_px",
        "source_image_width_px",
        "source_image_height_px",
        "ownership_policy_version",
        "ownership_mode",
        "ownership_overlap_x_px",
        "ownership_overlap_y_px",
        "has_left_neighbour",
        "has_right_neighbour",
        "has_top_neighbour",
        "has_bottom_neighbour",
        "ownership_left_px",
        "ownership_right_px",
        "ownership_top_px",
        "ownership_bottom_px",
        "retained_by_ownership",
        "ownership_exclusion_reason",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "crop_margin_px",
        "padding_x_px",
        "padding_y_px",
        "output_width_px",
        "output_height_px",
        "masked_relative_path",
        "unmasked_relative_path",
        "masked_sha256",
        "unmasked_sha256",
        "valid_broad",
        "valid_strict",
        "strict_exclusion_reason",
        "exporter_version",
    ]

    output_fields = (
        input_fields + appended_fields
    )

    archive_path = (
        ARCHIVE_ROOT
        / f"{safe_id}.noncore.tar"
    )

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    ownership_mode = policy[
        "ownership_mode"
    ]

    overlap_x = float(
        policy["overlap_x_px"] or 0
    )

    overlap_y = float(
        policy["overlap_y_px"] or 0
    )

    half_overlap_x = overlap_x / 2.0
    half_overlap_y = overlap_y / 2.0

    masked_root = (
        output_root / "images_masked"
    )

    unmasked_root = (
        output_root / "images_unmasked"
    )

    locality_component = (
        sanitise_path_component(locality)
    )

    sample_component = (
        sanitise_path_component(sample_id)
    )

    rows = sorted(
        rows,
        key=lambda row: (
            row["source_relative_path"],
            int(row["instance_label"]),
        ),
    )

    retained_count = 0
    excluded_count = 0
    area_mismatches = 0
    errors = []
    reason_counts: Counter[str] = Counter()

    current_source_relative = None
    current_source_image = None
    current_source_array = None

    print()
    print("=" * 72)
    print("Exporting:", safe_id)
    print("Candidates:", len(rows))
    print("Ownership mode:", ownership_mode)
    print("Overlap X:", overlap_x)
    print("Overlap Y:", overlap_y)

    with (
        retained_partial.open(
            "w",
            newline="",
        ) as retained_handle,
        excluded_partial.open(
            "w",
            newline="",
        ) as excluded_handle,
        tarfile.open(
            archive_path,
            "r",
        ) as archive,
    ):
        retained_writer = csv.DictWriter(
            retained_handle,
            fieldnames=output_fields,
            lineterminator="\n",
        )

        excluded_writer = csv.DictWriter(
            excluded_handle,
            fieldnames=output_fields,
            lineterminator="\n",
        )

        retained_writer.writeheader()
        excluded_writer.writeheader()

        (
            members,
            members_by_basename,
        ) = build_member_indices(archive)

        member_names = set(members)

        for index, row in enumerate(
            rows,
            start=1,
        ):
            try:
                source_csv_row = json.loads(
                    row["source_csv_row_json"]
                )

                mask_path = str(
                    source_csv_row["mask_path"]
                )

                tile_stem = Path(
                    row["source_filename"]
                ).stem

                member_name = (
                    resolve_mask_member(
                        member_names=member_names,
                        members_by_basename=(
                            members_by_basename
                        ),
                        mask_path=mask_path,
                        tile_stem=tile_stem,
                    )
                )

                mask, mask_mode = read_mask(
                    archive=archive,
                    member=members[member_name],
                )

                # Recalculate geometry from the exact retained mask before applying crop and
                # quality rules; detector boxes alone are not used for final occupancy.
                geometry = (
                    calculate_exact_geometry(mask)
                )

                csv_area = int(
                    row["csv_mask_area_px"]
                )

                if (
                    int(geometry["area_px"])
                    != csv_area
                ):
                    area_mismatches += 1

                    raise RuntimeError(
                        "Exact mask area mismatch: "
                        f"csv={csv_area}, "
                        f"exact={geometry['area_px']}"
                    )

                image_height, image_width = (
                    mask.shape
                )

                grid_index_1 = int(
                    row["grid_index_1"]
                )

                grid_index_2 = int(
                    row["grid_index_2"]
                )

                # Apply the tile-ownership rule before exporting a crystal so detections in
                # overlapping source tiles do not become duplicate training images.
                ownership = determine_ownership(
                    geometry=geometry,
                    image_width=image_width,
                    image_height=image_height,
                    grid_index_1=grid_index_1,
                    grid_index_2=grid_index_2,
                    sample_grid=sample_grid,
                    ownership_mode=ownership_mode,
                    half_overlap_x=half_overlap_x,
                    half_overlap_y=half_overlap_y,
                )

                bbox_area = (
                    int(
                        geometry["width_px"]
                    )
                    * int(
                        geometry["height_px"]
                    )
                )

                common = {
                    **row,
                    "exact_mask_archive": (
                        archive_path.name
                    ),
                    "exact_mask_archive_member": (
                        normalise_member_name(
                            member_name
                        )
                    ),
                    "exact_mask_mode": (
                        mask_mode
                    ),
                    "exact_mask_area_px": (
                        geometry["area_px"]
                    ),
                    "exact_bbox_x_min": (
                        geometry["x_min"]
                    ),
                    "exact_bbox_y_min": (
                        geometry["y_min"]
                    ),
                    "exact_bbox_x_max_exclusive": (
                        geometry[
                            "x_max_exclusive"
                        ]
                    ),
                    "exact_bbox_y_max_exclusive": (
                        geometry[
                            "y_max_exclusive"
                        ]
                    ),
                    "exact_bbox_width_px": (
                        geometry["width_px"]
                    ),
                    "exact_bbox_height_px": (
                        geometry["height_px"]
                    ),
                    "exact_mask_bbox_occupancy": (
                        geometry["area_px"]
                        / bbox_area
                    ),
                    "exact_centroid_x_px": (
                        geometry[
                            "centroid_x_px"
                        ]
                    ),
                    "exact_centroid_y_px": (
                        geometry[
                            "centroid_y_px"
                        ]
                    ),
                    "exact_touches_tile_border": (
                        geometry[
                            "touches_tile_border"
                        ]
                    ),
                    "exact_border_distance_px": (
                        geometry[
                            "border_distance_px"
                        ]
                    ),
                    "source_image_width_px": (
                        image_width
                    ),
                    "source_image_height_px": (
                        image_height
                    ),
                    "ownership_policy_version": (
                        "tile_ownership_policy_v1"
                    ),
                    "ownership_mode": (
                        ownership_mode
                    ),
                    "ownership_overlap_x_px": (
                        overlap_x
                    ),
                    "ownership_overlap_y_px": (
                        overlap_y
                    ),
                    **ownership,
                    "crop_x1": "",
                    "crop_y1": "",
                    "crop_x2": "",
                    "crop_y2": "",
                    "crop_margin_px": "",
                    "padding_x_px": "",
                    "padding_y_px": "",
                    "output_width_px": "",
                    "output_height_px": "",
                    "masked_relative_path": "",
                    "unmasked_relative_path": "",
                    "masked_sha256": "",
                    "unmasked_sha256": "",
                    "valid_broad": (
                        ownership[
                            "retained_by_ownership"
                        ]
                    ),
                    "valid_strict": "",
                    "strict_exclusion_reason": (
                        "PENDING_STRICT_POLICY"
                        if ownership[
                            "retained_by_ownership"
                        ]
                        else ""
                    ),
                    "exporter_version": (
                        EXPORTER_VERSION
                    ),
                }

                if not ownership[
                    "retained_by_ownership"
                ]:
                    reason = ownership[
                        "ownership_exclusion_reason"
                    ]

                    for item in reason.split(";"):
                        reason_counts[item] += 1

                    excluded_writer.writerow(
                        common
                    )

                    excluded_count += 1
                    continue

                source_relative = (
                    row["source_relative_path"]
                )

                if (
                    source_relative
                    != current_source_relative
                ):
                    source_path = (
                        RAW_ROOT
                        / source_relative
                    )

                    with Image.open(
                        source_path
                    ) as image:
                        current_source_image = (
                            image.convert("RGB").copy()
                        )

                    current_source_array = (
                        np.asarray(
                            current_source_image
                        )
                    )

                    current_source_relative = (
                        source_relative
                    )

                assert (
                    current_source_image
                    is not None
                )

                assert (
                    current_source_array
                    is not None
                )

                if (
                    current_source_image.size
                    != (
                        image_width,
                        image_height,
                    )
                ):
                    raise RuntimeError(
                        "Source/mask shape mismatch: "
                        f"source="
                        f"{current_source_image.size}, "
                        f"mask="
                        f"{(image_width, image_height)}"
                    )

                (
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2,
                    margin_px,
                ) = expanded_crop_box(
                    geometry=geometry,
                    image_width=image_width,
                    image_height=image_height,
                    margin_fraction=(
                        margin_fraction
                    ),
                    minimum_margin_px=(
                        minimum_margin_px
                    ),
                )

                image_crop = (
                    current_source_array[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2,
                    ].copy()
                )

                mask_crop = mask[
                    crop_y1:crop_y2,
                    crop_x1:crop_x2,
                ]

                masked_crop = image_crop.copy()
                masked_crop[~mask_crop] = 0

                masked_image = (
                    Image.fromarray(
                        masked_crop,
                        mode="RGB",
                    )
                )

                (
                    masked_square,
                    padding_x,
                    padding_y,
                ) = square_pad(
                    masked_image,
                    fill=(0, 0, 0),
                )

                filename = (
                    sanitise_path_component(
                        row["candidate_id"]
                    )
                    + ".png"
                )

                masked_path = (
                    masked_root
                    / locality_component
                    / sample_component
                    / filename
                )

                save_png(
                    masked_square,
                    masked_path,
                )

                masked_hash = sha256_file(
                    masked_path
                )

                unmasked_relative = ""
                unmasked_hash = ""

                if write_unmasked:
                    unmasked_image = (
                        Image.fromarray(
                            image_crop,
                            mode="RGB",
                        )
                    )

                    (
                        unmasked_square,
                        unmasked_padding_x,
                        unmasked_padding_y,
                    ) = square_pad(
                        unmasked_image,
                        fill=(0, 0, 0),
                    )

                    if (
                        padding_x
                        != unmasked_padding_x
                        or padding_y
                        != unmasked_padding_y
                    ):
                        raise RuntimeError(
                            "Masked/unmasked padding "
                            "mismatch"
                        )

                    unmasked_path = (
                        unmasked_root
                        / locality_component
                        / sample_component
                        / filename
                    )

                    save_png(
                        unmasked_square,
                        unmasked_path,
                    )

                    unmasked_relative = (
                        unmasked_path.relative_to(
                            output_root
                        ).as_posix()
                    )

                    unmasked_hash = (
                        sha256_file(
                            unmasked_path
                        )
                    )

                common.update({
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                    "crop_margin_px": (
                        margin_px
                    ),
                    "padding_x_px": (
                        padding_x
                    ),
                    "padding_y_px": (
                        padding_y
                    ),
                    "output_width_px": (
                        masked_square.width
                    ),
                    "output_height_px": (
                        masked_square.height
                    ),
                    "masked_relative_path": (
                        masked_path.relative_to(
                            output_root
                        ).as_posix()
                    ),
                    "unmasked_relative_path": (
                        unmasked_relative
                    ),
                    "masked_sha256": (
                        masked_hash
                    ),
                    "unmasked_sha256": (
                        unmasked_hash
                    ),
                })

                retained_writer.writerow(common)
                retained_count += 1

            except Exception as exc:
                errors.append({
                    "candidate_id": (
                        row["candidate_id"]
                    ),
                    "source_filename": (
                        row["source_filename"]
                    ),
                    "instance_label": (
                        row["instance_label"]
                    ),
                    "error": repr(exc),
                })

            if index % 500 == 0:
                print(
                    f"  {index}/{len(rows)} "
                    f"candidates; "
                    f"retained={retained_count}; "
                    f"excluded={excluded_count}; "
                    f"errors={len(errors)}",
                    flush=True,
                )

    if errors:
        with error_path.open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "candidate_id",
                    "source_filename",
                    "instance_label",
                    "error",
                ],
                delimiter="\t",
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(errors)

        raise RuntimeError(
            f"{safe_id}: "
            f"{len(errors)} export errors; "
            f"see {error_path}"
        )

    retained_partial.replace(
        retained_path
    )

    excluded_partial.replace(
        excluded_path
    )

    summary = {
        "schema_version": (
            EXPORTER_VERSION
        ),
        "safe_id": safe_id,
        "locality": locality,
        "sample_id": sample_id,
        "candidate_count": len(rows),
        "retained_count": retained_count,
        "ownership_excluded_count": (
            excluded_count
        ),
        "retained_fraction": (
            retained_count / len(rows)
        ),
        "ownership_mode": (
            ownership_mode
        ),
        "overlap_x_px": overlap_x,
        "overlap_y_px": overlap_y,
        "ownership_exclusion_reasons": (
            dict(reason_counts)
        ),
        "area_mismatches": (
            area_mismatches
        ),
        "errors": 0,
        "write_unmasked": (
            write_unmasked
        ),
        "margin_fraction": (
            margin_fraction
        ),
        "minimum_margin_px": (
            minimum_margin_px
        ),
        "retained_manifest": str(
            retained_path
        ),
        "excluded_manifest": str(
            excluded_path
        ),
    }

    write_json(
        summary_path,
        summary,
    )

    marker_path.write_text(
        json.dumps(
            {
                "safe_id": safe_id,
                "retained_manifest_sha256": (
                    sha256_file(
                        retained_path
                    )
                ),
                "excluded_manifest_sha256": (
                    sha256_file(
                        excluded_path
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("Sample:", safe_id)
    print("Candidates:", len(rows))
    print("Retained:", retained_count)
    print(
        "Ownership excluded:",
        excluded_count,
    )
    print("Errors: 0")
    print("SAMPLE SSL EXPORT: PASS")

    return summary


def merge_csv_files(
    *,
    paths: list[Path],
    output_path: Path,
) -> int:
    if not paths:
        return 0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_suffix(
        output_path.suffix + ".partial"
    )

    total_rows = 0
    output_fields = None

    with temporary.open(
        "w",
        newline="",
    ) as output_handle:
        writer = None

        for path in sorted(paths):
            with path.open(
                newline=""
            ) as input_handle:
                reader = csv.DictReader(
                    input_handle
                )

                fields = reader.fieldnames or []

                if output_fields is None:
                    output_fields = fields

                    writer = csv.DictWriter(
                        output_handle,
                        fieldnames=(
                            output_fields
                        ),
                        lineterminator="\n",
                    )

                    writer.writeheader()

                elif fields != output_fields:
                    raise RuntimeError(
                        "Per-sample manifest "
                        f"schema mismatch: {path}"
                    )

                assert writer is not None

                for row in reader:
                    writer.writerow(row)
                    total_rows += 1

    temporary.replace(output_path)

    return total_rows


def rebuild_merged_outputs(
    output_root: Path,
) -> dict[str, Any]:
    per_sample_root = (
        output_root / "manifests/per_sample"
    )

    provenance_root = (
        output_root / "provenance/per_sample"
    )

    complete_ids = sorted(
        path.name.removesuffix(
            "__EXPORT_COMPLETE"
        )
        for path in provenance_root.glob(
            "*__EXPORT_COMPLETE"
        )
    )

    retained_paths = [
        per_sample_root
        / f"{safe_id}__retained.csv"
        for safe_id in complete_ids
    ]

    excluded_paths = [
        per_sample_root
        / f"{safe_id}__excluded.csv"
        for safe_id in complete_ids
    ]

    broad_path = (
        output_root
        / "manifests/crystals_broad.csv"
    )

    excluded_path = (
        output_root
        / "manifests/ownership_excluded.csv"
    )

    retained_rows = merge_csv_files(
        paths=retained_paths,
        output_path=broad_path,
    )

    excluded_rows = merge_csv_files(
        paths=excluded_paths,
        output_path=excluded_path,
    )

    sample_summaries = []

    for safe_id in complete_ids:
        path = (
            provenance_root
            / f"{safe_id}__summary.json"
        )

        sample_summaries.append(
            json.loads(path.read_text())
        )

    locality_counts: dict[
        str,
        dict[str, int],
    ] = defaultdict(
        lambda: {
            "samples": 0,
            "candidates": 0,
            "retained": 0,
            "excluded": 0,
        }
    )

    for summary in sample_summaries:
        record = locality_counts[
            summary["locality"]
        ]

        record["samples"] += 1
        record["candidates"] += int(
            summary["candidate_count"]
        )
        record["retained"] += int(
            summary["retained_count"]
        )
        record["excluded"] += int(
            summary[
                "ownership_excluded_count"
            ]
        )

    merged_summary = {
        "schema_version": (
            EXPORTER_VERSION
        ),
        "completed_samples": len(
            complete_ids
        ),
        "completed_safe_ids": (
            complete_ids
        ),
        "candidate_rows": (
            retained_rows
            + excluded_rows
        ),
        "broad_crystal_rows": (
            retained_rows
        ),
        "ownership_excluded_rows": (
            excluded_rows
        ),
        "locality_counts": dict(
            locality_counts
        ),
        "broad_manifest": str(
            broad_path
        ),
        "ownership_excluded_manifest": (
            str(excluded_path)
        ),
    }

    write_json(
        output_root
        / "provenance"
        / "dataset_build_summary.json",
        merged_summary,
    )

    if broad_path.is_file():
        (
            broad_path.with_suffix(
                ".csv.sha256"
            )
        ).write_text(
            f"{sha256_file(broad_path)}  "
            f"{broad_path.name}\n"
        )

    if excluded_path.is_file():
        (
            excluded_path.with_suffix(
                ".csv.sha256"
            )
        ).write_text(
            f"{sha256_file(excluded_path)}  "
            f"{excluded_path.name}\n"
        )

    return merged_summary


def main() -> None:
    args = parse_args()

    output_root = (
        args.output_root.resolve()
    )

    for directory in (
        output_root / "images_masked",
        output_root / "images_unmasked",
        output_root / "manifests/per_sample",
        output_root / "provenance/per_sample",
        output_root / "qc",
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    input_fields, candidate_rows = (
        read_csv_rows(
            CANDIDATE_MANIFEST
        )
    )

    _, tile_rows = read_csv_rows(
        TILE_MANIFEST
    )

    _, policy_rows = read_csv_rows(
        POLICY_PATH
    )

    policy_by_safe_id = {
        row["safe_id"]: row
        for row in policy_rows
    }

    invalid_policy = [
        row
        for row in policy_rows
        if row["status"] != "PASS"
    ]

    if invalid_policy:
        raise RuntimeError(
            "Ownership policy contains "
            "non-PASS samples"
        )

    candidates_by_safe_id: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in candidate_rows:
        candidates_by_safe_id[
            row["safe_id"]
        ].append(row)

    grid_by_safe_id: dict[
        str,
        set[tuple[int, int]],
    ] = defaultdict(set)

    for row in tile_rows:
        grid_by_safe_id[
            row["safe_id"]
        ].add((
            int(row["grid_index_1"]),
            int(row["grid_index_2"]),
        ))

    requested_ids = (
        args.safe_id
        if args.safe_id
        else sorted(
            candidates_by_safe_id
        )
    )

    unknown = (
        set(requested_ids)
        - set(candidates_by_safe_id)
    )

    if unknown:
        raise RuntimeError(
            f"Unknown safe IDs: "
            f"{sorted(unknown)}"
        )

    for safe_id in requested_ids:
        if safe_id not in policy_by_safe_id:
            raise RuntimeError(
                f"Missing policy for {safe_id}"
            )

        process_sample(
            safe_id=safe_id,
            rows=candidates_by_safe_id[
                safe_id
            ],
            input_fields=input_fields,
            policy=policy_by_safe_id[
                safe_id
            ],
            sample_grid=grid_by_safe_id[
                safe_id
            ],
            output_root=output_root,
            margin_fraction=(
                args.margin_fraction
            ),
            minimum_margin_px=(
                args.minimum_margin_px
            ),
            write_unmasked=(
                args.write_unmasked
            ),
            overwrite_sample=(
                args.overwrite_sample
            ),
        )

    # Rebuild merged manifests from per-sample outputs at the end so interrupted
    # runs can be resumed without trusting stale aggregate files.
    merged = rebuild_merged_outputs(
        output_root
    )

    print()
    print("=" * 72)
    print("Completed samples:", merged[
        "completed_samples"
    ])
    print(
        "Candidate rows:",
        merged["candidate_rows"],
    )
    print(
        "Broad crystal rows:",
        merged["broad_crystal_rows"],
    )
    print(
        "Ownership excluded rows:",
        merged[
            "ownership_excluded_rows"
        ],
    )
    print(
        "Merged summary:",
        output_root
        / "provenance"
        / "dataset_build_summary.json",
    )
    print()
    print(
        "PLAGNET SSL PRODUCTION EXPORT: PASS"
    )


if __name__ == "__main__":
    main()
