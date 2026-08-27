from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import os
import csv
import hashlib
import json
import re
import sys

import numpy as np
from PIL import Image


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

LABEL_ROOT = (
    IRP_ROOT
    / "data/interim/plagnetplus"
    / "non_aegina_clean_labels_v1"
)

MANIFEST_ROOT = (
    IRP_ROOT
    / "data/manifests/plagnetplus"
)

TASKS_PATH = SEG_ROOT / "tasks.tsv"
SEG_OUTPUT_ROOT = SEG_ROOT / "output"
SEG_MANIFEST_ROOT = SEG_ROOT / "manifests"

TILE_MANIFEST = (
    MANIFEST_ROOT
    / "non_aegina_tiles_v1.csv"
)

CANDIDATE_MANIFEST = (
    MANIFEST_ROOT
    / "non_aegina_candidate_instances_v1.csv"
)

SAMPLE_SUMMARY = (
    MANIFEST_ROOT
    / "non_aegina_candidate_counts_by_sample_v1.csv"
)

LOCALITY_SUMMARY = (
    MANIFEST_ROOT
    / "non_aegina_candidate_counts_by_locality_v1.csv"
)

SUMMARY_JSON = (
    MANIFEST_ROOT
    / "non_aegina_candidate_manifest_summary_v1.json"
)

ERRORS_TSV = (
    MANIFEST_ROOT
    / "non_aegina_candidate_manifest_errors_v1.tsv"
)

GRID_PATTERN = re.compile(
    r"_(\d{5})_(\d{5})$"
)

LABEL_ALIASES = (
    "instance_label",
    "label_id",
    "instance_id",
    "clean_instance_id",
    "clean_label",
    "clean_id",
    "label",
)

DETECTION_ALIASES = (
    "detection_index",
    "source_detection_index",
    "detection_id",
)

SCORE_ALIASES = (
    "segmentation_score",
    "score",
    "confidence",
    "confidence_score",
)

CLASS_ALIASES = (
    "class_id",
    "category_id",
    "label_class",
)

CSV_AREA_ALIASES = (
    "area_px",
    "mask_area_px",
    "area",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalise_field(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.strip().lower(),
    ).strip("_")


def select_field(
    fields: list[str],
    aliases: tuple[str, ...],
) -> str | None:
    normalised = {
        normalise_field(field): field
        for field in fields
    }

    for alias in aliases:
        if alias in normalised:
            return normalised[alias]

    return None


def parse_optional_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return int(float(text))


def parse_optional_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return float(text)


def build_artifact_index(
    root: Path,
    filename: str,
) -> tuple[
    dict[tuple[str, str], Path],
    list[dict[str, str]],
]:
    index: dict[tuple[str, str], Path] = {}
    duplicates: list[dict[str, str]] = []

    for safe_dir in sorted(root.iterdir()):
        if not safe_dir.is_dir():
            continue

        safe_id = safe_dir.name

        for path in sorted(
            safe_dir.rglob(filename)
        ):
            tile_stem = path.parent.name
            key = (safe_id, tile_stem)

            if key in index:
                duplicates.append({
                    "safe_id": safe_id,
                    "tile_stem": tile_stem,
                    "first_path": str(index[key]),
                    "second_path": str(path),
                })
            else:
                index[key] = path

    return index, duplicates


def output_relative(path: Path) -> str:
    return path.relative_to(
        SEG_ROOT
    ).as_posix()


def label_relative(path: Path) -> str:
    return path.relative_to(
        LABEL_ROOT
    ).as_posix()


def raw_relative(path: Path) -> str:
    return path.relative_to(
        RAW_ROOT
    ).as_posix()


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".partial"
    )

    with temporary.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def make_error(
    errors: list[dict[str, str]],
    *,
    safe_id: str,
    tile_stem: str,
    category: str,
    detail: str,
) -> None:
    errors.append({
        "safe_id": safe_id,
        "tile_stem": tile_stem,
        "category": category,
        "detail": detail,
    })


MANIFEST_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

with TASKS_PATH.open(newline="") as handle:
    tasks = list(
        csv.DictReader(
            handle,
            delimiter="\t",
        )
    )

if len(tasks) != 20:
    raise RuntimeError(
        f"Expected 20 task rows, found {len(tasks)}"
    )

# Index segmentation artifacts once so candidate rows can resolve reports,
# masks, and source scans consistently across all production samples.
report_index, duplicate_reports = (
    build_artifact_index(
        SEG_OUTPUT_ROOT,
        "report.json",
    )
)

csv_index, duplicate_csvs = (
    build_artifact_index(
        SEG_OUTPUT_ROOT,
        "clean_instances.csv",
    )
)

label_index, duplicate_labels = (
    build_artifact_index(
        LABEL_ROOT,
        "clean_instance_labels_uint16.png",
    )
)

errors: list[dict[str, str]] = []

for item in duplicate_reports:
    make_error(
        errors,
        safe_id=item["safe_id"],
        tile_stem=item["tile_stem"],
        category="duplicate_report",
        detail=json.dumps(
            item,
            sort_keys=True,
        ),
    )

for item in duplicate_csvs:
    make_error(
        errors,
        safe_id=item["safe_id"],
        tile_stem=item["tile_stem"],
        category="duplicate_clean_csv",
        detail=json.dumps(
            item,
            sort_keys=True,
        ),
    )

for item in duplicate_labels:
    make_error(
        errors,
        safe_id=item["safe_id"],
        tile_stem=item["tile_stem"],
        category="duplicate_label_map",
        detail=json.dumps(
            item,
            sort_keys=True,
        ),
    )

tile_fields = [
    "tile_id",
    "task_index",
    "locality",
    "locality_display",
    "sample_id",
    "safe_id",
    "source_relative_path",
    "source_filename",
    "source_size_bytes",
    "source_sha256",
    "grid_index_1",
    "grid_index_2",
    "image_width_px",
    "image_height_px",
    "image_mode",
    "label_map_mode",
    "label_map_dtype",
    "label_map_max_label",
    "clean_instance_count",
    "foreground_area_px",
    "foreground_fraction",
    "zero_instance_tile",
    "csv_label_mapping",
    "csv_label_field",
    "csv_score_field",
    "csv_detection_field",
    "csv_class_field",
    "report_relative_path",
    "clean_instances_relative_path",
    "label_map_relative_path",
    "report_sha256",
    "clean_instances_sha256",
    "label_map_sha256",
    "validation_status",
]

candidate_fields = [
    "candidate_id",
    "tile_id",
    "task_index",
    "locality",
    "locality_display",
    "sample_id",
    "safe_id",
    "source_relative_path",
    "source_filename",
    "source_sha256",
    "grid_index_1",
    "grid_index_2",
    "instance_label",
    "source_csv_mapping",
    "detection_index",
    "segmentation_score",
    "class_id",
    "mask_area_px",
    "csv_mask_area_px",
    "label_overwritten_area_px",
    "label_visible_fraction_of_csv_mask",
    "mask_geometry_source",
    "mask_area_fraction",
    "bbox_x_min",
    "bbox_y_min",
    "bbox_x_max_exclusive",
    "bbox_y_max_exclusive",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px",
    "mask_bbox_occupancy",
    "centroid_x_px",
    "centroid_y_px",
    "touches_tile_border",
    "border_distance_px",
    "label_map_relative_path",
    "clean_instances_relative_path",
    "report_relative_path",
    "source_csv_row_json",
]

candidate_partial = CANDIDATE_MANIFEST.with_suffix(
    CANDIDATE_MANIFEST.suffix + ".partial"
)

tile_rows: list[dict[str, Any]] = []

candidate_ids: set[str] = set()
tile_ids: set[str] = set()

sample_stats: dict[
    tuple[str, str, str],
    dict[str, int],
] = defaultdict(
    lambda: {
        "tiles": 0,
        "zero_instance_tiles": 0,
        "candidate_instances": 0,
        "foreground_area_px": 0,
    }
)

locality_stats: dict[
    tuple[str, str],
    dict[str, int],
] = defaultdict(
    lambda: {
        "samples": 0,
        "tiles": 0,
        "zero_instance_tiles": 0,
        "candidate_instances": 0,
        "foreground_area_px": 0,
    }
)

csv_schema_counter: Counter[
    tuple[str, ...]
] = Counter()

mapping_counter: Counter[str] = Counter()

expected_tile_rows = 0
processed_tile_rows = 0
candidate_count = 0

with candidate_partial.open(
    "w",
    newline="",
) as candidate_handle:
    candidate_writer = csv.DictWriter(
        candidate_handle,
        fieldnames=candidate_fields,
        lineterminator="\n",
    )
    candidate_writer.writeheader()

    for task_number, task in enumerate(
        tasks,
        start=1,
    ):
        safe_id = task["safe_id"]
        task_index = int(task["task_index"])
        locality = task["locality"]
        locality_display = (
            "Romania"
            if locality == "Rumania"
            else locality
        )
        sample_id = task["sample"]

        manifest_path = (
            SEG_MANIFEST_ROOT
            / f"{safe_id}.csv"
        )

        if not manifest_path.is_file():
            make_error(
                errors,
                safe_id=safe_id,
                tile_stem="",
                category="missing_sample_manifest",
                detail=str(manifest_path),
            )
            continue

        with manifest_path.open(
            newline=""
        ) as handle:
            manifest_rows = list(
                csv.DictReader(handle)
            )

        expected_from_task = int(
            task["tile_count"]
        )

        if len(manifest_rows) != expected_from_task:
            make_error(
                errors,
                safe_id=safe_id,
                tile_stem="",
                category="sample_manifest_count",
                detail=(
                    f"manifest={len(manifest_rows)} "
                    f"tasks.tsv={expected_from_task}"
                ),
            )

        expected_tile_rows += len(manifest_rows)

        for manifest_row in manifest_rows:
            source_relative_path = (
                manifest_row["input_relative_path"]
            )
            source_path = (
                RAW_ROOT
                / source_relative_path
            )
            source_filename = (
                manifest_row["filename"]
            )
            tile_stem = Path(
                source_filename
            ).stem

            key = (safe_id, tile_stem)

            tile_id = (
                f"{safe_id}__{tile_stem}"
            )

            if tile_id in tile_ids:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="duplicate_tile_id",
                    detail=tile_id,
                )
                continue

            tile_ids.add(tile_id)

            report_path = report_index.get(key)
            csv_path = csv_index.get(key)
            label_path = label_index.get(key)

            missing_components = []

            if not source_path.is_file():
                missing_components.append(
                    f"source:{source_path}"
                )

            if report_path is None:
                missing_components.append(
                    "report.json"
                )

            if csv_path is None:
                missing_components.append(
                    "clean_instances.csv"
                )

            if label_path is None:
                missing_components.append(
                    "clean_instance_labels_uint16.png"
                )

            if missing_components:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="missing_component",
                    detail="; ".join(
                        missing_components
                    ),
                )
                continue

            assert report_path is not None
            assert csv_path is not None
            assert label_path is not None

            try:
                report = json.loads(
                    report_path.read_text()
                )
            except Exception as exc:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="invalid_report_json",
                    detail=repr(exc),
                )
                continue

            try:
                with Image.open(
                    source_path
                ) as source_image:
                    image_width, image_height = (
                        source_image.size
                    )
                    image_mode = (
                        source_image.mode
                    )
                    source_image.verify()
            except Exception as exc:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="invalid_source_image",
                    detail=repr(exc),
                )
                continue

            try:
                with Image.open(
                    label_path
                ) as label_image:
                    label_mode = (
                        label_image.mode
                    )
                    label_array = np.asarray(
                        label_image
                    )
            except Exception as exc:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="invalid_label_map",
                    detail=repr(exc),
                )
                continue

            if label_array.ndim != 2:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="label_map_dimensions",
                    detail=(
                        f"shape={label_array.shape}"
                    ),
                )
                continue

            if not np.issubdtype(
                label_array.dtype,
                np.integer,
            ):
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="label_map_dtype",
                    detail=str(label_array.dtype),
                )
                continue

            if label_array.shape != (
                image_height,
                image_width,
            ):
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="image_label_shape_mismatch",
                    detail=(
                        f"source={(image_height, image_width)} "
                        f"label={label_array.shape}"
                    ),
                )
                continue

            flat_labels = label_array.ravel()

            max_label = (
                int(flat_labels.max())
                if flat_labels.size
                else 0
            )

            if max_label > 65535:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="label_value_overflow",
                    detail=str(max_label),
                )
                continue

            area_counts = np.bincount(
                flat_labels
            )

            if len(area_counts) <= max_label:
                area_counts = np.pad(
                    area_counts,
                    (
                        0,
                        max_label
                        + 1
                        - len(area_counts),
                    ),
                )

            present_labels = (
                np.flatnonzero(
                    area_counts[1:] > 0
                )
                + 1
            )

            expected_labels = np.arange(
                1,
                max_label + 1,
                dtype=present_labels.dtype,
            )

            if not np.array_equal(
                present_labels,
                expected_labels,
            ):
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="noncontiguous_labels",
                    detail=(
                        f"present={present_labels.tolist()}"
                    ),
                )
                continue

            try:
                with csv_path.open(
                    newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    csv_fields = (
                        reader.fieldnames or []
                    )
                    instance_rows = list(reader)
            except Exception as exc:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="invalid_clean_csv",
                    detail=repr(exc),
                )
                continue

            csv_schema_counter[
                tuple(csv_fields)
            ] += 1

            label_field = select_field(
                csv_fields,
                LABEL_ALIASES,
            )
            detection_field = select_field(
                csv_fields,
                DETECTION_ALIASES,
            )
            score_field = select_field(
                csv_fields,
                SCORE_ALIASES,
            )
            class_field = select_field(
                csv_fields,
                CLASS_ALIASES,
            )
            csv_area_field = select_field(
                csv_fields,
                CSV_AREA_ALIASES,
            )

            if len(instance_rows) != max_label:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="csv_label_count_mismatch",
                    detail=(
                        f"csv_rows={len(instance_rows)} "
                        f"labels={max_label}"
                    ),
                )
                continue

            if label_field is None:
                source_mapping = "csv_row_order"
                csv_rows_by_label = {
                    index: row
                    for index, row in enumerate(
                        instance_rows,
                        start=1,
                    )
                }
            else:
                source_mapping = (
                    f"explicit:{label_field}"
                )
                csv_rows_by_label = {}

                try:
                    for row in instance_rows:
                        label_id = parse_optional_int(
                            row.get(label_field)
                        )

                        if label_id is None:
                            raise ValueError(
                                "blank label ID"
                            )

                        if label_id in csv_rows_by_label:
                            raise ValueError(
                                f"duplicate label ID "
                                f"{label_id}"
                            )

                        csv_rows_by_label[
                            label_id
                        ] = row
                except Exception as exc:
                    make_error(
                        errors,
                        safe_id=safe_id,
                        tile_stem=tile_stem,
                        category="invalid_csv_label_ids",
                        detail=repr(exc),
                    )
                    continue

            mapping_counter[
                source_mapping
            ] += 1

            csv_label_ids = set(
                csv_rows_by_label
            )
            map_label_ids = set(
                int(value)
                for value in present_labels
            )

            if csv_label_ids != map_label_ids:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="csv_label_id_mismatch",
                    detail=(
                        f"csv={sorted(csv_label_ids)} "
                        f"map={sorted(map_label_ids)}"
                    ),
                )
                continue

            foreground_area = int(
                area_counts[1:].sum()
            )

            if max_label > 0:
                ys, xs = np.nonzero(
                    label_array
                )

                foreground_ids = (
                    label_array[ys, xs]
                    .astype(
                        np.int64,
                        copy=False,
                    )
                )

                min_x = np.full(
                    max_label + 1,
                    image_width,
                    dtype=np.int64,
                )
                min_y = np.full(
                    max_label + 1,
                    image_height,
                    dtype=np.int64,
                )
                max_x = np.full(
                    max_label + 1,
                    -1,
                    dtype=np.int64,
                )
                max_y = np.full(
                    max_label + 1,
                    -1,
                    dtype=np.int64,
                )

                np.minimum.at(
                    min_x,
                    foreground_ids,
                    xs,
                )
                np.minimum.at(
                    min_y,
                    foreground_ids,
                    ys,
                )
                np.maximum.at(
                    max_x,
                    foreground_ids,
                    xs,
                )
                np.maximum.at(
                    max_y,
                    foreground_ids,
                    ys,
                )

                sum_x = np.bincount(
                    foreground_ids,
                    weights=xs,
                    minlength=max_label + 1,
                )
                sum_y = np.bincount(
                    foreground_ids,
                    weights=ys,
                    minlength=max_label + 1,
                )
            else:
                min_x = np.array([image_width])
                min_y = np.array([image_height])
                max_x = np.array([-1])
                max_y = np.array([-1])
                sum_x = np.array([0.0])
                sum_y = np.array([0.0])

            grid_match = GRID_PATTERN.search(
                tile_stem
            )

            if grid_match is None:
                grid_index_1 = ""
                grid_index_2 = ""

                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="unparsed_grid_indices",
                    detail=tile_stem,
                )
            else:
                grid_index_1 = int(
                    grid_match.group(1)
                )
                grid_index_2 = int(
                    grid_match.group(2)
                )

            report_source = Path(
                str(
                    report.get(
                        "input_image",
                        source_filename,
                    )
                )
            ).name

            if report_source != source_filename:
                make_error(
                    errors,
                    safe_id=safe_id,
                    tile_stem=tile_stem,
                    category="report_source_mismatch",
                    detail=(
                        f"report={report_source} "
                        f"manifest={source_filename}"
                    ),
                )
                continue

            tile_row = {
                "tile_id": tile_id,
                "task_index": task_index,
                "locality": locality,
                "locality_display": locality_display,
                "sample_id": sample_id,
                "safe_id": safe_id,
                "source_relative_path": (
                    raw_relative(source_path)
                ),
                "source_filename": source_filename,
                "source_size_bytes": (
                    manifest_row["size_bytes"]
                ),
                "source_sha256": (
                    manifest_row["sha256"]
                ),
                "grid_index_1": grid_index_1,
                "grid_index_2": grid_index_2,
                "image_width_px": image_width,
                "image_height_px": image_height,
                "image_mode": image_mode,
                "label_map_mode": label_mode,
                "label_map_dtype": (
                    str(label_array.dtype)
                ),
                "label_map_max_label": max_label,
                "clean_instance_count": max_label,
                "foreground_area_px": (
                    foreground_area
                ),
                "foreground_fraction": (
                    foreground_area
                    / (image_width * image_height)
                ),
                "zero_instance_tile": (
                    max_label == 0
                ),
                "csv_label_mapping": (
                    source_mapping
                ),
                "csv_label_field": (
                    label_field or ""
                ),
                "csv_score_field": (
                    score_field or ""
                ),
                "csv_detection_field": (
                    detection_field or ""
                ),
                "csv_class_field": (
                    class_field or ""
                ),
                "report_relative_path": (
                    output_relative(report_path)
                ),
                "clean_instances_relative_path": (
                    output_relative(csv_path)
                ),
                "label_map_relative_path": (
                    label_relative(label_path)
                ),
                "report_sha256": (
                    sha256_file(report_path)
                ),
                "clean_instances_sha256": (
                    sha256_file(csv_path)
                ),
                "label_map_sha256": (
                    sha256_file(label_path)
                ),
                "validation_status": "PASS",
            }

            tile_rows.append(tile_row)

            sample_key = (
                locality,
                sample_id,
                safe_id,
            )
            sample_record = (
                sample_stats[sample_key]
            )
            sample_record["tiles"] += 1
            sample_record[
                "candidate_instances"
            ] += max_label
            sample_record[
                "foreground_area_px"
            ] += foreground_area

            if max_label == 0:
                sample_record[
                    "zero_instance_tiles"
                ] += 1

            for label_id in present_labels:
                label_id = int(label_id)
                area_px = int(
                    area_counts[label_id]
                )

                x_min = int(min_x[label_id])
                y_min = int(min_y[label_id])
                x_max_exclusive = int(
                    max_x[label_id] + 1
                )
                y_max_exclusive = int(
                    max_y[label_id] + 1
                )

                bbox_width = (
                    x_max_exclusive - x_min
                )
                bbox_height = (
                    y_max_exclusive - y_min
                )
                bbox_area = (
                    bbox_width * bbox_height
                )

                border_distance = min(
                    x_min,
                    y_min,
                    image_width
                    - x_max_exclusive,
                    image_height
                    - y_max_exclusive,
                )

                touches_border = (
                    border_distance == 0
                )

                centroid_x = (
                    float(sum_x[label_id])
                    / area_px
                )
                centroid_y = (
                    float(sum_y[label_id])
                    / area_px
                )

                source_csv_row = (
                    csv_rows_by_label[label_id]
                )

                try:
                    detection_index = (
                        parse_optional_int(
                            source_csv_row.get(
                                detection_field
                            )
                        )
                        if detection_field
                        else None
                    )

                    segmentation_score = (
                        parse_optional_float(
                            source_csv_row.get(
                                score_field
                            )
                        )
                        if score_field
                        else None
                    )

                    class_id = (
                        parse_optional_int(
                            source_csv_row.get(
                                class_field
                            )
                        )
                        if class_field
                        else None
                    )
                except Exception as exc:
                    make_error(
                        errors,
                        safe_id=safe_id,
                        tile_stem=tile_stem,
                        category="invalid_csv_numeric_value",
                        detail=(
                            f"label={label_id}: "
                            f"{exc!r}"
                        ),
                    )
                    continue

                reported_area = None

                if csv_area_field:
                    try:
                        reported_area = (
                            parse_optional_int(
                                source_csv_row.get(
                                    csv_area_field
                                )
                            )
                        )
                    except Exception as exc:
                        make_error(
                            errors,
                            safe_id=safe_id,
                            tile_stem=tile_stem,
                            category="invalid_csv_area",
                            detail=(
                                f"label={label_id}: "
                                f"{exc!r}"
                            ),
                        )
                        continue

                    # The CSV stores the complete retained mask area.
                    # The combined integer label map stores only the
                    # visible pixels after overlapping masks are
                    # rasterised into a single-label image. Therefore,
                    # map area may be smaller than CSV area.
                    #
                    # The reverse condition is not expected and remains
                    # a validation failure.
                    if (
                        reported_area is not None
                        and area_px > reported_area
                    ):
                        make_error(
                            errors,
                            safe_id=safe_id,
                            tile_stem=tile_stem,
                            category="label_area_exceeds_csv",
                            detail=(
                                f"label={label_id} "
                                f"csv={reported_area} "
                                f"map={area_px}"
                            ),
                        )
                        continue

                label_overwritten_area = (
                    ""
                    if reported_area is None
                    else reported_area - area_px
                )

                label_visible_fraction = (
                    ""
                    if not reported_area
                    else area_px / reported_area
                )

                candidate_id = (
                    f"{safe_id}__"
                    f"{tile_stem}__"
                    f"label_{label_id:04d}"
                )

                if candidate_id in candidate_ids:
                    make_error(
                        errors,
                        safe_id=safe_id,
                        tile_stem=tile_stem,
                        category="duplicate_candidate_id",
                        detail=candidate_id,
                    )
                    continue

                candidate_ids.add(
                    candidate_id
                )

                # Record every candidate with provenance first; acceptance and production
                # export remain separate stages so rejected objects are still auditable.
                candidate_writer.writerow({
                    "candidate_id": candidate_id,
                    "tile_id": tile_id,
                    "task_index": task_index,
                    "locality": locality,
                    "locality_display": (
                        locality_display
                    ),
                    "sample_id": sample_id,
                    "safe_id": safe_id,
                    "source_relative_path": (
                        raw_relative(source_path)
                    ),
                    "source_filename": (
                        source_filename
                    ),
                    "source_sha256": (
                        manifest_row["sha256"]
                    ),
                    "grid_index_1": (
                        grid_index_1
                    ),
                    "grid_index_2": (
                        grid_index_2
                    ),
                    "instance_label": (
                        label_id
                    ),
                    "source_csv_mapping": (
                        source_mapping
                    ),
                    "detection_index": (
                        ""
                        if detection_index is None
                        else detection_index
                    ),
                    "segmentation_score": (
                        ""
                        if segmentation_score is None
                        else segmentation_score
                    ),
                    "class_id": (
                        ""
                        if class_id is None
                        else class_id
                    ),
                    # Geometry in this preliminary manifest is
                    # measured from the visible integer-label raster.
                    # Exact full-mask geometry will be computed from
                    # instance_####.png during the extraction pilot.
                    "mask_area_px": area_px,
                    "csv_mask_area_px": (
                        ""
                        if reported_area is None
                        else reported_area
                    ),
                    "label_overwritten_area_px": (
                        label_overwritten_area
                    ),
                    "label_visible_fraction_of_csv_mask": (
                        label_visible_fraction
                    ),
                    "mask_geometry_source": (
                        "clean_label_map_visible_raster"
                    ),
                    "mask_area_fraction": (
                        area_px
                        / (
                            image_width
                            * image_height
                        )
                    ),
                    "bbox_x_min": x_min,
                    "bbox_y_min": y_min,
                    "bbox_x_max_exclusive": (
                        x_max_exclusive
                    ),
                    "bbox_y_max_exclusive": (
                        y_max_exclusive
                    ),
                    "bbox_width_px": (
                        bbox_width
                    ),
                    "bbox_height_px": (
                        bbox_height
                    ),
                    "bbox_area_px": bbox_area,
                    "mask_bbox_occupancy": (
                        area_px / bbox_area
                    ),
                    "centroid_x_px": (
                        centroid_x
                    ),
                    "centroid_y_px": (
                        centroid_y
                    ),
                    "touches_tile_border": (
                        touches_border
                    ),
                    "border_distance_px": (
                        border_distance
                    ),
                    "label_map_relative_path": (
                        label_relative(
                            label_path
                        )
                    ),
                    "clean_instances_relative_path": (
                        output_relative(
                            csv_path
                        )
                    ),
                    "report_relative_path": (
                        output_relative(
                            report_path
                        )
                    ),
                    "source_csv_row_json": (
                        json.dumps(
                            source_csv_row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                })

                candidate_count += 1

            processed_tile_rows += 1

            if processed_tile_rows % 100 == 0:
                print(
                    f"Processed "
                    f"{processed_tile_rows}/"
                    f"{expected_tile_rows} tiles; "
                    f"{candidate_count} candidates",
                    flush=True,
                )

sample_rows: list[dict[str, Any]] = []

for (
    locality,
    sample_id,
    safe_id,
), values in sorted(
    sample_stats.items()
):
    sample_rows.append({
        "locality": locality,
        "locality_display": (
            "Romania"
            if locality == "Rumania"
            else locality
        ),
        "sample_id": sample_id,
        "safe_id": safe_id,
        **values,
    })

samples_per_locality: dict[
    str,
    set[str],
] = defaultdict(set)

for row in sample_rows:
    locality = str(row["locality"])
    samples_per_locality[locality].add(
        str(row["sample_id"])
    )

    key = (
        locality,
        str(row["locality_display"]),
    )
    record = locality_stats[key]

    record["tiles"] += int(
        row["tiles"]
    )
    record["zero_instance_tiles"] += int(
        row["zero_instance_tiles"]
    )
    record["candidate_instances"] += int(
        row["candidate_instances"]
    )
    record["foreground_area_px"] += int(
        row["foreground_area_px"]
    )

locality_rows: list[dict[str, Any]] = []

for (
    locality,
    locality_display,
), values in sorted(
    locality_stats.items()
):
    locality_rows.append({
        "locality": locality,
        "locality_display": locality_display,
        "samples": len(
            samples_per_locality[locality]
        ),
        **values,
    })

error_fields = [
    "safe_id",
    "tile_stem",
    "category",
    "detail",
]

write_csv(
    ERRORS_TSV,
    error_fields,
    errors,
)

summary = {
    "schema_version": (
        "non_aegina_candidate_manifest_v1"
    ),
    "status": (
        "PASS"
        if not errors
        else "FAIL"
    ),
    "task_rows": len(tasks),
    "expected_tiles": expected_tile_rows,
    "processed_tiles": processed_tile_rows,
    "tile_manifest_rows": len(tile_rows),
    "candidate_instances": candidate_count,
    "unique_tile_ids": len(tile_ids),
    "unique_candidate_ids": len(
        candidate_ids
    ),
    "report_index_entries": len(
        report_index
    ),
    "clean_csv_index_entries": len(
        csv_index
    ),
    "label_map_index_entries": len(
        label_index
    ),
    "samples": len(sample_rows),
    "localities": len(locality_rows),
    "validation_errors": len(errors),
    "csv_label_mapping_counts": dict(
        mapping_counter
    ),
    "csv_schemas": [
        {
            "fields": list(schema),
            "tile_count": count,
        }
        for schema, count in (
            csv_schema_counter.most_common()
        )
    ],
    "important_note": (
        "Candidate instances are tile-level "
        "instances. Cross-tile duplicate "
        "resolution has not yet been applied."
    ),
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

if errors:
    print()
    print("VALIDATION FAILED")
    print("Errors:", len(errors))
    print("Error table:", ERRORS_TSV)
    print()
    print("First 20 errors:")

    for error in errors[:20]:
        print(
            error["safe_id"],
            error["tile_stem"],
            error["category"],
            error["detail"],
            sep="\t",
        )

    print()
    print(
        "Candidate partial file retained:",
        candidate_partial,
    )

    sys.exit(1)

if expected_tile_rows != 2650:
    raise RuntimeError(
        f"Expected 2650 tiles, found "
        f"{expected_tile_rows}"
    )

if processed_tile_rows != 2650:
    raise RuntimeError(
        f"Processed {processed_tile_rows} "
        f"of 2650 tiles"
    )

if len(tile_rows) != 2650:
    raise RuntimeError(
        f"Tile manifest has "
        f"{len(tile_rows)} rows"
    )

if candidate_count != len(candidate_ids):
    raise RuntimeError(
        "Candidate ID count mismatch"
    )

write_csv(
    TILE_MANIFEST,
    tile_fields,
    tile_rows,
)

write_csv(
    SAMPLE_SUMMARY,
    [
        "locality",
        "locality_display",
        "sample_id",
        "safe_id",
        "tiles",
        "zero_instance_tiles",
        "candidate_instances",
        "foreground_area_px",
    ],
    sample_rows,
)

write_csv(
    LOCALITY_SUMMARY,
    [
        "locality",
        "locality_display",
        "samples",
        "tiles",
        "zero_instance_tiles",
        "candidate_instances",
        "foreground_area_px",
    ],
    locality_rows,
)

candidate_partial.replace(
    CANDIDATE_MANIFEST
)

for path in (
    TILE_MANIFEST,
    CANDIDATE_MANIFEST,
    SAMPLE_SUMMARY,
    LOCALITY_SUMMARY,
    SUMMARY_JSON,
):
    checksum_path = path.with_suffix(
        path.suffix + ".sha256"
    )
    checksum_path.write_text(
        f"{sha256_file(path)}  {path.name}\n"
    )

print()
print("Expected tiles:", expected_tile_rows)
print("Validated tiles:", len(tile_rows))
print("Candidate instances:", candidate_count)
print("Samples:", len(sample_rows))
print("Localities:", len(locality_rows))
print("CSV schemas:", len(csv_schema_counter))
print(
    "CSV label mappings:",
    dict(mapping_counter),
)
print("Validation errors:", len(errors))
print("Summary:", SUMMARY_JSON)
print("Tile manifest:", TILE_MANIFEST)
print(
    "Candidate manifest:",
    CANDIDATE_MANIFEST,
)
print()
print(
    "NON-AEGINA CANDIDATE MANIFEST: PASS"
)
