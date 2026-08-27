from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import cv2
import mmcv
import mmdet
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from pycocotools import mask as mask_util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--duplicate-iou", type=float, default=0.50)
    parser.add_argument(
        "--duplicate-containment",
        type=float,
        default=0.80,
    )
    return parser.parse_args()


def decode_mask(item):
    if isinstance(item, dict):
        array = mask_util.decode(item)
    else:
        array = np.asarray(item)

    if array.ndim == 3:
        array = array[:, :, 0]

    return array.astype(bool)


def parse_result(result):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result = result
        segm_result = None

    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]

    bboxes = []
    labels = []
    masks = []

    for class_id, class_bboxes in enumerate(bbox_result):
        class_bboxes = np.asarray(class_bboxes)

        if class_bboxes.size == 0:
            class_bboxes = np.empty((0, 5), dtype=np.float32)

        bboxes.append(class_bboxes)
        labels.append(
            np.full(
                class_bboxes.shape[0],
                class_id,
                dtype=np.int64,
            )
        )

        if segm_result is not None:
            masks.extend(segm_result[class_id])

    if bboxes:
        bboxes = np.vstack(bboxes)
        labels = np.concatenate(labels)
    else:
        bboxes = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0,), dtype=np.int64)

    return bboxes, labels, masks


def overlap_metrics(mask_a, mask_b):
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = area_a + area_b - intersection

    iou = intersection / union if union else 0.0

    smaller = min(area_a, area_b)
    containment = intersection / smaller if smaller else 0.0

    return {
        "intersection_px": intersection,
        "iou": float(iou),
        "containment": float(containment),
    }


def suppress_duplicates(
    candidates,
    iou_threshold,
    containment_threshold,
):
    # Prioritise high-confidence, larger detections before overlap suppression so
    # the retained instance is deterministic when masks represent the same crystal.
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -candidates[index]["score"],
            -candidates[index]["area_px"],
            candidates[index]["detection_index"],
        ),
    )

    retained = []
    suppressed = []

    for candidate_index in order:
        candidate = candidates[candidate_index]
        duplicate_match = None

        for retained_index, retained_candidate in enumerate(retained):
            metrics = overlap_metrics(
                candidate["mask"],
                retained_candidate["mask"],
            )

            if (
                metrics["iou"] >= iou_threshold
                or metrics["containment"]
                >= containment_threshold
            ):
                duplicate_match = {
                    "suppressed_detection_index":
                        candidate["detection_index"],
                    "suppressed_score": candidate["score"],
                    "suppressed_area_px": candidate["area_px"],
                    "retained_detection_index":
                        retained_candidate["detection_index"],
                    "retained_score":
                        retained_candidate["score"],
                    "retained_area_px":
                        retained_candidate["area_px"],
                    "iou": metrics["iou"],
                    "containment": metrics["containment"],
                    "intersection_px":
                        metrics["intersection_px"],
                    "reason": (
                        "iou"
                        if metrics["iou"] >= iou_threshold
                        else "containment"
                    ),
                }
                break

        if duplicate_match is None:
            retained.append(candidate)
        else:
            suppressed.append(duplicate_match)

    retained = sorted(
        retained,
        key=lambda item: item["detection_index"],
    )

    return retained, suppressed


def colour_for_instance(instance_id):
    hue = int((instance_id * 47) % 180)

    hsv = np.uint8([[[hue, 210, 235]]])
    bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR,
    )[0, 0]

    return tuple(int(value) for value in bgr)


def save_overlay(image, retained, path):
    overlay = image.copy()

    for instance_id, item in enumerate(retained, start=1):
        mask = item["mask"]
        colour = colour_for_instance(instance_id)
        colour_array = np.asarray(colour, dtype=np.float32)

        pixels = overlay[mask].astype(np.float32)
        overlay[mask] = (
            0.55 * pixels
            + 0.45 * colour_array
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            colour,
            2,
        )

        x1, y1, x2, y2 = [
            int(round(value))
            for value in item["bbox"][:4]
        ]

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            colour,
            2,
        )

        text = "P{} {:.2f}".format(
            instance_id,
            item["score"],
        )

        cv2.putText(
            overlay,
            text,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

    success = cv2.imwrite(str(path), overlay)
    assert success, "Failed to save {}".format(path)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def process_image(
    model,
    image_path,
    output_dir,
    score_threshold,
    duplicate_iou,
    duplicate_containment,
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None, image_path

    height, width = image.shape[:2]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
    # Run inference on the original cold-CL tile; all later masks and geometry
    # remain in this native pixel coordinate system.
    result = inference_detector(model, str(image_path))
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    bboxes, labels, encoded_masks = parse_result(result)

    if encoded_masks:
        assert len(encoded_masks) == len(bboxes), (
            len(encoded_masks),
            len(bboxes),
        )

    candidates = []
    raw_rows = []

    for detection_index in range(len(bboxes)):
        bbox = bboxes[detection_index]
        score = float(bbox[4])
        above_threshold = score >= score_threshold

        mask = None
        area_px = 0

        if encoded_masks:
            mask = decode_mask(encoded_masks[detection_index])

            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            area_px = int(mask.sum())

        raw_rows.append(
            {
                "detection_index": detection_index,
                "class_id": int(labels[detection_index]),
                "class_name": "Plagioclase",
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "score": score,
                "above_score_threshold": above_threshold,
                "area_px": area_px,
            }
        )

        if above_threshold and mask is not None:
            candidates.append(
                {
                    "detection_index": detection_index,
                    "bbox": bbox,
                    "score": score,
                    "mask": mask,
                    "area_px": area_px,
                }
            )

    # Suppress strongly overlapping detections before writing clean masks while
    # retaining a separate record of what was removed and why.
    retained, suppressed = suppress_duplicates(
        candidates,
        duplicate_iou,
        duplicate_containment,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_masks_dir = output_dir / "raw_masks"
    clean_masks_dir = output_dir / "clean_masks"

    raw_masks_dir.mkdir(exist_ok=True)
    clean_masks_dir.mkdir(exist_ok=True)

    raw_label_map = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    for raw_instance_id, candidate in enumerate(
        candidates,
        start=1,
    ):
        mask = candidate["mask"]

        cv2.imwrite(
            str(
                raw_masks_dir
                / "raw_instance_{:04d}.png".format(
                    raw_instance_id
                )
            ),
            mask.astype(np.uint8) * 255,
        )

        raw_label_map[mask] = raw_instance_id

    # Write cleaned instance IDs contiguously so downstream mask filenames and
    # label-map values have a simple one-to-one correspondence.
    clean_label_map = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    clean_rows = []

    for clean_instance_id, candidate in enumerate(
        retained,
        start=1,
    ):
        mask = candidate["mask"]
        bbox = candidate["bbox"]

        mask_name = "instance_{:04d}.png".format(
            clean_instance_id
        )

        cv2.imwrite(
            str(clean_masks_dir / mask_name),
            mask.astype(np.uint8) * 255,
        )

        clean_label_map[mask] = clean_instance_id

        clean_rows.append(
            {
                "instance_id": clean_instance_id,
                "source_detection_index":
                    candidate["detection_index"],
                "class_name": "Plagioclase",
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "score": candidate["score"],
                "area_px": candidate["area_px"],
                "mask_path": (
                    "clean_masks/" + mask_name
                ),
            }
        )

    cv2.imwrite(
        str(output_dir / "raw_instance_labels_uint16.png"),
        raw_label_map,
    )

    cv2.imwrite(
        str(output_dir / "clean_instance_labels_uint16.png"),
        clean_label_map,
    )

    model.show_result(
        str(image_path),
        result,
        score_thr=score_threshold,
        out_file=str(output_dir / "raw_overlay.png"),
    )

    save_overlay(
        image,
        retained,
        output_dir / "clean_overlay.png",
    )

    write_csv(
        output_dir / "raw_detections.csv",
        raw_rows,
        [
            "detection_index",
            "class_id",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "score",
            "above_score_threshold",
            "area_px",
        ],
    )

    write_csv(
        output_dir / "clean_instances.csv",
        clean_rows,
        [
            "instance_id",
            "source_detection_index",
            "class_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "score",
            "area_px",
            "mask_path",
        ],
    )

    write_csv(
        output_dir / "duplicate_suppressions.csv",
        suppressed,
        [
            "suppressed_detection_index",
            "suppressed_score",
            "suppressed_area_px",
            "retained_detection_index",
            "retained_score",
            "retained_area_px",
            "iou",
            "containment",
            "intersection_px",
            "reason",
        ],
    )

    report = {
        "status": "PASS",
        "input_image": str(image_path),
        "width": width,
        "height": height,
        "score_threshold": score_threshold,
        "duplicate_iou_threshold": duplicate_iou,
        "duplicate_containment_threshold":
            duplicate_containment,
        "detections_all": int(len(bboxes)),
        "detections_above_score_threshold":
            int(len(candidates)),
        "instances_after_duplicate_suppression":
            int(len(retained)),
        "duplicate_detections_suppressed":
            int(len(suppressed)),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
        ),
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return report


def main():
    args = parse_args()

    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    assert torch.cuda.is_available()

    model_load_start = time.perf_counter()

    model = init_detector(
        str(config),
        str(checkpoint),
        device="cuda:0",
    )

    model.CLASSES = ("Plagioclase",)

    torch.cuda.synchronize()
    model_load_seconds = (
        time.perf_counter() - model_load_start
    )

    with manifest.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        manifest_rows = list(csv.DictReader(handle))

    summary_rows = []

    for row in manifest_rows:
        locality = row["locality"]
        image_path = (
            input_root
            / row["input_relative_path"]
        )

        image_output = (
            output_root
            / locality
            / image_path.stem
        )

        report_path = image_output / "report.json"

        if report_path.exists():
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            report["locality"] = locality
            report["source_relative_path"] = (
                row["source_relative_path"]
            )
            summary_rows.append(report)
            print(
                f"Skipping completed image: {image_path}"
            )
            continue

        print(
            "\nProcessing {}: {}".format(
                locality,
                image_path,
            )
        )

        report = process_image(
            model=model,
            image_path=image_path,
            output_dir=image_output,
            score_threshold=args.score_thr,
            duplicate_iou=args.duplicate_iou,
            duplicate_containment=(
                args.duplicate_containment
            ),
        )

        report["locality"] = locality
        report["source_relative_path"] = (
            row["source_relative_path"]
        )

        summary_rows.append(report)

        print(json.dumps(report, indent=2, sort_keys=True))

    aggregate = {
        "status": "PASS",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "model_load_seconds": model_load_seconds,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(
            torch.cuda.get_device_capability(0)
        ),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "mmcv": mmcv.__version__,
        "mmdetection": mmdet.__version__,
        "images": summary_rows,
    }

    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "pilot_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n" + json.dumps(aggregate, indent=2, sort_keys=True))
    print("PLAGNET FOUR-LOCALITY PILOT: PASS")


if __name__ == "__main__":
    main()
