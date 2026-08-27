from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tarfile
from pathlib import Path


EXCLUDED_BASENAMES = {
    "report.json",
    "raw_overlay.png",
    "clean_overlay.png",
    "clean_instances.csv",
    "_TASK_COMPLETE",
}


def default_project_root() -> Path:
    return Path(
        os.environ.get(
            "PNP_PROJECT_ROOT",
            str(Path(__file__).resolve().parents[3]),
        )
    ).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Package PlagNet per-tile non-core segmentation outputs into "
            "the per-sample .noncore.tar archives used by the dataset exporter."
        )
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=(
            project_root
            / "runs/hpc/segmentation/plagnet_non_aegina_all_v1"
        ),
        help="Production segmentation root containing tasks.tsv and output/.",
    )
    parser.add_argument(
        "--safe-id",
        action="append",
        help="Package only this safe ID; may be supplied multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing archive for a selected sample.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"No task rows found in {path}")
    required = {"task_index", "locality", "sample", "safe_id"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(
            f"tasks.tsv is missing required fields: {sorted(missing)}"
        )
    return rows


def noncore_files(sample_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sample_dir.rglob("*")
        if path.is_file() and path.name not in EXCLUDED_BASENAMES
    ]
    return sorted(paths, key=lambda p: p.relative_to(sample_dir).as_posix())


def package_task(
    *,
    root: Path,
    task: dict[str, str],
    overwrite: bool,
) -> None:
    safe_id = task["safe_id"]
    sample_dir = root / "output" / safe_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)

    archive_dir = root / "noncore_archives"
    list_dir = root / "noncore_archive_lists"
    archive_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"{safe_id}.noncore.tar"
    archive_path = archive_dir / archive_name
    archive_tmp = archive_dir / f".{archive_name}.partial"
    list_path = list_dir / f"{safe_id}.files.txt"
    list_tmp = list_dir / f"{safe_id}.files.txt.partial"
    marker = archive_dir / f"{safe_id}.PACK_COMPLETE"

    if archive_path.exists() and not overwrite:
        raise FileExistsError(
            f"Archive already exists: {archive_path}. "
            "Use --overwrite to replace it."
        )

    # Package only non-core segmentation artifacts; large transient working
    # outputs are intentionally kept out of the portable reproducibility bundle.
    paths = noncore_files(sample_dir)
    if not paths:
        raise RuntimeError(f"No non-core files found under {sample_dir}")

    relative_names = [
        path.relative_to(sample_dir).as_posix() for path in paths
    ]
    list_tmp.write_text("\n".join(relative_names) + "\n", encoding="utf-8")
    list_tmp.replace(list_path)

    archive_tmp.unlink(missing_ok=True)
    with tarfile.open(archive_tmp, "w") as archive:
        for path, relative_name in zip(paths, relative_names, strict=True):
            archive.add(path, arcname=relative_name, recursive=False)

    with tarfile.open(archive_tmp, "r") as archive:
        archived_count = sum(1 for member in archive.getmembers() if member.isfile())

    if archived_count != len(relative_names):
        archive_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Archive file-count mismatch: "
            f"listed={len(relative_names)}, archived={archived_count}"
        )

    archive_tmp.replace(archive_path)
    # Write a checksum beside each archive so transfers from HPC storage can be
    # verified before the artifacts are used downstream.
    digest = sha256_file(archive_path)
    checksum_path = archive_dir / f"{archive_name}.sha256"
    checksum_path.write_text(
        f"{digest}  {archive_name}\n",
        encoding="utf-8",
    )
    marker.touch()

    print(
        f"PACK COMPLETE: {task['locality']} / {task['sample']} | "
        f"files={len(relative_names)} | {archive_path}"
    )


def main() -> None:
    args = parse_args()
    root = args.segmentation_root.expanduser().resolve()
    tasks_path = root / "tasks.tsv"
    if not tasks_path.is_file():
        raise FileNotFoundError(tasks_path)

    tasks = read_tasks(tasks_path)
    if args.safe_id:
        requested = set(args.safe_id)
        known = {row["safe_id"] for row in tasks}
        missing = sorted(requested - known)
        if missing:
            raise RuntimeError(f"Unknown safe IDs: {missing}")
        tasks = [row for row in tasks if row["safe_id"] in requested]

    for task in tasks:
        package_task(root=root, task=task, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
