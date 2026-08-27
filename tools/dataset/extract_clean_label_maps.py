from __future__ import annotations

from pathlib import Path, PurePosixPath
import os
import csv
import shutil
import tarfile


IRP_ROOT = Path(
    os.environ.get(
        "PNP_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

SEG_ROOT = (
    IRP_ROOT
    / "runs/hpc/segmentation/plagnet_non_aegina_all_v1"
)

ARCHIVE_ROOT = SEG_ROOT / "noncore_archives"

LABEL_ROOT = (
    IRP_ROOT
    / "data/interim/plagnetplus"
    / "non_aegina_clean_labels_v1"
)

INVENTORY_PATH = (
    LABEL_ROOT
    / "clean_label_map_inventory.tsv"
)

TARGET_NAME = "clean_instance_labels_uint16.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def safe_member_parts(name: str) -> tuple[str, ...]:
    parts = tuple(
        part
        for part in PurePosixPath(name).parts
        if part not in ("", ".")
    )

    if not parts:
        raise ValueError(
            f"Empty archive member path: {name!r}"
        )

    if any(part == ".." for part in parts):
        raise ValueError(
            f"Unsafe archive member path: {name!r}"
        )

    return parts


LABEL_ROOT.mkdir(parents=True, exist_ok=True)

inventory_rows: list[dict[str, object]] = []

archives = sorted(
    ARCHIVE_ROOT.glob("*.noncore.tar")
)

for archive_index, archive_path in enumerate(
    archives,
    start=1,
):
    suffix = ".noncore.tar"

    if not archive_path.name.endswith(suffix):
        raise RuntimeError(
            f"Unexpected archive name: {archive_path.name}"
        )

    safe_id = archive_path.name[: -len(suffix)]

    extracted_for_archive = 0

    print(
        f"[{archive_index:02d}/{len(archives):02d}] "
        f"{archive_path.name}"
    )

    # Extract only the cleaned instance-label maps required downstream rather
    # than unpacking the complete segmentation working archive.
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            if (
                PurePosixPath(member.name).name
                != TARGET_NAME
            ):
                continue

            parts = safe_member_parts(member.name)

            destination = (
                LABEL_ROOT
                / safe_id
                / Path(*parts)
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            needs_write = (
                not destination.is_file()
                or destination.stat().st_size
                != member.size
            )

            if needs_write:
                source = archive.extractfile(member)

                if source is None:
                    raise RuntimeError(
                        f"Could not read {member.name} "
                        f"from {archive_path}"
                    )

                temporary = destination.with_suffix(
                    destination.suffix + ".partial"
                )

                with source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)

                temporary.replace(destination)

            actual_size = destination.stat().st_size

            if actual_size != member.size:
                raise RuntimeError(
                    f"Size mismatch for {destination}: "
                    f"{actual_size} != {member.size}"
                )

            with destination.open("rb") as handle:
                signature = handle.read(8)

            if signature != PNG_SIGNATURE:
                raise RuntimeError(
                    f"Invalid PNG signature: {destination}"
                )

            inventory_rows.append({
                "safe_id": safe_id,
                "archive_name": archive_path.name,
                "archive_member": member.name,
                "label_map_relative_path": (
                    destination.relative_to(
                        LABEL_ROOT
                    ).as_posix()
                ),
                "size_bytes": actual_size,
            })

            extracted_for_archive += 1

    print(
        f"  clean label maps: "
        f"{extracted_for_archive}"
    )

fields = [
    "safe_id",
    "archive_name",
    "archive_member",
    "label_map_relative_path",
    "size_bytes",
]

with INVENTORY_PATH.open(
    "w",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(inventory_rows)

total_bytes = sum(
    int(row["size_bytes"])
    for row in inventory_rows
)

print()
print("Archives processed:", len(archives))
print("Label maps extracted:", len(inventory_rows))
print("Total bytes:", total_bytes)
print("Inventory:", INVENTORY_PATH)

assert len(archives) == 20
assert len(inventory_rows) == 2650
assert len({
    row["label_map_relative_path"]
    for row in inventory_rows
}) == 2650

print("CLEAN LABEL-MAP EXTRACTION: PASS")
