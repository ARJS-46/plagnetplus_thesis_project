from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plagnetplus.data.dataset import (
    CrystalDataset,
    locality_sampling_weights,
)
from plagnetplus.data.transforms import MultiViewTransform, PadToSquare


def save_image(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (value, value, value))
    image.save(path)


def multi_locality_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crystal_id": ["a1", "a2", "m1"],
            "locality": ["aegina", "aegina", "methana"],
            "sample_id": ["Aeg6", "Aeg12", "Met1"],
            "scan_id": ["aegina/Aeg6", "aegina/Aeg12", "methana/scan1"],
            "relative_path": [
                "aegina/Aeg6/a1.png",
                "aegina/Aeg12/a2.png",
                "methana/Met1/m1.png",
            ],
            "split": ["train", "test", "train"],
        }
    )


def test_dataset_loads_multiple_localities(tmp_path: Path):
    manifest = multi_locality_manifest()

    for row, size, value in zip(
        manifest.itertuples(index=False),
        [(20, 10), (12, 18), (14, 14)],
        [20, 80, 140],
        strict=True,
    ):
        save_image(tmp_path / row.relative_path, size, value)

    dataset = CrystalDataset(
        manifest,
        tmp_path,
        verify_files="all",
    )

    assert len(dataset) == 3
    assert dataset[0]["image"].mode == "RGB"
    assert dataset[0]["metadata"]["locality"] == "aegina"
    assert dataset[2]["metadata"]["locality"] == "methana"

    summary = dataset.summary()
    assert summary["locality_counts"] == {"aegina": 2, "methana": 1}
    assert summary["scan_groups"] == 3


def test_dataset_filters_by_locality_and_split(tmp_path: Path):
    manifest = multi_locality_manifest()

    for row in manifest.itertuples(index=False):
        save_image(tmp_path / row.relative_path, (10, 10), 50)

    dataset = CrystalDataset(
        manifest,
        tmp_path,
        localities="aegina",
        splits="train",
    )

    assert len(dataset) == 1
    assert dataset[0]["metadata"]["crystal_id"] == "a1"


def test_missing_image_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing"):
        CrystalDataset(
            multi_locality_manifest(),
            tmp_path,
            verify_files="all",
        )


def test_locality_weights_equalise_total_weight():
    manifest = multi_locality_manifest()
    weights = locality_sampling_weights(manifest)

    aegina_total = weights[
        manifest["locality"].to_numpy() == "aegina"
    ].sum()
    methana_total = weights[
        manifest["locality"].to_numpy() == "methana"
    ].sum()

    assert np.isclose(aegina_total, methana_total)


def test_pad_to_square_preserves_content_dimensions():
    image = Image.new("RGB", (20, 10), (255, 255, 255))
    padded = PadToSquare(fill=0)(image)

    assert padded.size == (20, 20)


def test_multi_view_transform_returns_requested_views():
    image = Image.new("RGB", (10, 10), (255, 255, 255))
    transform = MultiViewTransform(lambda value: value.size, views=3)

    assert transform(image) == ((10, 10), (10, 10), (10, 10))
