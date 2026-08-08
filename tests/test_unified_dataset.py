import json

import pandas as pd
import pytest
from PIL import Image

from plagnetplus.data.transforms import MultiViewTransform
from plagnetplus.data.unified_dataset import (
    UnifiedCrystalDataset,
    load_image_root_map,
)


def _save_image(path, *, size, colour):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.new(
        "RGB",
        size,
        colour,
    ).save(path)


def _root_map(tmp_path):
    payload = {
        "environment_variable": (
            "PNP_DATA_ROOT"
        ),
        "portable_roots": {
            "aegina": (
                "${PNP_DATA_ROOT}/aegina"
            ),
            "non_aegina": (
                "${PNP_DATA_ROOT}/non_aegina"
            ),
        },
        "current_resolved_roots": {
            "aegina": str(
                tmp_path / "aegina"
            ),
            "non_aegina": str(
                tmp_path / "non_aegina"
            ),
        },
    }

    path = tmp_path / "root_map.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return path


def _manifest():
    return pd.DataFrame({
        "manifest_order": [2, 0, 1],
        "crystal_id": ["a1", "n1", "n2"],
        "locality": [
            "Aegina",
            "FCT",
            "Methana",
        ],
        "sample_id": [
            "Aeg6",
            "6861",
            "M1",
        ],
        "physical_sample_id": [
            "Aeg6",
            "6861",
            "M1",
        ],
        "acquisition_id": [
            "Aeg6",
            "6861_recoated",
            "M1",
        ],
        "scan_id": [
            "aegina/Aeg6",
            "FCT/6861_recoated",
            "Methana/M1",
        ],
        "split_group_id": [
            "Aegina__Aeg6",
            "FCT__6861",
            "Methana__M1",
        ],
        "benchmark_id": ["test"] * 3,
        "benchmark_split": [
            "test",
            "train",
            "validation",
        ],
        "image_root_key": [
            "aegina",
            "non_aegina",
            "non_aegina",
        ],
        "image_relative_path": [
            "images/a1.png",
            "images/n1.png",
            "images/n2.png",
        ],
        "image_locator": [
            "aegina::images/a1.png",
            "non_aegina::images/n1.png",
            "non_aegina::images/n2.png",
        ],
        "image_sha256": ["0" * 64] * 3,
        "image_width_px": [11, 12, 13],
        "image_height_px": [11, 12, 13],
        "source_dataset": [
            "aegina",
            "non_aegina",
            "non_aegina",
        ],
    })


def _build_fixture(tmp_path):
    aegina = tmp_path / "aegina"
    non_aegina = tmp_path / "non_aegina"

    aegina.mkdir()
    non_aegina.mkdir()

    _save_image(
        aegina / "images/a1.png",
        size=(11, 11),
        colour=(10, 20, 30),
    )

    _save_image(
        non_aegina / "images/n1.png",
        size=(12, 12),
        colour=(40, 50, 60),
    )

    _save_image(
        non_aegina / "images/n2.png",
        size=(13, 13),
        colour=(70, 80, 90),
    )

    return _root_map(tmp_path)


def test_root_map_resolves_portable_roots(tmp_path):
    root_map = _build_fixture(
        tmp_path
    )

    roots = load_image_root_map(
        root_map,
        data_root=tmp_path,
    )

    assert roots["aegina"] == (
        tmp_path / "aegina"
    ).resolve()

    assert roots["non_aegina"] == (
        tmp_path / "non_aegina"
    ).resolve()


def test_unified_dataset_loads_both_roots_in_manifest_order(
    tmp_path,
):
    root_map = _build_fixture(
        tmp_path
    )

    dataset = UnifiedCrystalDataset(
        _manifest(),
        root_map,
        data_root=tmp_path,
        verify_files="all",
    )

    assert len(dataset) == 3

    assert [
        dataset[index]["metadata"][
            "crystal_id"
        ]
        for index in range(len(dataset))
    ] == ["n1", "n2", "a1"]

    first = dataset[0]

    assert first["image"].mode == "RGB"
    assert first["image"].size == (12, 12)
    assert first["metadata"][
        "benchmark_split"
    ] == "train"

    assert first["metadata"][
        "image_root_key"
    ] == "non_aegina"


def test_unified_dataset_filters_by_split(
    tmp_path,
):
    root_map = _build_fixture(
        tmp_path
    )

    dataset = UnifiedCrystalDataset(
        _manifest(),
        root_map,
        data_root=tmp_path,
        splits="validation",
        verify_files="all",
    )

    assert len(dataset) == 1

    assert dataset[0]["metadata"][
        "crystal_id"
    ] == "n2"


def test_multiview_transform_receives_fresh_rgb_images(
    tmp_path,
):
    root_map = _build_fixture(
        tmp_path
    )

    transform = MultiViewTransform(
        lambda image: image,
        views=2,
    )

    dataset = UnifiedCrystalDataset(
        _manifest(),
        root_map,
        data_root=tmp_path,
        splits="train",
        transform=transform,
        verify_files="all",
    )

    views = dataset[0]["image"]

    assert len(views) == 2

    assert views[0].mode == "RGB"
    assert views[1].mode == "RGB"

    assert views[0].size == (12, 12)
    assert views[1].size == (12, 12)

    assert views[0] is not views[1]


def test_parent_traversal_is_rejected(
    tmp_path,
):
    root_map = _build_fixture(
        tmp_path
    )

    manifest = _manifest()
    manifest.loc[
        0,
        "image_relative_path",
    ] = "../escape.png"

    with pytest.raises(
        ValueError,
        match="Parent traversal",
    ):
        UnifiedCrystalDataset(
            manifest,
            root_map,
            data_root=tmp_path,
            verify_files="none",
        )


def test_locality_sampling_weights_align_with_rows(
    tmp_path,
):
    root_map = _build_fixture(
        tmp_path
    )

    dataset = UnifiedCrystalDataset(
        _manifest(),
        root_map,
        data_root=tmp_path,
        verify_files="all",
    )

    weights = dataset.sampling_weights()

    assert weights.shape == (3,)
    assert weights.mean() == pytest.approx(
        1.0
    )
