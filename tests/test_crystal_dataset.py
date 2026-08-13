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










def test_pad_to_square_preserves_content_dimensions():
    image = Image.new("RGB", (20, 10), (255, 255, 255))
    padded = PadToSquare(fill=0)(image)

    assert padded.size == (20, 20)
