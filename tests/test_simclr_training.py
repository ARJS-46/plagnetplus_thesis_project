from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from plagnetplus.models.simclr import (
    nt_xent_loss,
)
from plagnetplus.training import (
    config_from_mapping,
    evaluate_contrastive_epoch,
    learning_rate_for_epoch,
    load_checkpoint,
    save_checkpoint_atomic,
    train_one_epoch,
    write_history_csv_atomic,
)


class ToyTwoViewDataset(Dataset):
    def __init__(
        self,
        rows: int = 8,
    ) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        base = torch.tensor(
            [
                float(index),
                float(index + 1),
                float(index + 2),
                float(index + 3),
            ],
            dtype=torch.float32,
        )

        return {
            "image": (
                base,
                base + 0.05,
            )
        }


class ToySimCLR(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Linear(
            4,
            8,
        )

        self.projector = nn.Linear(
            8,
            4,
        )

    def contrastive_objective(
        self,
        views,
    ):
        first = self.projector(
            self.encoder(
                views[0]
            )
        )

        second = self.projector(
            self.encoder(
                views[1]
            )
        )

        return nt_xent_loss(
            first,
            second,
            temperature=0.2,
        )








def test_checkpoint_round_trip(
    tmp_path: Path,
):
    torch.manual_seed(42)

    model = ToySimCLR()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    path = (
        tmp_path
        / "checkpoint.pt"
    )

    history = [{
        "epoch": 1,
        "split": "train",
        "loss": 1.0,
        "positive_pair_accuracy": 0.5,
        "learning_rate": 1.0e-3,
        "batches": 2,
        "images": 8,
        "seconds": 1.0,
    }]

    save_checkpoint_atomic(
        path,
        completed_epoch=1,
        model=model,
        optimizer=optimizer,
        scaler=None,
        history=history,
        config={"run_name": "test"},
        benchmark_id="benchmark_v1",
        manifest_fingerprint_sha256=(
            "abc123"
        ),
        train_manifest_sha256=(
            "def456"
        ),
        train_generator=generator,
    )

    assert path.is_file()

    reloaded_model = ToySimCLR()

    reloaded_optimizer = (
        torch.optim.AdamW(
            reloaded_model.parameters(),
            lr=1.0e-3,
        )
    )

    reloaded_generator = (
        torch.Generator()
    )

    checkpoint = load_checkpoint(
        path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        device=torch.device("cpu"),
        expected_benchmark_id=(
            "benchmark_v1"
        ),
        expected_manifest_fingerprint_sha256=(
            "abc123"
        ),
        train_generator=(
            reloaded_generator
        ),
    )

    assert checkpoint[
        "completed_epoch"
    ] == 1

    for name, expected in (
        model.state_dict().items()
    ):
        assert torch.equal(
            expected,
            reloaded_model
            .state_dict()[name],
        )

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        load_checkpoint(
            path,
            model=reloaded_model,
            optimizer=(
                reloaded_optimizer
            ),
            device=torch.device(
                "cpu"
            ),
            expected_benchmark_id=(
                "benchmark_v1"
            ),
            expected_manifest_fingerprint_sha256=(
                "wrong"
            ),
        )
