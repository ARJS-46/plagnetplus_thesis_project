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


def test_configuration_from_mapping(
    tmp_path: Path,
):
    mapping = {
        "run": {
            "name": "test_run",
            "output_root": (
                str(
                    tmp_path
                    / "runs"
                )
            ),
        },
        "model": {
            "backbone": "resnet-18",
            "initialization": "scratch",
            "projection_dimension": 128,
            "temperature": 0.2,
        },
        "data": {
            "benchmark_id": "benchmark_v1",
            "sampling": "natural_frequency",
            "train_manifest": "train.csv",
            "validation_manifest": "validation.csv",
            "test_manifest": "test.csv",
            "image_root_map": "root_map.json",
            "transform_config": "transform.json",
            "expected_train_rows": 10,
            "expected_validation_rows": 2,
            "expected_test_rows": 3,
        },
        "optimization": {
            "epochs": 5,
            "batch_size": 4,
            "learning_rate": 3.0e-4,
            "minimum_learning_rate": 0.0,
            "weight_decay": 1.0e-4,
            "warmup_epochs": 1,
            "gradient_clip_norm": 5.0,
        },
        "runtime": {
            "num_workers": 2,
            "checkpoint_every": 1,
            "validation_every": 1,
            "seed": 42,
            "use_amp": True,
        },
    }

    config = config_from_mapping(
        mapping,
        source_directory=tmp_path,
    )

    assert config.backbone_name == (
        "resnet18"
    )

    assert config.run_name == (
        "test_run"
    )

    assert config.epochs == 5
    assert config.batch_size == 4

    assert config.train_manifest == (
        tmp_path
        / "train.csv"
    ).resolve()


def test_learning_rate_warmup_and_cosine():
    values = [
        learning_rate_for_epoch(
            epoch_index=epoch,
            total_epochs=5,
            warmup_epochs=1,
            base_learning_rate=1.0,
            minimum_learning_rate=0.1,
        )
        for epoch in range(5)
    ]

    assert values[0] == pytest.approx(
        1.0
    )

    assert values[1] == pytest.approx(
        1.0
    )

    assert values[-1] == pytest.approx(
        0.1
    )

    assert all(
        later <= earlier
        for earlier, later in zip(
            values[1:],
            values[2:],
        )
    )


def test_train_and_validation_epoch():
    torch.manual_seed(42)

    model = ToySimCLR()

    loader = DataLoader(
        ToyTwoViewDataset(),
        batch_size=4,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
    )

    initial = {
        name: value.detach().clone()
        for name, value
        in model.state_dict().items()
    }

    train_metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epoch=1,
        learning_rate=1.0e-3,
        use_amp=False,
        scaler=None,
        gradient_clip_norm=5.0,
    )

    assert train_metrics.split == (
        "train"
    )

    assert train_metrics.images == 8
    assert train_metrics.batches == 2

    assert torch.isfinite(
        torch.tensor(
            train_metrics.loss
        )
    )

    changed = any(
        not torch.equal(
            initial[name],
            current,
        )
        for name, current
        in model.state_dict().items()
    )

    assert changed

    validation_metrics = (
        evaluate_contrastive_epoch(
            model=model,
            loader=loader,
            device=torch.device(
                "cpu"
            ),
            epoch=1,
            learning_rate=1.0e-3,
            use_amp=False,
        )
    )

    assert validation_metrics.split == (
        "validation"
    )

    assert validation_metrics.images == 8
    assert validation_metrics.batches == 2


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


def test_history_csv(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "history.csv"
    )

    write_history_csv_atomic(
        path,
        [{
            "epoch": 1,
            "split": "train",
            "loss": 2.5,
            "positive_pair_accuracy": 0.25,
            "learning_rate": 1.0e-3,
            "batches": 2,
            "images": 8,
            "seconds": 0.5,
        }],
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert "positive_pair_accuracy" in text
    assert "2.5" in text
