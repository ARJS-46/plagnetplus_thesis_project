"""SimCLR representation model."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from plagnetplus.models.backbones import (
    build_backbone,
)
from plagnetplus.models.base import (
    RepresentationModel,
)


class ProjectionHead(nn.Module):
    """Two-layer SimCLR projection head."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
    ) -> None:
        super().__init__()

        if min(
            input_dimension,
            hidden_dimension,
            output_dimension,
        ) <= 0:
            raise ValueError(
                "Projection dimensions must be positive."
            )

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
                bias=False,
            ),
            nn.BatchNorm1d(
                hidden_dimension
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dimension,
                output_dimension,
                bias=False,
            ),
            nn.BatchNorm1d(
                output_dimension,
                affine=False,
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        return self.network(features)


def nt_xent_loss(
    first_projection: Tensor,
    second_projection: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return symmetric in-batch NT-Xent loss and positive-pair accuracy."""

    if first_projection.shape != second_projection.shape:
        raise ValueError(
            "Projection tensors must have identical shapes."
        )

    if first_projection.ndim != 2:
        raise ValueError(
            "Projection tensors must have shape "
            "(batch, projection_dimension)."
        )

    batch_size = int(
        first_projection.shape[0]
    )

    if batch_size < 2:
        raise ValueError(
            "NT-Xent requires a batch size of at least two."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    first = functional.normalize(
        first_projection,
        dim=1,
    )

    second = functional.normalize(
        second_projection,
        dim=1,
    )

    combined = torch.cat(
        [
            first,
            second,
        ],
        dim=0,
    )

    logits = (
        combined
        @ combined.transpose(0, 1)
    ) / temperature

    self_mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=logits.device,
    )

    logits = logits.masked_fill(
        self_mask,
        torch.finfo(
            logits.dtype
        ).min,
    )

    targets = torch.arange(
        2 * batch_size,
        device=logits.device,
    )

    targets[:batch_size] += (
        batch_size
    )

    targets[batch_size:] -= (
        batch_size
    )

    loss = functional.cross_entropy(
        logits,
        targets,
    )

    positive_pair_accuracy = (
        logits.argmax(dim=1)
        == targets
    ).to(torch.float32).mean()

    return (
        loss,
        positive_pair_accuracy,
    )


class SimCLRModel(
    RepresentationModel
):
    """SimCLR with a configurable visual backbone."""

    def __init__(
        self,
        *,
        backbone_name: str,
        projection_dimension: int = 128,
        projection_hidden_dimension: int | None = None,
        temperature: float = 0.2,
        initialization: str = "scratch",
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        built = build_backbone(
            backbone_name,
            initialization=initialization,
        )

        hidden_dimension = (
            built.feature_dimension
            if projection_hidden_dimension
            is None
            else int(
                projection_hidden_dimension
            )
        )

        self.backbone = built.model
        self.backbone_name = (
            built.name
        )
        self.initialization = (
            built.initialization
        )
        self.feature_dimension = (
            built.feature_dimension
        )
        self.projection_hidden_dimension = (
            hidden_dimension
        )
        self.projection_dimension = int(
            projection_dimension
        )
        self.temperature = float(
            temperature
        )

        self.projector = ProjectionHead(
            input_dimension=(
                self.feature_dimension
            ),
            hidden_dimension=(
                self.projection_hidden_dimension
            ),
            output_dimension=(
                self.projection_dimension
            ),
        )

    def encode(
        self,
        images: Tensor,
    ) -> Tensor:
        """Return one backbone representation per image."""

        return self.backbone(images)

    def project(
        self,
        features: Tensor,
    ) -> Tensor:
        """Map backbone features into the contrastive space."""

        return self.projector(
            features
        )

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = self.encode(
            images
        )

        projections = self.project(
            features
        )

        return (
            features,
            projections,
        )

    def contrastive_objective(
        self,
        views: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        """Return SimCLR loss and positive-pair retrieval accuracy."""

        if len(views) != 2:
            raise ValueError(
                "SimCLR requires exactly two augmented views."
            )

        _, first_projection = self(
            views[0]
        )

        _, second_projection = self(
            views[1]
        )

        return nt_xent_loss(
            first_projection,
            second_projection,
            temperature=self.temperature,
        )

    def training_loss(
        self,
        views: tuple[Tensor, ...],
    ) -> Tensor:
        """Return the scalar SimCLR training loss."""

        loss, _ = (
            self.contrastive_objective(
                views
            )
        )

        return loss

    def model_metadata(
        self,
    ) -> dict[str, object]:
        """Return architecture metadata suitable for run provenance."""

        return {
            "method": "simclr",
            "backbone": (
                self.backbone_name
            ),
            "initialization": (
                self.initialization
            ),
            "feature_dimension": (
                self.feature_dimension
            ),
            "projection_hidden_dimension": (
                self.projection_hidden_dimension
            ),
            "projection_dimension": (
                self.projection_dimension
            ),
            "temperature": (
                self.temperature
            ),
        }
