"""Native SwAV model, balanced assignments, and swapped prediction loss."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from plagnetplus.models.backbones import build_backbone
from plagnetplus.models.base import RepresentationModel


class SwAVProjectionHead(nn.Module):
    """Two-layer projection head used before the prototype layer."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int = 2048,
        output_dimension: int = 128,
    ) -> None:
        super().__init__()

        if min(input_dimension, hidden_dimension, output_dimension) <= 0:
            raise ValueError(
                "Projection dimensions must be positive."
            )

        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.output_dimension = int(output_dimension)

        self.network = nn.Sequential(
            nn.Linear(
                self.input_dimension,
                self.hidden_dimension,
            ),
            nn.BatchNorm1d(
                self.hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.hidden_dimension,
                self.output_dimension,
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """Project backbone features into the assignment space."""

        if features.ndim != 2:
            raise ValueError(
                "Projection-head input must have shape "
                "(batch, feature_dimension)."
            )

        if features.shape[1] != self.input_dimension:
            raise ValueError(
                "Projection-head feature dimension mismatch: "
                f"{features.shape[1]} != "
                f"{self.input_dimension}."
            )

        return self.network(features)


@torch.no_grad()
def sinkhorn_knopp(
    scores: Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 3,
) -> Tensor:
    """Return balanced soft assignments for prototype scores."""

    if scores.ndim != 2:
        raise ValueError(
            "scores must have shape "
            "(batch, prototype_count)."
        )

    batch_size, prototype_count = scores.shape

    if batch_size < 2:
        raise ValueError(
            "Sinkhorn-Knopp requires at least "
            "two batch items."
        )

    if prototype_count < 2:
        raise ValueError(
            "Sinkhorn-Knopp requires at least "
            "two prototypes."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    if iterations < 1:
        raise ValueError(
            "iterations must be at least one."
        )

    stable_scores = scores.detach().to(
        torch.float32
    )

    stable_scores = (
        stable_scores
        - stable_scores.max()
    )

    assignments = torch.exp(
        stable_scores / float(epsilon)
    ).transpose(0, 1)

    total_mass = assignments.sum()

    if (
        not torch.isfinite(total_mass)
        or total_mass <= 0
    ):
        raise FloatingPointError(
            "Sinkhorn-Knopp received invalid "
            "assignment mass."
        )

    assignments /= total_mass

    floor = torch.finfo(
        assignments.dtype
    ).eps

    for _ in range(iterations):
        assignments /= assignments.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(floor)

        assignments /= float(
            prototype_count
        )

        assignments /= assignments.sum(
            dim=0,
            keepdim=True,
        ).clamp_min(floor)

        assignments /= float(
            batch_size
        )

    assignments *= float(
        batch_size
    )

    result = assignments.transpose(
        0,
        1,
    ).contiguous()

    if not torch.isfinite(
        result
    ).all():
        raise FloatingPointError(
            "Sinkhorn-Knopp produced "
            "non-finite assignments."
        )

    return result


def swapped_prediction_loss(
    first_scores: Tensor,
    second_scores: Tensor,
    *,
    temperature: float = 0.1,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return symmetric loss and detached assignment targets."""

    if (
        first_scores.shape
        != second_scores.shape
    ):
        raise ValueError(
            "Prototype-score tensors must "
            "have identical shapes."
        )

    if first_scores.ndim != 2:
        raise ValueError(
            "Prototype-score tensors must "
            "have shape "
            "(batch, prototype_count)."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    first_assignments = sinkhorn_knopp(
        first_scores,
        epsilon=sinkhorn_epsilon,
        iterations=sinkhorn_iterations,
    )

    second_assignments = sinkhorn_knopp(
        second_scores,
        epsilon=sinkhorn_epsilon,
        iterations=sinkhorn_iterations,
    )

    first_predictions = (
        functional.log_softmax(
            first_scores.to(
                torch.float32
            )
            / float(temperature),
            dim=1,
        )
    )

    second_predictions = (
        functional.log_softmax(
            second_scores.to(
                torch.float32
            )
            / float(temperature),
            dim=1,
        )
    )

    predict_second_from_first = -(
        second_assignments
        * first_predictions
    ).sum(
        dim=1
    ).mean()

    predict_first_from_second = -(
        first_assignments
        * second_predictions
    ).sum(
        dim=1
    ).mean()

    loss = 0.5 * (
        predict_second_from_first
        + predict_first_from_second
    )

    return (
        loss,
        first_assignments,
        second_assignments,
    )


class SwAVModel(RepresentationModel):
    """SwAV with a PlagNetPlus backbone and prototypes."""

    def __init__(
        self,
        *,
        backbone_name: str,
        initialization: str = "scratch",
        projection_hidden_dimension: int = 2048,
        projection_dimension: int = 128,
        prototype_count: int = 500,
        temperature: float = 0.1,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_iterations: int = 3,
    ) -> None:
        super().__init__()

        if prototype_count < 2:
            raise ValueError(
                "prototype_count must be "
                "at least two."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        if sinkhorn_epsilon <= 0:
            raise ValueError(
                "sinkhorn_epsilon must be positive."
            )

        if sinkhorn_iterations < 1:
            raise ValueError(
                "sinkhorn_iterations must be "
                "at least one."
            )

        built = build_backbone(
            backbone_name,
            initialization=initialization,
        )

        self.backbone = built.model
        self.backbone_name = built.name
        self.initialization = (
            built.initialization
        )
        self.feature_dimension = (
            built.feature_dimension
        )

        self.projection_hidden_dimension = int(
            projection_hidden_dimension
        )

        self.projection_dimension = int(
            projection_dimension
        )

        self.prototype_count = int(
            prototype_count
        )

        self.temperature = float(
            temperature
        )

        self.sinkhorn_epsilon = float(
            sinkhorn_epsilon
        )

        self.sinkhorn_iterations = int(
            sinkhorn_iterations
        )

        self.projector = SwAVProjectionHead(
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

        self.prototypes = nn.Linear(
            self.projection_dimension,
            self.prototype_count,
            bias=False,
        )

    def encode(
        self,
        images: Tensor,
    ) -> Tensor:
        """Return comparable backbone features."""

        features = self.backbone(
            images
        )

        if features.ndim > 2:
            features = torch.flatten(
                features,
                start_dim=1,
            )

        if features.ndim != 2:
            raise RuntimeError(
                "Backbone output could not "
                "be reduced to "
                "(batch, feature_dimension)."
            )

        if (
            features.shape[1]
            != self.feature_dimension
        ):
            raise RuntimeError(
                "Backbone feature dimension "
                "mismatch: "
                f"{features.shape[1]} != "
                f"{self.feature_dimension}."
            )

        return features

    def project(
        self,
        features: Tensor,
    ) -> Tensor:
        """Return L2-normalised projection coordinates."""

        return functional.normalize(
            self.projector(features),
            dim=1,
        )

    def prototype_scores(
        self,
        projections: Tensor,
    ) -> Tensor:
        """Return similarities to prototype vectors."""

        if projections.ndim != 2:
            raise ValueError(
                "Prototype input must have shape "
                "(batch, projection_dimension)."
            )

        if (
            projections.shape[1]
            != self.projection_dimension
        ):
            raise ValueError(
                "Prototype input dimension "
                "mismatch: "
                f"{projections.shape[1]} != "
                f"{self.projection_dimension}."
            )

        return self.prototypes(
            projections
        )

    @torch.no_grad()
    def normalise_prototypes(
        self,
    ) -> None:
        """Constrain prototype vectors to unit norm."""

        self.prototypes.weight.copy_(
            functional.normalize(
                self.prototypes.weight,
                dim=1,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return features, projections, and scores."""

        features = self.encode(
            images
        )

        projections = self.project(
            features
        )

        scores = self.prototype_scores(
            projections
        )

        return (
            features,
            projections,
            scores,
        )

    def training_loss(
        self,
        views: tuple[Tensor, ...],
    ) -> Tensor:
        """Return the scalar two-view SwAV loss."""

        loss, _ = self.swav_objective(
            views
        )

        return loss

    def swav_objective(
        self,
        views: Sequence[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Evaluate swapped prediction for two global crops."""

        if len(views) != 2:
            raise ValueError(
                "The initial SwAV baseline "
                "requires exactly two global views."
            )

        first_view = views[0]
        second_view = views[1]

        if (
            first_view.shape
            != second_view.shape
        ):
            raise ValueError(
                "Global view tensors must "
                "have identical shapes."
            )

        self.normalise_prototypes()

        (
            first_features,
            first_projections,
            first_scores,
        ) = self(first_view)

        (
            second_features,
            second_projections,
            second_scores,
        ) = self(second_view)

        (
            loss,
            first_assignments,
            second_assignments,
        ) = swapped_prediction_loss(
            first_scores,
            second_scores,
            temperature=self.temperature,
            sinkhorn_epsilon=(
                self.sinkhorn_epsilon
            ),
            sinkhorn_iterations=(
                self.sinkhorn_iterations
            ),
        )

        combined_assignments = torch.cat(
            [
                first_assignments,
                second_assignments,
            ],
            dim=0,
        )

        hard_assignments = (
            combined_assignments.argmax(
                dim=1
            )
        )

        occupied_prototypes = (
            torch.unique(
                hard_assignments
            ).numel()
        )

        occupancy = torch.bincount(
            hard_assignments,
            minlength=self.prototype_count,
        ).to(
            torch.float32
        )

        occupancy_probabilities = (
            occupancy
            / occupancy.sum()
        )

        nonzero_probabilities = (
            occupancy_probabilities[
                occupancy_probabilities > 0
            ]
        )

        floor = torch.finfo(
            combined_assignments.dtype
        ).eps

        assignment_entropy = -(
            combined_assignments
            * combined_assignments
            .clamp_min(floor)
            .log()
        ).sum(
            dim=1
        ).mean()

        normalised_occupancy_entropy = -(
            nonzero_probabilities
            * nonzero_probabilities.log()
        ).sum() / math.log(
            self.prototype_count
        )

        diagnostics = {
            "assignment_entropy": (
                assignment_entropy.detach()
            ),
            "normalised_occupancy_entropy": (
                normalised_occupancy_entropy
                .detach()
            ),
            "occupied_prototype_fraction": (
                torch.tensor(
                    occupied_prototypes
                    / self.prototype_count,
                    device=loss.device,
                    dtype=loss.dtype,
                )
            ),
            "largest_prototype_fraction": (
                occupancy_probabilities
                .max()
                .to(
                    device=loss.device,
                    dtype=loss.dtype,
                )
            ),
            "assignment_agreement": (
                first_assignments.argmax(
                    dim=1
                )
                == second_assignments.argmax(
                    dim=1
                )
            ).to(
                torch.float32
            ).mean(),
            "backbone_feature_norm": (
                0.5
                * (
                    first_features
                    .detach()
                    .norm(dim=1)
                    .mean()
                    + second_features
                    .detach()
                    .norm(dim=1)
                    .mean()
                )
            ),
            "projection_norm": (
                0.5
                * (
                    first_projections
                    .detach()
                    .norm(dim=1)
                    .mean()
                    + second_projections
                    .detach()
                    .norm(dim=1)
                    .mean()
                )
            ),
            "prototype_weight_norm": (
                self.prototypes
                .weight
                .detach()
                .norm(dim=1)
                .mean()
            ),
        }

        return (
            loss,
            diagnostics,
        )

    def model_metadata(
        self,
    ) -> dict[str, object]:
        """Return architecture provenance metadata."""

        return {
            "method": "swav",
            "backbone": self.backbone_name,
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
            "prototype_count": (
                self.prototype_count
            ),
            "temperature": (
                self.temperature
            ),
            "sinkhorn_epsilon": (
                self.sinkhorn_epsilon
            ),
            "sinkhorn_iterations": (
                self.sinkhorn_iterations
            ),
            "assignment_crops": 2,
            "local_crops": 0,
        }
