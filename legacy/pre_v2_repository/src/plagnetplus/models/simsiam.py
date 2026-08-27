"""Native SimSiam model and explicit stop-gradient objective."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from plagnetplus.models.backbones import build_backbone
from plagnetplus.models.base import RepresentationModel


class SimSiamProjectionHead(nn.Module):
    """Method-faithful three-layer SimSiam projection MLP."""

    def __init__(
        self,
        *,
        input_dimension: int,
        hidden_dimension: int = 2048,
        output_dimension: int = 2048,
        final_batch_norm_affine: bool = False,
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

        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.output_dimension = int(output_dimension)

        self.network = nn.Sequential(
            nn.Linear(
                self.input_dimension,
                self.hidden_dimension,
                bias=False,
            ),
            nn.BatchNorm1d(
                self.hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.hidden_dimension,
                self.hidden_dimension,
                bias=False,
            ),
            nn.BatchNorm1d(
                self.hidden_dimension,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.hidden_dimension,
                self.output_dimension,
                bias=False,
            ),
            nn.BatchNorm1d(
                self.output_dimension,
                affine=final_batch_norm_affine,
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """Project backbone features into the SimSiam target space."""

        if features.ndim != 2:
            raise ValueError(
                "Projection-head input must have shape "
                "(batch, feature_dimension)."
            )

        if features.shape[1] != self.input_dimension:
            raise ValueError(
                "Projection-head feature dimension mismatch: "
                f"{features.shape[1]} != {self.input_dimension}."
            )

        return self.network(features)


class SimSiamPredictionHead(nn.Module):
    """Two-layer bottleneck predictor used only during training."""

    def __init__(
        self,
        *,
        input_dimension: int = 2048,
        hidden_dimension: int = 512,
        output_dimension: int = 2048,
    ) -> None:
        super().__init__()

        if min(
            input_dimension,
            hidden_dimension,
            output_dimension,
        ) <= 0:
            raise ValueError(
                "Prediction dimensions must be positive."
            )

        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.output_dimension = int(output_dimension)

        self.network = nn.Sequential(
            nn.Linear(
                self.input_dimension,
                self.hidden_dimension,
                bias=False,
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
        projections: Tensor,
    ) -> Tensor:
        """Predict the detached target representation of another view."""

        if projections.ndim != 2:
            raise ValueError(
                "Prediction-head input must have shape "
                "(batch, projection_dimension)."
            )

        if projections.shape[1] != self.input_dimension:
            raise ValueError(
                "Prediction-head feature dimension mismatch: "
                f"{projections.shape[1]} != "
                f"{self.input_dimension}."
            )

        return self.network(projections)


def negative_cosine_similarity(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Return negative cosine similarity with a detached target."""

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target tensors must have identical shapes."
        )

    if prediction.ndim != 2:
        raise ValueError(
            "Prediction and target tensors must have shape "
            "(batch, projection_dimension)."
        )

    if prediction.shape[0] < 2:
        raise ValueError(
            "SimSiam requires at least two items per batch."
        )

    normalised_prediction = functional.normalize(
        prediction,
        dim=1,
    )

    normalised_target = functional.normalize(
        target.detach(),
        dim=1,
    )

    return -(
        normalised_prediction
        * normalised_target
    ).sum(dim=1).mean()


def symmetric_negative_cosine_loss(
    first_prediction: Tensor,
    second_projection: Tensor,
    second_prediction: Tensor,
    first_projection: Tensor,
) -> Tensor:
    """Return the symmetric SimSiam objective with stop-gradient targets."""

    first_term = negative_cosine_similarity(
        first_prediction,
        second_projection,
    )

    second_term = negative_cosine_similarity(
        second_prediction,
        first_projection,
    )

    return 0.5 * (
        first_term
        + second_term
    )


class SimSiamModel(RepresentationModel):
    """SimSiam with a configurable PlagNetPlus visual backbone."""

    def __init__(
        self,
        *,
        backbone_name: str,
        initialization: str = "scratch",
        projection_hidden_dimension: int = 2048,
        projection_dimension: int = 2048,
        prediction_hidden_dimension: int = 512,
        prediction_dimension: int | None = None,
        final_projection_batch_norm_affine: bool = False,
    ) -> None:
        super().__init__()

        built = build_backbone(
            backbone_name,
            initialization=initialization,
        )

        resolved_prediction_dimension = (
            projection_dimension
            if prediction_dimension is None
            else int(prediction_dimension)
        )

        if (
            resolved_prediction_dimension
            != projection_dimension
        ):
            raise ValueError(
                "The SimSiam prediction output dimension must "
                "match the projection dimension."
            )

        self.backbone = built.model
        self.backbone_name = built.name
        self.initialization = built.initialization
        self.feature_dimension = built.feature_dimension

        self.projection_hidden_dimension = int(
            projection_hidden_dimension
        )

        self.projection_dimension = int(
            projection_dimension
        )

        self.prediction_hidden_dimension = int(
            prediction_hidden_dimension
        )

        self.prediction_dimension = int(
            resolved_prediction_dimension
        )

        self.projector = SimSiamProjectionHead(
            input_dimension=self.feature_dimension,
            hidden_dimension=self.projection_hidden_dimension,
            output_dimension=self.projection_dimension,
            final_batch_norm_affine=(
                final_projection_batch_norm_affine
            ),
        )

        self.predictor = SimSiamPredictionHead(
            input_dimension=self.projection_dimension,
            hidden_dimension=self.prediction_hidden_dimension,
            output_dimension=self.prediction_dimension,
        )

    def encode(
        self,
        images: Tensor,
    ) -> Tensor:
        """Return backbone features used in downstream comparisons."""

        features = self.backbone(images)

        if features.ndim > 2:
            features = torch.flatten(
                features,
                start_dim=1,
            )

        if features.ndim != 2:
            raise RuntimeError(
                "Backbone output could not be reduced to "
                "(batch, feature_dimension)."
            )

        if features.shape[1] != self.feature_dimension:
            raise RuntimeError(
                "Backbone feature dimension mismatch: "
                f"{features.shape[1]} != "
                f"{self.feature_dimension}."
            )

        return features

    def project(
        self,
        features: Tensor,
    ) -> Tensor:
        """Return projector coordinates used by the SimSiam objective."""

        return self.projector(features)

    def predict(
        self,
        projections: Tensor,
    ) -> Tensor:
        """Return predictor coordinates used by the SimSiam objective."""

        return self.predictor(projections)

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return backbone features, projections and predictions."""

        features = self.encode(images)
        projections = self.project(features)
        predictions = self.predict(projections)

        return (
            features,
            projections,
            predictions,
        )

    def training_loss(
        self,
        batch: Sequence[Tensor],
    ) -> Tensor:
        """Return the scalar SimSiam loss required by RepresentationModel."""

        loss, _ = self.simsiam_objective(
            batch
        )

        return loss

    def simsiam_objective(
        self,
        views: Sequence[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Evaluate the symmetric loss and collapse diagnostics."""

        if len(views) != 2:
            raise ValueError(
                "SimSiam requires exactly two augmented views."
            )

        first_view = views[0]
        second_view = views[1]

        if first_view.shape != second_view.shape:
            raise ValueError(
                "Augmented view tensors must have identical shapes."
            )

        (
            first_features,
            first_projection,
            first_prediction,
        ) = self(first_view)

        (
            second_features,
            second_projection,
            second_prediction,
        ) = self(second_view)

        loss = symmetric_negative_cosine_loss(
            first_prediction,
            second_projection,
            second_prediction,
            first_projection,
        )

        first_normalised_projection = (
            functional.normalize(
                first_projection.detach(),
                dim=1,
            )
        )

        second_normalised_projection = (
            functional.normalize(
                second_projection.detach(),
                dim=1,
            )
        )

        projection_standard_deviation = 0.5 * (
            first_normalised_projection.std(
                dim=0,
            ).mean()
            + second_normalised_projection.std(
                dim=0,
            ).mean()
        )

        collapse_level = torch.clamp(
            1.0
            - math.sqrt(
                self.projection_dimension
            )
            * projection_standard_deviation,
            min=0.0,
            max=1.0,
        )

        diagnostics = {
            "mean_cosine_similarity": (
                -loss.detach()
            ),
            "projection_standard_deviation": (
                projection_standard_deviation
            ),
            "collapse_level": collapse_level,
            "backbone_feature_norm": 0.5
            * (
                first_features.detach().norm(
                    dim=1,
                ).mean()
                + second_features.detach().norm(
                    dim=1,
                ).mean()
            ),
            "projection_norm": 0.5
            * (
                first_projection.detach().norm(
                    dim=1,
                ).mean()
                + second_projection.detach().norm(
                    dim=1,
                ).mean()
            ),
            "prediction_norm": 0.5
            * (
                first_prediction.detach().norm(
                    dim=1,
                ).mean()
                + second_prediction.detach().norm(
                    dim=1,
                ).mean()
            ),
        }

        return loss, diagnostics


__all__ = [
    "SimSiamModel",
    "SimSiamPredictionHead",
    "SimSiamProjectionHead",
    "negative_cosine_similarity",
    "symmetric_negative_cosine_loss",
]
