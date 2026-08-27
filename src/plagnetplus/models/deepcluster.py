"""DeepCluster representation model and pseudo-label classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from plagnetplus.models.backbones import build_backbone
from plagnetplus.models.base import RepresentationModel


class DeepClusterModel(RepresentationModel):
    """Backbone encoder with a replaceable pseudo-label classifier."""

    def __init__(
        self,
        *,
        backbone_name: str,
        cluster_count: int,
        initialization: str = "scratch",
    ) -> None:
        super().__init__()

        if cluster_count < 2:
            raise ValueError(
                "cluster_count must be at least two."
            )

        built = build_backbone(
            backbone_name,
            initialization=initialization,
        )

        self.backbone = built.model
        self.backbone_name = built.name
        self.initialization = built.initialization
        self.feature_dimension = built.feature_dimension
        self.cluster_count = int(cluster_count)

        self.classifier = nn.Linear(
            self.feature_dimension,
            self.cluster_count,
        )

        self.reset_classifier()

    def encode(
        self,
        images: Tensor,
    ) -> Tensor:
        """Return one backbone representation per image."""
        return self.backbone(images)

    def classify(
        self,
        features: Tensor,
    ) -> Tensor:
        """Return logits for the current pseudo-label classes."""
        if features.ndim != 2:
            raise ValueError(
                "features must be two-dimensional."
            )

        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                "Unexpected feature dimension: "
                f"{features.shape[1]} != "
                f"{self.feature_dimension}."
            )

        return self.classifier(features)

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return backbone features and pseudo-label logits."""
        features = self.encode(images)
        logits = self.classify(features)

        return features, logits

    def pseudo_label_loss(
        self,
        images: Tensor,
        pseudo_labels: Tensor,
    ) -> Tensor:
        """Calculate cross-entropy against current pseudo-labels."""
        if pseudo_labels.ndim != 1:
            raise ValueError(
                "pseudo_labels must be one-dimensional."
            )

        if images.shape[0] != pseudo_labels.shape[0]:
            raise ValueError(
                "Image and pseudo-label batch sizes differ."
            )

        _, logits = self(images)

        labels = pseudo_labels.to(
            device=logits.device,
            dtype=torch.long,
        )

        if torch.any(labels < 0):
            raise ValueError(
                "pseudo_labels contain negative values."
            )

        if torch.any(
            labels >= self.cluster_count
        ):
            raise ValueError(
                "pseudo_labels exceed cluster_count."
            )

        # Treat the current k-means assignments as temporary labels for this
        # training cycle; they are regenerated after the next feature extraction.
        return F.cross_entropy(
            logits,
            labels,
        )

    def training_loss(
        self,
        views: tuple[Tensor, ...],
    ) -> Tensor:
        """Satisfy the shared representation-model interface.

        DeepCluster passes ``(images, pseudo_labels)`` rather than
        multiple augmented views.
        """
        if len(views) != 2:
            raise ValueError(
                "DeepCluster training_loss expects "
                "(images, pseudo_labels)."
            )

        images, pseudo_labels = views

        return self.pseudo_label_loss(
            images,
            pseudo_labels,
        )

    def reset_classifier(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        """Replace and reinitialise the pseudo-label classifier."""
        # Reset the classifier after reclustering because numeric cluster IDs have
        # no persistent semantic meaning between DeepCluster cycles.
        old_classifier = getattr(
            self,
            "classifier",
            None,
        )

        if old_classifier is not None:
            device = old_classifier.weight.device
            dtype = old_classifier.weight.dtype
        else:
            parameter = next(
                self.backbone.parameters()
            )
            device = parameter.device
            dtype = parameter.dtype

        cuda_devices: list[int] = []

        if device.type == "cuda":
            cuda_devices = [
                (
                    torch.cuda.current_device()
                    if device.index is None
                    else device.index
                )
            ]

        with torch.random.fork_rng(
            devices=cuda_devices
        ):
            if seed is not None:
                torch.manual_seed(seed)

                if device.type == "cuda":
                    torch.cuda.manual_seed_all(
                        seed
                    )

            classifier = nn.Linear(
                self.feature_dimension,
                self.cluster_count,
                device=device,
                dtype=dtype,
            )

            nn.init.normal_(
                classifier.weight,
                mean=0.0,
                std=0.01,
            )

            nn.init.zeros_(
                classifier.bias
            )

        self.classifier = classifier

    def model_metadata(
        self,
    ) -> dict[str, object]:
        """Return architecture information for provenance."""
        return {
            "method": "deepcluster",
            "backbone": self.backbone_name,
            "initialization": (
                self.initialization
            ),
            "feature_dimension": (
                self.feature_dimension
            ),
            "cluster_count": (
                self.cluster_count
            ),
            "classifier_reinitialised_each_cycle": (
                True
            ),
        }
