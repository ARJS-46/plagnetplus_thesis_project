"""Representation models."""

from plagnetplus.models.backbones import (
    BackboneBuild,
    BackboneName,
    build_backbone,
    normalise_backbone_name,
)
from plagnetplus.models.base import (
    RepresentationModel,
)
from plagnetplus.models.deepcluster import (
    DeepClusterModel,
)
from plagnetplus.models.simclr import (
    ProjectionHead,
    SimCLRModel,
    nt_xent_loss,
)

from plagnetplus.models.simsiam import (
    SimSiamModel,
    SimSiamPredictionHead,
    SimSiamProjectionHead,
    negative_cosine_similarity,
    symmetric_negative_cosine_loss,
)

from plagnetplus.models.swav import (
    SwAVModel,
    SwAVProjectionHead,
    sinkhorn_knopp,
    swapped_prediction_loss,
)

__all__ = [
    "BackboneBuild",
    "BackboneName",
    "DeepClusterModel",
    "ProjectionHead",
    "RepresentationModel",
    "SimCLRModel",
    "build_backbone",
    "normalise_backbone_name",
    "nt_xent_loss",
    "SimSiamModel",
    "SimSiamPredictionHead",
    "SimSiamProjectionHead",
    "negative_cosine_similarity",
    "symmetric_negative_cosine_loss",
    "SwAVModel",
    "SwAVProjectionHead",
    "sinkhorn_knopp",
    "swapped_prediction_loss",
]
