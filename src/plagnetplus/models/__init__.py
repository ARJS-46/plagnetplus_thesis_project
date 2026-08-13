"""Shared components for PlagNetPlus representation models."""

from plagnetplus.models.backbones import (
    BackboneBuild,
    BackboneName,
    build_backbone,
    normalise_backbone_name,
)
from plagnetplus.models.base import RepresentationModel
from plagnetplus.models.deepcluster import DeepClusterModel
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


__all__ = [
    "BackboneBuild",
    "BackboneName",
    "DeepClusterModel",
    "ProjectionHead",
    "RepresentationModel",
    "SimCLRModel",
    "SimSiamModel",
    "SimSiamPredictionHead",
    "SimSiamProjectionHead",
    "build_backbone",
    "negative_cosine_similarity",
    "normalise_backbone_name",
    "nt_xent_loss",
    "symmetric_negative_cosine_loss",
]
