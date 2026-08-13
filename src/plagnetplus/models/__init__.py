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
]
