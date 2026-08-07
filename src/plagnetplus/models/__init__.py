"""Shared components for PlagNetPlus representation models."""

from plagnetplus.models.backbones import (
    BackboneBuild,
    BackboneName,
    build_backbone,
    normalise_backbone_name,
)
from plagnetplus.models.base import RepresentationModel


__all__ = [
    "BackboneBuild",
    "BackboneName",
    "RepresentationModel",
    "build_backbone",
    "normalise_backbone_name",
]
