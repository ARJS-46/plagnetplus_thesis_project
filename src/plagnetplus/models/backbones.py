"""Backbone construction for PlagNetPlus representation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn
from torchvision.models import resnet18, resnet50


BackboneName = Literal[
    "resnet18",
    "resnet50",
]


@dataclass(frozen=True)
class BackboneBuild:
    """A constructed encoder and its representation metadata."""

    model: nn.Module
    name: BackboneName
    feature_dimension: int
    initialization: str


def normalise_backbone_name(
    name: str,
) -> BackboneName:
    """Normalize a user-facing backbone identifier."""

    normalized = (
        str(name)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    aliases: dict[str, BackboneName] = {
        "resnet18": "resnet18",
        "rn18": "resnet18",
        "resnet50": "resnet50",
        "rn50": "resnet50",
    }

    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(
            sorted({
                "resnet18",
                "resnet50",
            })
        )

        raise ValueError(
            f"Unsupported backbone {name!r}. "
            f"Supported backbones: {supported}."
        ) from exc


def build_backbone(
    name: str,
    *,
    initialization: Literal["scratch"] = "scratch",
) -> BackboneBuild:
    """Build a classification-head-free image encoder."""

    selected = normalise_backbone_name(
        name
    )

    if initialization != "scratch":
        raise ValueError(
            "The controlled SSL comparison currently supports "
            "only initialization='scratch'."
        )

    if selected == "resnet18":
        backbone = resnet18(
            weights=None,
            zero_init_residual=True,
        )
    else:
        backbone = resnet50(
            weights=None,
            zero_init_residual=True,
        )

    feature_dimension = int(
        backbone.fc.in_features
    )

    backbone.fc = nn.Identity()

    return BackboneBuild(
        model=backbone,
        name=selected,
        feature_dimension=feature_dimension,
        initialization=initialization,
    )
