"""Backbone construction for PlagNetPlus representation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn
from torchvision.models import resnet18, resnet50


# Restrict the supported identifiers to architectures that are used by
# the current representation-learning experiments.
BackboneName = Literal[
    "resnet18",
    "resnet50",
]


@dataclass(frozen=True)
class BackboneBuild:
    """Store a constructed encoder together with its representation metadata."""

    model: nn.Module
    name: BackboneName
    feature_dimension: int
    initialization: str


def normalise_backbone_name(name: str) -> BackboneName:
    """Convert common ResNet identifiers to the package's canonical names."""

    # Remove separators so forms such as "resnet-18", "resnet_18" and
    # "rn18" can be handled consistently by the same alias table.
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
            sorted(
                {
                    "resnet18",
                    "resnet50",
                }
            )
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
    """Build a ResNet encoder with its classification layer removed."""

    selected = normalise_backbone_name(name)

    # All baseline comparisons use randomly initialized networks. Keeping
    # this restriction explicit prevents pretrained weights being introduced
    # accidentally into an otherwise controlled comparison.
    if initialization != "scratch":
        raise ValueError(
            "The controlled SSL comparison currently supports "
            "only initialization='scratch'."
        )

    # Torchvision's classification head is replaced below so that each
    # network returns its final backbone representation instead of logits.
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

    # Capture the dimensionality before replacing the fully connected
    # classification layer. ResNet-18 returns 512 features and ResNet-50
    # returns 2048 features at this point in the network.
    feature_dimension = int(backbone.fc.in_features)
    backbone.fc = nn.Identity()

    return BackboneBuild(
        model=backbone,
        name=selected,
        feature_dimension=feature_dimension,
        initialization=initialization,
    )
