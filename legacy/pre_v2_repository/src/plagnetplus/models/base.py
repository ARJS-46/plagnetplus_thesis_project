"""Common interface for self-supervised representation models."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class RepresentationModel(nn.Module, ABC):
    """Define the operations shared by PlagNetPlus representation models."""

    @abstractmethod
    def encode(self, images: Tensor) -> Tensor:
        """Return one comparable backbone embedding for each input image."""

    @abstractmethod
    def training_loss(self, views: tuple[Tensor, ...]) -> Tensor:
        """Return the self-supervised training loss for a batch of views."""
