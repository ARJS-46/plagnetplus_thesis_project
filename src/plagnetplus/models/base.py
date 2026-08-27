from abc import ABC, abstractmethod
from torch import Tensor, nn

class RepresentationModel(nn.Module, ABC):
    @abstractmethod
    def encode(self, images: Tensor) -> Tensor:
        """Return one comparable backbone embedding per image."""

    @abstractmethod
    def training_loss(self, views: tuple[Tensor, ...]) -> Tensor:
        """Return the model-specific SSL loss."""
