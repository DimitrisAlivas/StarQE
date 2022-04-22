"""Similarity functions."""
from abc import abstractmethod
from typing import Union

import torch
from class_resolver import Resolver
from torch import nn
from torch.nn import functional


class Similarity(nn.Module):
    """Base class for similarity functions."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pair-wise similarities.

        :param x: shape: (*, n, d)
            The first vectors.
        :param y: shape: (*, m, d)
            The second vectors.

        :return: shape: (*, n, m)
            The similarity values.
        """
        raise NotImplementedError

    def one_to_one(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute batched one-to-one similarities.

        :param x: shape: (*, d)
            The first vectors.
        :param y: shape: (*, d)
            The second vectors.

        :return: shape: (*)
            The similarity values.
        """
        return self(x.unsqueeze(dim=-2), y.unsqueeze(dim=-2)).squeeze(dim=-1).squeeze(dim=-1)


class NegativeLpSimilarity(Similarity):
    """Negative l_p distance similarity."""

    def __init__(self, p: float = 2.0):
        """
        Initialize the similarity.

        :param p:
            The parameter p for the l_p distance. See also: torch.cdist
        """
        super().__init__()
        self.p = p

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return -torch.cdist(x, y, p=self.p)


class NegativePowerNormSimilarity(Similarity):
    r"""Negative power norm: -\|x - y\|_p^p."""

    def __init__(self, p: Union[int, float] = 2):
        """
        Initialize the similarity.

        :param p:
            The parameter p for the p-norm.
        """
        super().__init__()
        self.p = p

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return -(x.unsqueeze(dim=-2) - y.unsqueeze(dim=-3)).pow(self.p).sum(dim=-1)


class DotProductSimilarity(Similarity):
    """Dot product similarity."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return x @ y.transpose(-2, -1)


class CosineSimilarity(DotProductSimilarity):
    """Cosine similarity."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        x = functional.normalize(x, p=2, dim=-1)
        y = functional.normalize(y, p=2, dim=-1)
        return super().forward(x=x, y=y)


similarity_resolver: Resolver[Similarity] = Resolver.from_subclasses(
    base=Similarity,  # type: ignore
    default=DotProductSimilarity,
)
