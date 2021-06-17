"""Message weighting methods."""
from typing import Any, Mapping, Optional, Tuple

import torch
from class_resolver import Hint, Resolver
from torch import nn

from .util import activation_resolver, get_parameter, softmax
from ..typing import FloatTensor

__all__ = [
    "MessageWeighting",
    "message_weighting_resolver",
]


class MessageWeighting(nn.Module):
    """Base class for message weighting."""

    def forward(
        self,
        edge_index: torch.LongTensor,
        message: torch.FloatTensor,
        x_e: torch.FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:  # noqa: D102
        """
        :param edge_index: shape: (2, num_edges)
            The edge index, (source, target) entity ID pairs.
        :param message: shape: (num_edges, dim)
            The messages.
        :param x_e: shape: (num_nodes, dim)
            The node representations.

        :return:
            messages: shape: (num_edges, *)
                The messages. The final dimension(s) may have been reshaped.
            weights: shape: messages.shape[:-1]
                A scalar weight for each edge.
        """
        raise NotImplementedError


class SymmetricMessageWeighting(MessageWeighting):
    r"""
    Static symmetric edge weighting.

    .. math ::
        \hat{A} = A + I

        \hat{D} = \sum_i \hat{A}_{ii}

        \hat{A}_n = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}
    """

    def __init__(self, directed: bool = False):
        super().__init__()
        self.directed = directed

    def forward(
        self,
        edge_index: torch.LongTensor,
        message: torch.FloatTensor,
        x_e: torch.FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:  # noqa: D102
        if self.directed:
            edge_weight = torch.ones_like(edge_index[0], dtype=torch.get_default_dtype())
            for index in edge_index:
                uniq, inverse_idx, count = index.unique(return_counts=True, return_inverse=True)
                deg_sqrt_inv = count.float().pow(-0.5)
                edge_weight[inverse_idx] *= deg_sqrt_inv
        else:
            uniq, inverse_idx, count = edge_index.unique(return_counts=True, return_inverse=True)
            edge_weight = count[inverse_idx].sum(dim=0).float().pow(-0.5)
        return message, edge_weight


class AttentionMessageWeighting(MessageWeighting):
    """Message weighting by attention."""

    def __init__(
        self,
        output_dim: int,
        num_heads: int = 8,
        activation: Hint[nn.Module] = nn.LeakyReLU,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(f"output_dim={output_dim} must be divisible by num_heads={num_heads}!")
        self.num_heads = num_heads
        self.weight = get_parameter(num_heads, 2 * output_dim // num_heads)
        self.activation = activation_resolver.make(activation, pos_kwargs=activation_kwargs)

    def forward(
        self,
        edge_index: torch.LongTensor,
        message: torch.FloatTensor,
        x_e: torch.FloatTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:  # noqa: D102
        source, target = edge_index
        message_ = message.view(message.shape[0], self.num_heads, -1)
        # Compute attention coefficients, shape: (num_edges, num_heads)
        alpha = self.activation(torch.einsum(
            "ihd,hd->ih",
            torch.cat([
                message_,
                x_e[target].view(target.shape[0], self.num_heads, -1),
            ], dim=-1),
            self.weight,
        ))
        alpha = softmax(alpha, index=target, num_nodes=x_e.shape[0], dim=0)
        return message_, alpha


message_weighting_resolver = Resolver.from_subclasses(
    base=MessageWeighting,
    default=SymmetricMessageWeighting,
)
