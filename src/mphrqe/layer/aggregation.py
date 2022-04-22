"""Qualifier pair aggregation per edge."""
from abc import abstractmethod
from typing import Any, Mapping, Optional

import torch
import torch_scatter
from class_resolver import HintOrType, Resolver
from torch import nn

from .util import activation_resolver, get_parameter, softmax
from ..typing import FloatTensor

__all__ = [
    "qualifier_aggregation_resolver",
    "QualifierAggregation",
]


def coalesce_qualifiers(
    x_q: torch.FloatTensor,
    edge_ids: torch.LongTensor,
    num_edges: Optional[int] = None,
    fill: float = 0.0,
) -> torch.FloatTensor:
    """
    Aggregate qualifier pair representations.

    :param x_q: (num_qualifier_pairs, dim)
        The qualifier pair representations.
    :param edge_ids: shape: (num_qualifier_pairs,)
        The ID of the corresponding edges for each qualifier pair.
    :param num_edges:
        The total number of edges. If None, is inferred as max(edge_ids) + 1.
    :param fill:
        The fill value to use for edges where no qualifier pair is present.

    :return: shape: (num_edges, dim)
        The aggregated qualifier pair representation per edge.
    """
    x = torch_scatter.scatter_add(src=x_q, index=edge_ids, dim=0, dim_size=num_edges)
    # index_add is somewhat slower for large number of edges
    # num_edges = num_edges or edge_ids.max() + 1
    # x = x_q.new_zeros(size=(num_edges, x_q.shape[-1])).index_add(
    #     dim=0,
    #     index=edge_ids,
    #     source=x_q,
    # )
    if fill != 0.0:
        x[(x == 0).all(dim=-1)] = fill
    return x


class QualifierAggregation(nn.Module):
    r"""
    An aggregator for qualifier pairs.

    .. math ::
        y_i = r_i + W Aggregate({q_ij}_j)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize the aggregator.

        :param output_dim:
            The representation dimension.
        """
        super().__init__()
        output_dim = output_dim or input_dim
        self.w_q = get_parameter(input_dim, output_dim)

    @property
    def input_dim(self) -> int:
        """Return the input dimension."""
        return self.w_q.shape[0]

    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self.w_q.shape[1]

    @abstractmethod
    def forward(
        self,
        x_q: torch.FloatTensor,
        x_edge: torch.FloatTensor,
        edge_ids: torch.LongTensor,
    ) -> FloatTensor:  # noqa: D102
        """
        Aggregate qualifier pair representations onto a relation representation.

        :param x_q: shape: (num_qualifier_pairs, dim)
            The qualifier pair representations.
        :param x_edge: shape: (num_edges, dim)
            The edge representations (equal to the relation representation corresponding to the edge).
        :param edge_ids: shape: (num_qualifier_pairs,)
            The edge Ids from the qualifier index, an integer indicating the mapping of a qualifier pair to the
            corresponding edge.

        :return: shape: (num_edges, dim)
            The updated edge representations.
        """
        raise NotImplementedError


class SumQualifierAggregation(QualifierAggregation):
    r"""
    Aggregation by sum.

    .. math ::
        y_i = \alpha r_i + (1 - \alpha) W \sum_j q_ij
    """

    def __init__(
        self,
        alpha: float = 0.5,
        **kwargs,
    ):
        """
        Initialize the aggregator.

        :param alpha:
            A skip connection weight.
        :param kwargs:
            Additional keyword-based arguments passed to QualifierAggregation.__init__
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def forward(
        self,
        x_q: torch.FloatTensor,
        x_edge: torch.FloatTensor,
        edge_ids: torch.LongTensor,
    ) -> FloatTensor:  # noqa: D102
        return self.alpha * x_edge + (1 - self.alpha) * torch.einsum(
            'ij,jk -> ik',
            coalesce_qualifiers(x_q=x_q, edge_ids=edge_ids, num_edges=x_edge.shape[0]),
            self.w_q,
        )  # [N_EDGES / 2 x EMB_DIM]


class ConcatQualifierAggregation(QualifierAggregation):
    r"""
    Aggregation by concatenation.

    .. math ::
        y_i = W [r_i; \sum_j q_ij]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
    ):
        output_dim = output_dim or input_dim
        super().__init__(input_dim=2 * input_dim, output_dim=output_dim)

    def forward(
        self,
        x_q: torch.FloatTensor,
        x_edge: torch.FloatTensor,
        edge_ids: torch.LongTensor,
    ) -> FloatTensor:  # noqa: D102
        return torch.cat(
            (x_edge, coalesce_qualifiers(x_q=x_q, edge_ids=edge_ids, num_edges=x_edge.shape[0])),
            dim=1,
        ) @ self.w_q


class MulQualifierAggregation(QualifierAggregation):
    r"""
    Aggregation by elementwise multiplication.

    .. math ::
        y_i = r_i \odot W \sum_j q_ij
    """

    def forward(
        self,
        x_q: torch.FloatTensor,
        x_edge: torch.FloatTensor,
        edge_ids: torch.LongTensor,
    ) -> FloatTensor:  # noqa: D102
        return x_edge * (
            coalesce_qualifiers(x_q=x_q, edge_ids=edge_ids, num_edges=x_edge.shape[0], fill=1) @ self.w_q
        )


class AttentionQualifierAggregation(QualifierAggregation):
    """Aggregation by attention."""

    def __init__(
        self,
        num_heads: int = 2,
        activation: HintOrType[nn.Module] = nn.LeakyReLU,
        activation_kwargs: Optional[Mapping[str, Any]] = None,
        attention_drop: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the aggregation.

        :param num_heads:
            The number of attention heads.
        :param activation:
            The activation function.
        :param activation_kwargs:
            Additional keyword based arguments passed to the activation function upon construction.
        :param attention_drop:
            A dropout value for attention.
        :param kwargs:
            Additional keyword based arguments passed to QualifierAggregation.
        """
        super().__init__(**kwargs)
        if self.input_dim % num_heads != 0:
            raise ValueError(f"output_dim={self.input_dim} must be divisible by num_heads={num_heads}!")
        self.num_heads = num_heads
        self.activation = activation_resolver.make(activation, pos_kwargs=activation_kwargs)
        self.dropout = nn.Dropout(attention_drop or 0.0)
        self.weight = get_parameter(self.num_heads, 2 * self.input_dim // self.num_heads)

    def forward(
        self,
        x_q: FloatTensor,
        x_edge: torch.FloatTensor,
        edge_ids: torch.LongTensor,
    ) -> FloatTensor:  # noqa: D102
        # includes inverse relations!
        num_qualifier_pairs = x_q.shape[0]
        # compute attention scores per qualifier pair, shape: (num_qualifiers, num_heads)
        x_q = (x_q @ self.w_q).view(num_qualifier_pairs, self.num_heads, -1)
        alpha = torch.einsum(
            "ehd,hd->eh",
            torch.cat([
                x_edge[edge_ids].view(num_qualifier_pairs, self.num_heads, -1),
                x_q,
            ], dim=-1),
            self.weight,
        )
        alpha = self.activation(alpha)
        alpha = softmax(src=alpha, index=edge_ids, num_nodes=x_edge.shape[0], dim=0)
        alpha = self.dropout(alpha)
        # weight messages
        x_q = alpha.unsqueeze(dim=-1) * x_q
        # TODO: why is this non-contiguous?
        x_q = x_q.reshape(num_qualifier_pairs, -1)
        # aggregate
        # index_add is somewhat slower for large number of edges
        # return torch.zeros_like(x_edge).index_add(dim=0, index=edge_ids, source=x_q)
        return torch_scatter.scatter_add(src=x_q, index=edge_ids, dim=0, dim_size=x_edge.shape[0])


qualifier_aggregation_resolver = Resolver.from_subclasses(
    base=QualifierAggregation,  # type: ignore
    default=SumQualifierAggregation,
)
