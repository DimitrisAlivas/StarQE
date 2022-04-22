# -*- coding: utf-8 -*-

"""Implementation of the StarE convolution layer."""
from typing import Any, Mapping, Optional, Tuple

import torch
import torch_scatter
from class_resolver import HintOrType
from torch import nn

from .aggregation import QualifierAggregation, qualifier_aggregation_resolver
from .composition import Composition, composition_resolver
from .util import activation_resolver, get_parameter
from .weighting import AttentionMessageWeighting, MessageWeighting, message_weighting_resolver
from ..typing import FloatTensor, LongTensor

__all__ = [
    "StarEConvLayer",
]


class StarEConvLayer(nn.Module):
    """StarE's convolution layer with qualifiers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.2,
        activation: HintOrType[nn.Module] = nn.ReLU,
        composition: HintOrType[Composition] = None,
        qualifier_aggregation: HintOrType[QualifierAggregation] = None,
        qualifier_aggregation_kwargs: Optional[Mapping[str, Any]] = None,
        qualifier_composition: HintOrType[Composition] = None,
        use_bias: bool = True,
        message_weighting: HintOrType[MessageWeighting] = None,
        message_weighting_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.0,
    ):
        """
        Initialize the layer.

        :param input_dim:
            The input dimension (entity and relation representations).
        :param output_dim:
            The output dimension. Defaults to the input dimension.
        :param dropout:
            The dropout to apply to the updated entity representations from forward / backward edges (but not for
            self-loops).
        :param activation:
            The activation function to use.
        :param composition:
            The composition function to use for merging entity and relation representations to messages.
        :param qualifier_aggregation:
            The aggregation method to use for aggregation of multiple qualifier pair representations for a single edge.
        :param qualifier_aggregation_kwargs:
            Additional keyword-based arguments for the aggregation method.
        :param qualifier_composition:
            The composition function to use to combine entity and relation representations from a qualifier pair.
        :param use_bias:
            Whether to add a trainable bias.
        :param edge_dropout:
            An additional dropout on the edges (applied by randomly setting edge weights to zero).
        """
        super().__init__()
        # Input normalization
        output_dim = output_dim or input_dim

        # sub-modules
        self.composition = composition_resolver.make(composition)
        self.qualifier_composition = composition_resolver.make(qualifier_composition)
        self.qualifier_aggregation = qualifier_aggregation_resolver.make(
            qualifier_aggregation,
            pos_kwargs=qualifier_aggregation_kwargs,
            input_dim=input_dim,
        )
        # attention layer needs to know about output dimension
        message_weighting_kwargs = dict(message_weighting_kwargs or {})
        if message_weighting == message_weighting_resolver.normalize_cls(AttentionMessageWeighting):
            message_weighting_kwargs.setdefault("output_dim", output_dim)
        self.message_weighting = message_weighting_resolver.make(
            message_weighting,
            pos_kwargs=message_weighting_kwargs,
        )
        self.activation = activation_resolver.make(activation)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.edge_dropout = nn.Dropout(edge_dropout)
        self.w_loop = get_parameter(input_dim, output_dim)
        self.w_in = get_parameter(input_dim, output_dim)
        self.w_out = get_parameter(input_dim, output_dim)
        self.w_rel = get_parameter(input_dim, output_dim)
        self.loop_rel = get_parameter(1, input_dim)
        self.bias = get_parameter(output_dim, initializer_=nn.init.zeros_) if use_bias else None

    def propagate(
        self,
        x_e: torch.FloatTensor,
        x_r: FloatTensor,
        edge_index: LongTensor,
        edge_type: LongTensor,
        qualifier_index: torch.LongTensor,
        weight: nn.Parameter,
    ) -> torch.FloatTensor:
        """
        The real message passing.

        :param x_e: shape: (num_entities, input_dim)
            The entity representations.
        :param x_r: shape: (2 * num_relations, dim)
            The relation representations. This includes the relation representation for inverse relations, but does
            not include the self-loop relation (which is learned independently for each layer).
        :param edge_index: shape: (2, num_edges)
            The edge index, pairs of source/target nodes. This does not include inverse edges, since they are created
            locally.
        :param edge_type: shape: (num_edges,)
            The edge type (=relation ID) for each edge.
        :param qualifier_index: shape: (3, num_qualifier_pairs)
            The qualifier index, triples of (qualifier-relation-ID, qualifier-entity-ID, edge-ID).
        :param weight: shape: (input_dim, output_dim)
            The transformation weight.
        """
        # split qualifier index: relation ID, entity ID, edge ID
        i_qr, i_qe, i_e = qualifier_index
        # get qualifier entity / relation embedding
        x_qr = x_r[i_qr]  # comment: we do not use inverse relations in qualifiers
        x_qe = x_e[i_qe]
        # pass it through qualifier transformation -> output is a representation for each qualifier pair
        x_q = self.qualifier_composition(x_qe, x_qr)
        # select relation representations for each edge
        # x_r = x_r[edge_type]
        # aggregate qualifier pair representations for each edge
        x_r = self.qualifier_aggregation(
            x_q,
            x_r[edge_type],
            edge_ids=i_e,
        )
        # split edge index
        source, target = edge_index
        # Use relations to transform entities
        # Note: Note that we generally refer to i as the central nodes that aggregates information, and refer to j
        # as the neighboring nodes, since this is the most common notation.
        m = self.composition(x_e[source], x_r) @ weight
        # weight messages
        m, message_weight = self.message_weighting(
            edge_index=edge_index,
            message=m,
            x_e=x_e,
        )
        # edge dropout
        message_weight = self.edge_dropout(message_weight)
        # weight messages
        m = m * message_weight.unsqueeze(dim=-1)
        # view as vectors again (needed for multi-head attention).
        m = m.view(m.shape[0], -1)
        # aggregate messages by sum
        # index_add is somewhat slower for large number of edges
        # x_e.new_zeros(x_e.shape[0], m.shape[1]).index_add(dim=0, index=target, source=m)
        return torch_scatter.scatter_add(src=m, index=target, dim=0, dim_size=x_e.shape[0])

    def forward(
        self,
        x_e: torch.FloatTensor,
        x_r: FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        qualifier_index: torch.LongTensor,
        entity_mask: Optional[torch.LongTensor],
    ) -> Tuple[FloatTensor, FloatTensor]:
        """
        Forward pass through the convolution layer.

        :param x_e: shape: (num_entities, input_dim)
            The entity representations.
        :param x_r: shape: (2 * num_relations, dim)
            The relation representations. This includes the relation representation for inverse relations, but does
            not include the self-loop relation (which is learned independently for each layer).
        :param edge_index: shape: (2, num_edges)
            The edge index, pairs of source/target nodes. This does not include inverse edges, since they are created
            locally.
        :param edge_type: shape: (num_edges,)
            The edge type (=relation ID) for each edge.
        :param qualifier_index: shape: (3, num_qualifier_pairs)
            The qualifier index, triples of (qualifier-relation-ID, qualifier-entity-ID, edge-ID).
        :param entity_mask: shape (num_entities, )
            If provided, this entities x_e[entity_mask] will not be updated be updated by this message passing layer.

        :return:
            The updated entity and relation representations.
        """
        assert (edge_type < x_r.shape[0] // 2).all()
        # self-loop
        out = 1 / 3 * self.composition(x_e, self.loop_rel) @ self.w_loop
        # messages
        for weight, edge_index_, edge_type_ in (
            (  # normal edges
                self.w_in,
                edge_index,
                edge_type,
            ),
            (  # reverse edges
                self.w_out,
                edge_index.flip(0),
                edge_type + x_r.shape[0] // 2,
            ),
        ):
            out = out + 1 / 3 * self.dropout(self.propagate(
                x_e=x_e,
                x_r=x_r,
                edge_index=edge_index_,
                edge_type=edge_type_,
                qualifier_index=qualifier_index,
                weight=weight,
            ))

        if self.bias is not None:
            out = out + self.bias

        out = self.batch_norm(out)
        out = self.activation(out)

        # Update relation representations
        x_r = x_r @ self.w_rel

        if entity_mask is not None:
            # forget about stuff which we should not have updated
            out[entity_mask] = x_e[entity_mask]

        return out, x_r
