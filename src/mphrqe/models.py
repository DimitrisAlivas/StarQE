"""A module for models."""
import itertools
import logging
from abc import abstractmethod
from typing import Any, Iterator, Mapping, Optional, cast

import torch
from class_resolver import Hint
from torch import nn

from .data import QueryGraphBatch
from .layer import StarEConvLayer
from .layer.aggregation import QualifierAggregation
from .layer.composition import Composition
from .layer.pooling import GraphPooling, graph_pooling_resolver
from .layer.util import get_parameter
from .layer.weighting import MessageWeighting
from .typing import FloatTensor, LongTensor

__all__ = [
    "QueryEmbeddingModel",
    "StarEQueryEmbeddingModel",
]

logger = logging.getLogger(__name__)


class QueryEmbeddingModel(nn.Module):
    """The API for query embedding models."""

    @abstractmethod
    def forward(
        self,
        query_graph_batch: QueryGraphBatch,
    ) -> torch.Tensor:
        """
        Embed a batch of query graphs.

        :return: shape: (batch_size, output_dim)
            A vector representation for the query.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """The model's device."""
        raise NotImplementedError


class StarEQueryEmbeddingModel(QueryEmbeddingModel):
    """Query embedding with StarE."""

    def __init__(
        self,
        num_relations: int,
        num_entities: Optional[int] = None,
        embedding_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: Hint[nn.Module] = nn.ReLU,
        composition: Hint[Composition] = None,
        qualifier_aggregation: Hint[QualifierAggregation] = None,
        qualifier_aggregation_kwargs: Optional[Mapping[str, Any]] = None,
        qualifier_composition: Hint[Composition] = None,
        use_bias: bool = False,
        message_weighting: Hint[MessageWeighting] = None,
        message_weighting_kwargs: Optional[Mapping[str, Any]] = None,
        edge_dropout: float = 0.0,
        graph_pooling: Hint[GraphPooling] = None,
        repeat_layers_until_diameter: bool = False,
        stop_at_diameter: bool = False,
    ):
        """Initialize the model."""
        super().__init__()
        # create node embeddings if necessary
        self.x_e = None if num_entities is None else get_parameter(
            num_entities,
            embedding_dim,
            initializer_=nn.init.xavier_normal_,
        )
        # create relation embeddings (normal + inverse relations)
        self.x_r = get_parameter(2 * num_relations, embedding_dim)
        # TODO: Composition specific relation initialization?

        # create message passing layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(StarEConvLayer(
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                dropout=dropout,
                activation=activation,
                composition=composition,
                qualifier_aggregation=qualifier_aggregation,
                qualifier_aggregation_kwargs=qualifier_aggregation_kwargs,
                qualifier_composition=qualifier_composition,
                use_bias=use_bias,
                message_weighting=message_weighting,
                message_weighting_kwargs=message_weighting_kwargs,
                edge_dropout=edge_dropout,
            ))

        self.repeat_layers_until_diameter = repeat_layers_until_diameter
        self.stop_at_diameter = stop_at_diameter

        # create graph pooling
        self.pooling = graph_pooling_resolver.make(graph_pooling)

    @property
    def device(self) -> torch.device:
        """The model's device."""
        return self.x_r.device

    def forward(
        self,
        query_graph_batch: QueryGraphBatch,
    ) -> torch.FloatTensor:  # noqa: D102
        # Comment: Everything should be send to device in training loop
        x_e: FloatTensor = cast(FloatTensor, self.x_e)
        x_r: FloatTensor = cast(FloatTensor, self.x_r)

        entity_ids = query_graph_batch.entity_ids.to(self.device)
        x_e = x_e[entity_ids]

        relation_ids: LongTensor = query_graph_batch.relation_ids.to(self.device)
        # add inverse relations
        relation_ids = torch.cat([relation_ids, relation_ids + self.x_r.shape[0] // 2], dim=-1)
        x_r = x_r[relation_ids]

        edge_index = query_graph_batch.edge_index.to(self.device)
        edge_type = query_graph_batch.edge_type.to(self.device)
        qualifier_index = query_graph_batch.qualifier_index.to(self.device)
        graph_ids = query_graph_batch.graph_ids.to(self.device)

        # Next we select the layers we are going to use. This is dependent on wether layers are repeated and whether we stop at the diameter
        # If we can stop at the diamter, and knowing the maximum diameter, we do not even take the unnecessary layers in the iterator
        if self.repeat_layers_until_diameter or self.stop_at_diameter:
            max_diameter = int(max(query_graph_batch.query_diameter))
        if self.repeat_layers_until_diameter and self.stop_at_diameter:
            needed_layers = itertools.islice(itertools.cycle(self.layers), 0, max_diameter)
        elif self.repeat_layers_until_diameter and not self.stop_at_diameter:
            # we have to continue to at least the numebr of layers
            raise NotImplementedError("This seemed like a strange combination. If needed, the code is commented out.")
            # needed_layers = itertools.islice(itertools.cycle(self.layers), 0, max(max_diameter, len(self.layers)))
        elif not self.repeat_layers_until_diameter and self.stop_at_diameter:
            needed_layers = itertools.islice(self.layers, 0, min(len(self.layers), max_diameter))
        elif not self.repeat_layers_until_diameter and not self.stop_at_diameter:
            needed_layers = cast(Iterator[Any], self.layers)
        else:
            raise Exception("")

        if self.stop_at_diameter:
            diameter_left = query_graph_batch.query_diameter.to(self.device)
            # expand to all entities in the batch, this makes it easier later.
            diameter_left = torch.index_select(diameter_left, 0, graph_ids)

        for layer in needed_layers:
            if self.stop_at_diameter:
                diameter_mask_tensor: torch.Tensor = diameter_left <= 0
                # if the logic above for limiting the number of layers is correct, then we should never do any useless layer() calls.
                assert not diameter_mask_tensor.all()
                diameter_mask: Optional[torch.Tensor] = diameter_mask_tensor
            else:
                diameter_mask = None

            x_e, x_r = layer(
                x_e=x_e,
                x_r=x_r,
                edge_index=edge_index,
                edge_type=edge_type,
                qualifier_index=qualifier_index,
                entity_mask=diameter_mask,
            )
            if self.stop_at_diameter:
                diameter_left -= 1

        return self.pooling(x_e=x_e, graph_ids=graph_ids, entity_ids=entity_ids)
