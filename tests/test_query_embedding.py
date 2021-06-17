"""Tests for query embedding model."""
from typing import Any, MutableMapping

import torch
import unittest_templates

from mphrqe.data import QueryGraphBatch
from mphrqe.models import QueryEmbeddingModel, StarEQueryEmbeddingModel


class QueryEmbeddingModelTests(unittest_templates.GenericTestCase[QueryEmbeddingModel]):
    """Tests for query embedding model."""

    num_graphs: int = 3
    num_entities: int = 33
    num_relations: int = 7
    num_triples: int = 101
    num_qualifier_pairs: int = 131
    input_dim: int = 2

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.edge_index = torch.randint(high=self.num_entities, size=(2, self.num_triples))
        self.edge_type = torch.randint(high=self.num_relations, size=(self.num_triples,))
        self.qualifier_index = torch.stack([
            torch.randint(high=self.num_relations, size=(self.num_qualifier_pairs,)),
            torch.randint(high=self.num_entities, size=(self.num_qualifier_pairs,)),
            torch.randint(high=self.num_triples, size=(self.num_qualifier_pairs,)),
        ], dim=0)
        self.graph_ids = torch.randint(high=self.num_graphs, size=(self.num_entities,))

    def test_forward_entity_features(self):
        """Test query embedding."""
        query_graph_batch = QueryGraphBatch(
            # x=torch.rand(self.num_entities, self.input_dim),  # entity features
            entity_ids=torch.randint(high=self.num_entities, size=(self.num_entities,)),
            relation_ids=torch.randint(high=self.num_relations, size=(self.num_relations,)),
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            qualifier_index=self.qualifier_index,
            graph_ids=self.graph_ids,
            query_diameter=...,
            targets=...,
        )
        x_g = self.instance(query_graph_batch)
        assert torch.is_tensor(x_g)
        assert x_g.shape == (self.num_graphs, self.input_dim)


class StarEQueryEmbeddingModelTests(QueryEmbeddingModelTests):
    """Test for the StarE query embedding model."""

    cls = StarEQueryEmbeddingModel

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["num_entities"] = self.num_entities
        kwargs["num_relations"] = self.num_relations
        kwargs["embedding_dim"] = self.input_dim
        return kwargs
