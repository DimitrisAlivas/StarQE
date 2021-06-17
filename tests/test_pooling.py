"""Test for graph pooling."""
import random

import pytest
import torch
import unittest_templates

from mphrqe.data.mapping import get_entity_mapper
from mphrqe.layer.pooling import GraphPooling, SumGraphPooling, TargetPooling


class GraphPoolingTests(unittest_templates.GenericTestCase[GraphPooling]):
    """Tests for graph pooling."""

    num_nodes: int = 33
    num_graphs: int = 7
    dim: int = 3

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.x_e = torch.rand(self.num_nodes, self.dim)
        self.entity_ids = torch.empty(self.num_nodes, dtype=torch.long)
        self.graph_ids = torch.empty(self.num_nodes, dtype=torch.long)
        j = 0
        for i in range(self.num_graphs - 1):
            # randomly select size of graph
            size = random.randrange(1, self.num_nodes - j - self.num_graphs + i)
            self.graph_ids[j:j + size] = i
            self.entity_ids[j] = self.target_index
            self.entity_ids[j + 1:j + size] = torch.randint(self.num_nodes, size=(size - 1,))
            j += size
        self.graph_ids[j:] = self.num_graphs - 1
        self.entity_ids[j] = self.target_index
        self.entity_ids[j + 1:] = torch.randint(self.num_nodes, size=(self.num_nodes - j - 1,))

    @property
    def target_index(self) -> int:
        # The target index. Only use get_entity_mapper for those which require the entity mapper
        return -1

    def test_forward(self):
        x_g = self.instance(
            x_e=self.x_e,
            graph_ids=self.graph_ids,
            entity_ids=self.entity_ids,
        )
        assert torch.is_tensor(x_g)
        assert x_g.shape == (self.num_graphs, self.dim)


class SumGraphPoolingTests(GraphPoolingTests):
    """Tests for sum aggregation."""

    cls = SumGraphPooling


@pytest.mark.full_data
class TargetPoolingTests(GraphPoolingTests):
    """Tests for target only aggregation."""

    cls = TargetPooling

    @property
    def target_index(self) -> int:
        # TODO: can we mock get_entity_mapper?
        return get_entity_mapper().highest_entity_index + 1


class GraphPoolingMetaTest(unittest_templates.MetaTestCase[GraphPooling]):
    """Test for tests for graph pooling."""

    base_cls = GraphPooling
    base_test = GraphPoolingTests
