"""Test GNN layers."""
from typing import Any, MutableMapping

import torch
import unittest_templates

from mphrqe.layer.gnn import StarEConvLayer


class StarEConvLayerTests(unittest_templates.GenericTestCase[StarEConvLayer]):
    """Tests for the StarE convolution layer."""

    cls = StarEConvLayer

    num_entities: int = 33
    num_relations: int = 7
    num_triples: int = 101
    num_qualifier_pairs: int = 131
    input_dim: int = 3

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["input_dim"] = self.input_dim
        return kwargs

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.x_e = torch.rand(self.num_entities, self.input_dim, requires_grad=True)
        self.x_r = torch.rand(2 * self.num_relations, self.input_dim, requires_grad=True)
        self.edge_index = torch.randint(high=self.num_entities, size=(2, self.num_triples))
        self.edge_type = torch.randint(high=self.num_relations, size=(self.num_triples,))
        self.qualifier_index = torch.stack([
            torch.randint(high=self.num_relations, size=(self.num_qualifier_pairs,)),
            torch.randint(high=self.num_entities, size=(self.num_qualifier_pairs,)),
            torch.randint(high=self.num_triples, size=(self.num_qualifier_pairs,)),
        ], dim=0)

    def test_forward(self):
        """Test the forward method."""
        out = self.instance(
            x_e=self.x_e,
            x_r=self.x_r,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            qualifier_index=self.qualifier_index,
            entity_mask=None,
        )
        # check type
        assert isinstance(out, tuple)
        assert len(out) == 2
        x_e, x_r = out
        assert torch.is_tensor(x_e)
        assert torch.is_tensor(x_r)

        # check shape
        assert x_e.shape == (self.num_entities, self.input_dim)
        assert x_r.shape == (2 * self.num_relations, self.input_dim)

        # check updated content
        assert not torch.allclose(x_e, self.x_e)
        assert not torch.allclose(x_r, self.x_r)
