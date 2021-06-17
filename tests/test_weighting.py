"""Tests for weighting."""
from typing import Any, MutableMapping

import torch
import unittest_templates

from mphrqe.layer.weighting import AttentionMessageWeighting, MessageWeighting, SymmetricMessageWeighting


class MessageWeightingTests(unittest_templates.GenericTestCase[MessageWeighting]):
    """Tests for message weighting."""

    num_entities: int = 33
    num_edges: int = 101
    dim: int = 3

    def test_forward(self):
        # prepare data
        x_e = torch.rand(self.num_entities, self.dim)
        edge_index = torch.randint(self.num_entities, size=(2, self.num_edges))
        message = torch.rand(self.num_edges, self.dim, requires_grad=True)

        # forward pass
        out = self.instance(
            edge_index=edge_index,
            message=message,
            x_e=x_e,
        )

        # check type
        assert isinstance(out, tuple)
        assert len(out) == 2
        message_, weight_ = out
        assert torch.is_tensor(message_)
        assert torch.is_tensor(weight_)

        # check shape
        assert message_.shape[0] == self.num_edges
        assert weight_.shape[0] == self.num_edges

        weighted_message = message_ * weight_.unsqueeze(dim=-1)

        # try backward pass
        weighted_message.mean().backward()


class SymmetricMessageWeightingTests(MessageWeightingTests):
    """Tests for static symmetric message weighting."""

    cls = SymmetricMessageWeighting


class AttentionMessageWeightingTests(MessageWeightingTests):
    """Tests for message weighting by attention."""

    cls = AttentionMessageWeighting
    # make divisible by number of heads
    dim = 8
    num_heads = 2

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        # make sure that the output dimension is divisible by the number of heads.
        kwargs["num_heads"] = self.num_heads
        kwargs["output_dim"] = self.dim
        return kwargs


class MessageWeightingMetaTest(unittest_templates.MetaTestCase[MessageWeighting]):
    """Test for tests for message weightings."""

    base_cls = MessageWeighting
    base_test = MessageWeightingTests
