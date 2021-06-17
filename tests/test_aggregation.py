"""Tests for qualifier aggregation."""
from typing import Any, MutableMapping, Optional

import torch
import unittest_templates

from mphrqe.layer.aggregation import AttentionQualifierAggregation, ConcatQualifierAggregation, MulQualifierAggregation, QualifierAggregation, SumQualifierAggregation


class QualifierAggregationTests(unittest_templates.GenericTestCase[QualifierAggregation]):
    """Tests for qualifier aggregation."""

    num_qualifier_pairs: int = 37
    num_edges: int = 33
    input_dim: int = 12
    output_dim: Optional[int] = None

    def pre_setup_hook(self) -> None:  # noqa: D102
        self.output_dim = self.output_dim or self.input_dim

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        kwargs = super()._pre_instantiation_hook(kwargs=kwargs)
        kwargs["input_dim"] = self.input_dim
        return kwargs

    @property
    def expected_input_dim(self):  # noqa: D102
        return self.input_dim

    def test_dims(self):
        """Test dimension properties."""
        assert self.instance.input_dim == self.expected_input_dim
        assert self.instance.output_dim == self.output_dim

    def test_forward(self):
        """Test for the forward method."""
        x_q = torch.rand(self.num_qualifier_pairs, self.input_dim)
        x_edge = torch.rand(self.num_edges, self.input_dim)
        edge_ids = torch.randint(high=self.num_edges, size=(self.num_qualifier_pairs,))
        y = self.instance(
            x_q=x_q,
            x_edge=x_edge,
            edge_ids=edge_ids,
        )

        # check type
        assert torch.is_tensor(y)
        assert y.shape == (self.num_edges, self.output_dim)


class SumQualifierAggregationTests(QualifierAggregationTests):
    """Tests for aggregation by sum."""

    cls = SumQualifierAggregation


class MulQualifierAggregationTests(QualifierAggregationTests):
    """Tests for aggregation by product."""

    cls = MulQualifierAggregation


class ConcatQualifierAggregationTests(QualifierAggregationTests):
    """Tests for aggregation by concatenation."""

    cls = ConcatQualifierAggregation

    @property
    def expected_input_dim(self):
        return 2 * super().expected_input_dim


class AttentionQualifierAggregationTests(QualifierAggregationTests):
    """Tests for aggregation by attention."""

    cls = AttentionQualifierAggregation


class QualifierAggregationMetaTest(unittest_templates.MetaTestCase[QualifierAggregation]):
    """Test for tests for qualifier aggregations."""

    base_cls = QualifierAggregation
    base_test = QualifierAggregationTests
