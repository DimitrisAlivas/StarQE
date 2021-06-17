"""Tests for losses."""
import torch
import unittest_templates

from mphrqe.loss import BCEQueryEmbeddingLoss, QueryEmbeddingLoss


class QueryEmbeddingLossTests(unittest_templates.GenericTestCase[QueryEmbeddingLoss]):
    """Tests for query embedding losses."""

    batch_size: int = 2
    num_entities: int = 7
    num_targets: int = 5

    def post_instantiation_hook(self) -> None:  # noqa: D102
        self.scores = torch.rand(self.batch_size, self.num_entities, requires_grad=True)
        self.targets = torch.stack([
            torch.randint(high=self.batch_size, size=(self.num_targets,)),
            torch.randint(high=self.num_entities, size=(self.num_targets,)),
        ], dim=0).unique(dim=1)

    def test_forward(self):
        """Test forward."""
        loss = self.instance(scores=self.scores, targets=self.targets)
        # check type
        assert torch.is_tensor(loss)
        # check scalar loss
        assert loss.shape == tuple()
        # check backward
        loss.backward()


class BCEQueryEmbeddingLossTests(QueryEmbeddingLossTests):
    """Tests for BCE loss."""

    cls = BCEQueryEmbeddingLoss

    def test_consistency_with_torch_builtin(self):
        """Test consistency with torch.nn.BCEWithLogits."""
        loss_value = self.instance(scores=self.scores, targets=self.targets)
        reference_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        dense_targets = torch.zeros_like(self.scores)
        row_id, col_id = self.targets
        dense_targets[row_id, col_id] = 1.0
        loss_reference_value = reference_loss(self.scores, dense_targets)
        assert torch.allclose(loss_value, loss_reference_value)


class QueryEmbeddingLossMetaTest(unittest_templates.MetaTestCase[QueryEmbeddingLoss]):
    """Test for tests for query embedding losses."""

    base_cls = QueryEmbeddingLoss
    base_test = QueryEmbeddingLossTests
