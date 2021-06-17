"""Tests for similarity functions."""
import unittest
from typing import ClassVar, Optional

import torch
import unittest_templates

from mphrqe.similarity import CosineSimilarity, DotProductSimilarity, NegativeLpSimilarity, NegativePowerNormSimilarity, Similarity


class SimilarityTests(unittest_templates.GenericTestCase[Similarity]):
    """Common test cases for similarity functions."""

    dim: int = 7
    left: int = 5
    right: int = 3

    lower_bound: ClassVar[Optional[float]] = None
    upper_bound: ClassVar[Optional[float]] = None

    def test_forward(self):
        """Test pairwise similarity computation."""
        batch_size = (11, 2)
        x = torch.rand(batch_size[0], 1, self.left, self.dim)
        y = torch.rand(1, batch_size[1], self.right, self.dim)
        s = self.instance(x=x, y=y)
        assert s.shape == batch_size + (self.left, self.right)

    def test_backward(self):
        """Test backward."""
        x = torch.rand(self.left, self.dim, requires_grad=True)
        y = torch.rand(self.right, self.dim)
        s = self.instance(x=x, y=y)
        assert s.requires_grad
        s.mean().backward()

    def test_one_to_one(self):
        """Test one-to-one similarity computation."""
        batch_size = (11, 2)
        x = torch.rand(*batch_size, self.dim, requires_grad=True)
        y = torch.rand(*batch_size, self.dim)
        s = self.instance.one_to_one(x=x, y=y)
        assert s.shape == batch_size

    def test_value_range(self):
        """Test value range (empirically)."""
        batch_size = (11, 2)
        x = torch.rand(*batch_size, self.left, self.dim)
        y = torch.rand(*batch_size, self.right, self.dim)
        s = self.instance(x=x, y=y)
        if self.lower_bound is not None:
            assert (s >= self.lower_bound).all()
        if self.upper_bound is not None:
            assert (s <= self.upper_bound).all()


class NegativeLpSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for negative l_p distance similarity."""

    cls = NegativeLpSimilarity
    upper_bound = 0.0


class NegativePowerNormSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for negative power norm similarity."""

    cls = NegativePowerNormSimilarity
    upper_bound = 0.0


class DotProductSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for dot product similarity."""

    cls = DotProductSimilarity


class CosineSimilarityTests(SimilarityTests, unittest.TestCase):
    """Tests for cosine similarity."""

    cls = CosineSimilarity
    lower_bound = -1.0
    upper_bound = 1.0
