"""Tests for composition functions."""
from typing import Collection, Sequence

import torch
import unittest_templates

import mphrqe.layer.composition


class CompositionTests(unittest_templates.GenericTestCase):
    """Common test cases for composition functions."""

    dim: int = 4
    prefix_shape: Sequence[int] = (5, 3)
    names: Collection[str] = tuple()

    def test_forward(self):
        """Test pairwise similarity computation."""
        x = torch.rand(*self.prefix_shape, self.dim)
        y = torch.rand(*self.prefix_shape, self.dim)
        c = self.instance(x=x, y=y)
        assert c.shape == x.shape

    def test_names(self):
        """Test lookup via names."""
        for name in self.names:
            cls = mphrqe.layer.composition.composition_resolver.lookup(query=name)
            assert cls is self.cls


class MultiplicationCompositionTests(CompositionTests):
    """Tests for composition by element-wise multiplication."""

    cls = mphrqe.layer.composition.MultiplicationComposition
    names = ["multiplication"]


class ComplexMultiplicationCompositionTests(CompositionTests):
    """Tests for composition by element-wise complex multiplication."""

    cls = mphrqe.layer.composition.ComplexMultiplicationComposition
    names = ["complex-multiplication"]


class CircularConvolutionCompositionTests(CompositionTests):
    """Tests for composition by circular convolution."""

    cls = mphrqe.layer.composition.CircularConvolutionComposition
    names = ["circular-convolution"]


class CircularCorrelationCompositionTests(CompositionTests):
    """Tests for composition by circular correlation."""

    cls = mphrqe.layer.composition.CircularCorrelationComposition
    names = ["circular-correlation"]


class ComplexRotationCompositionTests(CompositionTests):
    """Tests for composition by rotation in complex plane."""

    cls = mphrqe.layer.composition.ComplexRotationComposition
    names = ["complex-rotation"]
