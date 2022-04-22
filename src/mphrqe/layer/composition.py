"""Composition functions, i.e. binary operators on vectors."""
from abc import abstractmethod

import torch
from class_resolver import Resolver
from pykeen.moves import irfft, rfft
from torch import nn

__all__ = [
    "Composition",
    "composition_resolver",
]


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """View real tensor as complex."""
    return torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))


def _to_real(x: torch.Tensor) -> torch.Tensor:
    """View complex tensor as real."""
    x = torch.view_as_real(x)
    return x.view(*x.shape[:-2], -1)


def _complex_multiplication(
    x: torch.Tensor,
    y: torch.Tensor,
    y_norm: bool = False,
) -> torch.Tensor:
    """Element-wise multiplication as complex numbers."""
    # view complex
    x = _to_complex(x)
    y = _to_complex(y)

    if y_norm:
        # normalize y
        y = y / y.abs().clamp_min(1.0e-08)

    # (complex) multiplication
    x = x * y

    # view real
    return _to_real(x)


def _fourier_multiplication(
    x: torch.Tensor,
    y: torch.Tensor,
    x_conj: bool = False,
) -> torch.Tensor:
    """Element-wise multiplication in Fourier space."""
    d = x.shape[-1]

    # convert to fourier space
    x = rfft(x, n=d)
    y = rfft(y, n=d)

    # complex conjugate
    if x_conj:
        x = torch.conj(x)

    # (complex) multiplication
    x = x * y

    # convert back from Fourier space
    return irfft(x, n=d)


class Composition(nn.Module):
    """A base class for compositions."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compose two batches vectors.


        .. note ::
            The two batches have to be of broadcastable shape.

        :param x: shape: s_x
            The first batch of vectors.
        :param y: shape: s_y
            The second batch of vectors.

        :return: shape: s
            The compositionm, where `s` is the broadcasted shape.
        """
        raise NotImplementedError


class MultiplicationComposition(Composition):
    """Element-wise multiplication, a.k.a. Hadamard product."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return x * y


class ComplexMultiplicationComposition(Composition):
    """Element-wise multiplication, a.k.a. Hadamard product, of two complex numbers."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return _complex_multiplication(x, y)


class CircularConvolutionComposition(Composition):
    """Composition by circular convolution."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return _fourier_multiplication(x, y, x_conj=False)


class CircularCorrelationComposition(Composition):
    """Composition by circular correlation."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return _fourier_multiplication(x, y, x_conj=True)


class ComplexRotationComposition(Composition):
    """Composition by rotation in complex plane."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:  # noqa: D102
        return _complex_multiplication(x, y, y_norm=True)


composition_resolver: Resolver[Composition] = Resolver.from_subclasses(
    base=Composition,  # type: ignore
    default=MultiplicationComposition,
)
