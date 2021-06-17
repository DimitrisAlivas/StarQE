"""Utility methods."""
from typing import Union

import torch
from class_resolver import Resolver
from torch import nn
from torch_scatter import scatter_add, scatter_max

__all__ = [
    "softmax",
    "activation_resolver",
    "get_parameter",
]


def softmax(
    src: torch.Tensor,
    index: torch.LongTensor,
    num_nodes: Union[None, int, torch.Tensor] = None,
    dim: int = 0,
) -> torch.Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the given dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    :param src:
        The source tensor.
    :param index:
        The indices of elements for applying the softmax.
    :param num_nodes:
        The number of nodes, i.e., :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    :param dim:
        The dimension along which to compute the softmax.

    :return:
        The softmax-ed tensor.
    """
    num_nodes = num_nodes or index.max() + 1
    out = src.transpose(dim, 0)
    out = out - scatter_max(out, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / scatter_add(out, index, dim=0, dim_size=num_nodes)[index].clamp_min(1.0e-16)
    return out.transpose(0, dim)


def get_parameter(
    *shape: int,
    initializer_=nn.init.xavier_normal_,
) -> nn.Parameter:
    """Get an initialized parameter."""
    param = nn.Parameter(torch.empty(*shape))
    initializer_(param.data)
    return param


activation_resolver = Resolver(
    classes={
        nn.LeakyReLU,
        nn.Identity,
        nn.PReLU,
        nn.ReLU,
        nn.Sigmoid,
        nn.Tanh,
    },
    base=nn.Module,  # type: ignore
    default=nn.ReLU,
)
