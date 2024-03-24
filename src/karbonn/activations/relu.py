r"""Contain relu-like activation modules."""

from __future__ import annotations

__all__ = ["ReLUn", "SquaredReLU"]

from typing import TYPE_CHECKING

from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    import torch


class ReLUn(nn.Module):
    r"""Implements the ReLU-n module.

    The ReLU-n equation is: ``ReLUn(x, n)=min(max(0,x),n)``

    Args:
        max: The maximum value a.k.a. ``n`` in the equation above.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import ReLUn
    >>> m = ReLUn(max=5)
    >>> m
    ReLUn(max=5.0)
    >>> output = m(torch.arange(8).view(2, 4))
    >>> output
    tensor([[0., 1., 2., 3.],
            [4., 5., 5., 5.]])

    ```
    """

    def __init__(self, max: float = 1.0) -> None:  # noqa: A002
        super().__init__()
        self._max = float(max)

    def extra_repr(self) -> str:
        return f"max={self._max}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Apply the element-wise ReLU-n function.

        Args:
            tensor: The input tensor.

        Returns:
            The output tensor, which as the same size as the input.
        """
        return tensor.clamp(min=0.0, max=self._max)


class SquaredReLU(nn.Module):
    r"""Implements the Squared ReLU.

    Squared ReLU is defined in the following paper:

        Primer: Searching for Efficient Transformers for Language Modeling.
        So DR., Mańke W., Liu H., Dai Z., Shazeer N., Le QV.
        NeurIPS, 2021. (https://arxiv.org/pdf/2109.08668.pdf)

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import SquaredReLU
    >>> m = SquaredReLU()
    >>> m
    SquaredReLU()
    >>> output = m(torch.arange(8).view(2, 4))
    >>> output
    tensor([[ 0,  1,  4,  9],
            [16, 25, 36, 49]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        x = functional.relu(tensor)
        return x * x
