r"""Contain activation layers using mathematical functions."""

from __future__ import annotations

__all__ = ["Asinh", "Expm1", "Log1p", "Sinh"]

import torch
from torch import nn


class Asinh(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the inverse hyperbolic
    sine (arcsinh) of the elements.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import Asinh
    >>> m = Asinh()
    >>> m
    Asinh()
    >>> output = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
    >>> output
    tensor([[-0.8814,  0.0000,  0.8814],
            [-1.4436,  1.4436,  2.0947]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.asinh()


class Expm1(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    elements minus 1 of input.

    This module is equivalent to  ``exp(input) - 1``

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import Expm1
    >>> m = Expm1()
    >>> m
    Expm1()
    >>> output = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
    >>> output
    tensor([[-0.6321,  0.0000,  1.7183],
            [-0.8647,  6.3891, 53.5981]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.expm1()


class Log1p(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the natural logarithm
    of ``(1 + input)``.

    This module is equivalent to  ``log(1 + input)``

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import Log1p
    >>> m = Log1p()
    >>> m
    Log1p()
    >>> output = m(torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
    >>> output
    tensor([[0.0000, 0.6931, 1.0986],
            [1.3863, 1.6094, 1.7918]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.log1p()


class Sinh(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the hyperbolic sine
    (sinh) of the elements.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import Sinh
    >>> m = Sinh()
    >>> m
    Sinh()
    >>> output = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
    >>> output
    tensor([[-1.1752,  0.0000,  1.1752],
            [-3.6269,  3.6269, 27.2899]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sinh()
