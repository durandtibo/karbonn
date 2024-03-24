r"""Contain activation layers using mathematical functions."""

from __future__ import annotations

__all__ = ["Asinh", "Exp", "Expm1", "Log", "Log1p", "SafeExp", "SafeLog", "Sinh"]

import torch
from torch import nn

from karbonn.functional import safe_exp, safe_log


class Asinh(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the inverse hyperbolic
    sine (arcsinh) of the elements.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

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


class Exp(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    input.

    This module is equivalent to  ``exp(input)``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import Exp
    >>> m = Exp()
    >>> m
    Exp()
    >>> output = m(torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 2.0, 4.0]]))
    >>> output
    tensor([[ 0.3679,  1.0000,  2.7183],
            [ 0.1353,  7.3891, 54.5981]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.exp()


class Expm1(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    elements minus 1 of input.

    This module is equivalent to  ``exp(input) - 1``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

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


class Log(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the natural logarithm
    of the input.

    This module is equivalent to  ``log(input)``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import Log
    >>> m = Log()
    >>> m
    Log()
    >>> output = m(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    >>> output
    tensor([[0.0000, 0.6931, 1.0986],
            [1.3863, 1.6094, 1.7918]])

    ```
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.log()


class Log1p(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the natural logarithm
    of ``(1 + input)``.

    This module is equivalent to  ``log(1 + input)``

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

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


class SafeExp(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the exponential of the
    elements.

    The values that are higher than the specified minimum value are
    set to this maximum value. Using a not too large positive value
    leads to an output tensor without Inf.

    Args:
        max: The maximum value before to compute the exponential.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import SafeExp
    >>> m = SafeExp()
    >>> m
    SafeExp(max=20.0)
    >>> output = m(torch.tensor([[0.01, 0.1, 1.0], [10.0, 100.0, 1000.0]]))
    >>> output
    tensor([[1.0101e+00, 1.1052e+00, 2.7183e+00],
            [2.2026e+04, 4.8517e+08, 4.8517e+08]])

    ```
    """

    def __init__(self, max: float = 20.0) -> None:  # noqa: A002
        super().__init__()
        self._max = float(max)

    def extra_repr(self) -> str:
        return f"max={self._max}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return safe_exp(tensor, self._max)


class SafeLog(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the logarithm natural
    of the elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. Using a small positive value leads to an
    output tensor without NaN or Inf.

    Args:
        min: The minimum value before to compute the logarithm natural.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import SafeLog
    >>> m = SafeLog()
    >>> m
    SafeLog(min=1e-08)
    >>> output = m(torch.tensor([[1e-4, 1e-5, 1e-6], [1e-8, 1e-9, 1e-10]]))
    >>> output
    tensor([[ -9.2103, -11.5129, -13.8155],
            [-18.4207, -18.4207, -18.4207]])

    ```
    """

    def __init__(self, min: float = 1e-8) -> None:  # noqa: A002
        super().__init__()
        self._min = float(min)

    def extra_repr(self) -> str:
        return f"min={self._min}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return safe_log(tensor, self._min)


class Sinh(nn.Module):
    r"""Implement a ``torch.nn.Module`` to compute the hyperbolic sine
    (sinh) of the elements.

    Shape:
        - Input: ``(*)``, where ``*`` means any number of dimensions.
        - Output: ``(*)``, same shape as the input.

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
