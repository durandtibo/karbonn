r"""Contain ``torch.nn.Module``s to change tensor's shape."""

from __future__ import annotations

__all__ = ["Squeeze"]

import torch
from torch import nn


class Squeeze(nn.Module):
    r"""Implement a ``torch.nn.Module`` to squeeze the input tensor.

    Args:
        dim: The dimension to squeeze the input tensor. If ``None``,
            all the dimensions of the input tensor of size 1 are
            removed.

    Example usage:

    ```pycon
    >>> import torch
    >>> from karbonn import Squeeze
    >>> m = Squeeze()
    >>> m
    Squeeze(dim=None)
    >>> out = m(torch.ones(2, 1, 3, 1))
    >>> out.shape
    torch.Size([2, 3])

    ```
    """

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self._dim = dim

    def extra_repr(self) -> str:
        return f"dim={self._dim}"

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        if self._dim is None:
            return input.squeeze()
        return input.squeeze(self._dim)
