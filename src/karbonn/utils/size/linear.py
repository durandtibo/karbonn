r"""Contain a size finder for ``torch.nn.Linear`` layer or similar
layers."""

from __future__ import annotations

__all__ = ["LinearSizeFinder"]


from typing import TYPE_CHECKING

from karbonn.utils.size.base import BaseSizeFinder, SizeNotFoundError

if TYPE_CHECKING:
    from torch import nn


class LinearSizeFinder(BaseSizeFinder):
    r"""Implement a size finder for ``torch.nn.Linear`` layer or similar
    layers.

    This module size finder assumes the module has a single input and
    output, and the input size is given by the attribute
    ``in_features`` and the output size is given by the attribute
    ``out_features``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.size import LinearSizeFinder
    >>> size_finder = LinearSizeFinder()
    >>> module = torch.nn.Linear(4, 6)
    >>> in_features = size_finder.find_in_features(module)
    >>> in_features
    [4]
    >>> out_features = size_finder.find_out_features(module)
    >>> out_features
    [6]

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def find_in_features(self, module: nn.Module) -> list[int]:
        if not hasattr(module, "in_features"):
            msg = f"module {module} does not have attribute in_features"
            raise SizeNotFoundError(msg)
        return [module.in_features]

    def find_out_features(self, module: nn.Module) -> list[int]:
        if not hasattr(module, "out_features"):
            msg = f"module {module} does not have attribute out_features"
            raise SizeNotFoundError(msg)
        return [module.out_features]
