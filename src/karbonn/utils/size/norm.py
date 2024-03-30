r"""Contain a size finder for BatchNorm layers like
``torch.nn.BatchNorm1d``, ``torch.nn.BatchNorm2d``,
``torch.nn.BatchNorm3d``, or ``torch.nn.SyncBatchNorm``."""

from __future__ import annotations

__all__ = ["BatchNormSizeFinder"]


from torch import nn

from karbonn.utils.size.base import BaseSizeFinder, SizeNotFoundError


class BatchNormSizeFinder(BaseSizeFinder[nn.Module]):
    r"""Implement a size finder for BatchNorm layers like
    ``torch.nn.BatchNorm1d``, ``torch.nn.BatchNorm2d``,
    ``torch.nn.BatchNorm3d``, or ``torch.nn.SyncBatchNorm``.

    This module size finder assumes the module has a single input and
    output, and the input size is given by the attribute
    ``num_features`` and the output size is given by the attribute
    ``num_features``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.size import BatchNormSizeFinder
    >>> size_finder = BatchNormSizeFinder()
    >>> module = torch.nn.BatchNorm1d(num_features=6)
    >>> in_features = size_finder.find_in_features(module)
    >>> in_features
    [6]
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
        if not hasattr(module, "num_features"):
            msg = f"module {module} does not have attribute num_features"
            raise SizeNotFoundError(msg)
        return [module.num_features]

    def find_out_features(self, module: nn.Module) -> list[int]:
        if not hasattr(module, "num_features"):
            msg = f"module {module} does not have attribute num_features"
            raise SizeNotFoundError(msg)
        return [module.num_features]
