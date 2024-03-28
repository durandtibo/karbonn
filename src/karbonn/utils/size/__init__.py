r"""Contain code to find the input and output size of some
``torch.nn.Module``s."""

from __future__ import annotations

__all__ = [
    "BaseSizeFinder",
    "BilinearSizeFinder",
    "LinearSizeFinder",
    "AutoSizeFinder",
    "SizeFinderConfig",
    "get_karbonn_size_finders",
    "get_size_finders",
    "get_torch_size_finders",
    "register_size_finders",
]

from karbonn.utils.size.auto import AutoSizeFinder, register_size_finders
from karbonn.utils.size.base import BaseSizeFinder, SizeFinderConfig
from karbonn.utils.size.linear import BilinearSizeFinder, LinearSizeFinder
from karbonn.utils.size.utils import (
    get_karbonn_size_finders,
    get_size_finders,
    get_torch_size_finders,
)

register_size_finders()
