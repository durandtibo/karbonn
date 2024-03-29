r"""Contain code to find the input and output size of some
``torch.nn.Module``s."""

from __future__ import annotations

__all__ = [
    "AutoSizeFinder",
    "BaseSizeFinder",
    "BilinearSizeFinder",
    "LinearSizeFinder",
    "RecurrentSizeFinder",
    "SequentialSizeFinder",
    "SizeFinderConfig",
    "UnknownSizeFinder",
    "get_karbonn_size_finders",
    "get_size_finders",
    "get_torch_size_finders",
    "register_size_finders",
]

from karbonn.utils.size.auto import AutoSizeFinder, register_size_finders
from karbonn.utils.size.base import BaseSizeFinder, SizeFinderConfig
from karbonn.utils.size.linear import BilinearSizeFinder, LinearSizeFinder
from karbonn.utils.size.recurrent import RecurrentSizeFinder
from karbonn.utils.size.sequential import SequentialSizeFinder
from karbonn.utils.size.unknown import UnknownSizeFinder
from karbonn.utils.size.utils import (
    get_karbonn_size_finders,
    get_size_finders,
    get_torch_size_finders,
)

register_size_finders()
