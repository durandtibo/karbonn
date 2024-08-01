r"""Contain tensor utility functions."""

from __future__ import annotations

__all__ = ["FlattenBuffer", "quantile", "to_tensor"]

from karbonn.utils.tensor.buffer import FlattenBuffer
from karbonn.utils.tensor.conversion import to_tensor
from karbonn.utils.tensor.mathops import quantile
