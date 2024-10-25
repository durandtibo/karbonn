r"""Contain module to encode or decode numerical values."""

from __future__ import annotations

__all__ = ["AsinhNumericalEncoder", "CosSinNumericalEncoder", "AsinhCosSinNumericalEncoder"]

from karbonn.modules.numerical.asinh import AsinhNumericalEncoder
from karbonn.modules.numerical.sine import (
    AsinhCosSinNumericalEncoder,
    CosSinNumericalEncoder,
)
