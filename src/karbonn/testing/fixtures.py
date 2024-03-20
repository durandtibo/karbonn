r"""Define some PyTest fixtures."""

from __future__ import annotations

__all__ = ["objectory_available"]

import pytest

from karbonn.utils.imports import is_objectory_available

objectory_available = pytest.mark.skipif(not is_objectory_available(), reason="Require objectory")
