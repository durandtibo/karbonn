from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from coola.testing import numpy_available
from coola.utils import is_numpy_available

from karbonn.utils.tensor.conversion import to_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

###############################
#     Tests for to_tensor     #
###############################


@pytest.mark.parametrize(
    "value",
    [torch.tensor([-3, 1, 7]), [-3, 1, 7], (-3, 1, 7)],
)
def test_to_tensor(value: torch.Tensor | Sequence) -> None:
    assert to_tensor(value).equal(torch.tensor([-3, 1, 7]))


@numpy_available
def test_to_tensor_numpy() -> None:
    assert to_tensor(np.array([-3, 1, 7])).equal(torch.tensor([-3, 1, 7]))


def test_to_tensor_int() -> None:
    assert to_tensor(1).equal(torch.tensor(1, dtype=torch.long))


def test_to_tensor_float() -> None:
    assert to_tensor(1.5).equal(torch.tensor(1.5, dtype=torch.float))


def test_to_tensor_empty_list() -> None:
    assert to_tensor([]).equal(torch.tensor([]))


def test_to_tensor_incorrect() -> None:
    with pytest.raises(TypeError, match=r"Incorrect type:"):
        to_tensor(Mock())
