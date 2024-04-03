from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest
import torch

from karbonn.utils.summary.module import parse_batch_shape

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SIZES = [1, 2, 3]

#######################################
#     Tests for parse_batch_shape     #
#######################################


@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("input_size", SIZES)
def test_parse_batch_shape_tensor_2d(batch_size: int, input_size: int) -> None:
    assert parse_batch_shape(torch.ones(batch_size, input_size)) == torch.Size(
        [batch_size, input_size]
    )


@pytest.mark.parametrize(("batch_size", "seq_len", "input_size"), [(1, 1, 1), (2, 2, 2)])
def test_parse_batch_shape_tensor_3d(batch_size: int, seq_len: int, input_size: int) -> None:
    assert parse_batch_shape(torch.ones(batch_size, seq_len, input_size)) == torch.Size(
        [batch_size, seq_len, input_size]
    )


@pytest.mark.parametrize(
    "batch",
    [
        (torch.ones(2, 3), torch.ones(2, dtype=torch.long)),
        [torch.ones(2, 3), torch.ones(2, dtype=torch.long)],
    ],
)
def test_parse_batch_shape_sequence(batch: Sequence[torch.Tensor]) -> None:
    assert parse_batch_shape(batch) == (torch.Size([2, 3]), torch.Size([2]))


def test_parse_batch_shape_sequence_empty() -> None:
    assert parse_batch_shape([]) == ()


@pytest.mark.parametrize(
    "batch",
    [
        {"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)},
        OrderedDict({"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)}),
    ],
)
def test_parse_batch_shape_dict(batch: Mapping[str, torch.Tensor]) -> None:
    assert parse_batch_shape(batch) == {"feature1": torch.Size([2, 3]), "feature2": torch.Size([2])}


def test_parse_batch_shape_dict_empty() -> None:
    assert parse_batch_shape({}) == {}


def test_parse_batch_shape_invalid() -> None:
    assert parse_batch_shape(set()) is None
