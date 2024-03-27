from __future__ import annotations

from typing import NamedTuple

import pytest
from torch import nn

from karbonn import ExU
from karbonn.utils.size import LinearSizeFinder
from karbonn.utils.size.base import SizeNotFoundError


class ModuleSizes(NamedTuple):
    module: nn.Module
    in_features: list[int]
    out_features: list[int]


LINEAR_MODULES = [
    ModuleSizes(module=nn.Linear(4, 6), in_features=[4], out_features=[6]),
    ModuleSizes(module=nn.Linear(2, 1), in_features=[2], out_features=[1]),
    ModuleSizes(module=ExU(4, 6), in_features=[4], out_features=[6]),
    ModuleSizes(module=ExU(2, 1), in_features=[2], out_features=[1]),
]


######################################
#     Tests for LinearSizeFinder     #
######################################


def test_linear_size_finder_repr() -> None:
    assert repr(LinearSizeFinder()).startswith("LinearSizeFinder(")


def test_linear_size_finder_str() -> None:
    assert str(LinearSizeFinder()).startswith("LinearSizeFinder(")


def test_linear_size_finder_eq_true() -> None:
    assert LinearSizeFinder() == LinearSizeFinder()


def test_linear_size_finder_eq_false() -> None:
    assert LinearSizeFinder() != 42


@pytest.mark.parametrize("module", LINEAR_MODULES)
def test_linear_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert LinearSizeFinder().find_in_features(module.module) == module.in_features


def test_linear_size_finder_find_in_features_incorrect() -> None:
    size_finder = LinearSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute in_features"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", LINEAR_MODULES)
def test_linear_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert LinearSizeFinder().find_out_features(module.module) == module.out_features


def test_linear_size_finder_find_out_features_incorrect() -> None:
    size_finder = LinearSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute out_features"):
        size_finder.find_out_features(module)
