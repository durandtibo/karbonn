from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import SequentialSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

SEQUENTIAL_MODULES = [
    ModuleSizes(module=nn.Sequential(nn.Linear(4, 6)), in_features=[4], out_features=[6]),
    ModuleSizes(
        module=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)),
        in_features=[4],
        out_features=[2],
    ),
    ModuleSizes(
        module=nn.Sequential(nn.Tanh(), nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2), nn.ReLU()),
        in_features=[4],
        out_features=[2],
    ),
]


##########################################
#     Tests for SequentialSizeFinder     #
##########################################


def test_sequential_size_finder_repr() -> None:
    assert repr(SequentialSizeFinder()).startswith("SequentialSizeFinder(")


def test_sequential_size_finder_str() -> None:
    assert str(SequentialSizeFinder()).startswith("SequentialSizeFinder(")


def test_sequential_size_finder_eq_true() -> None:
    assert SequentialSizeFinder() == SequentialSizeFinder()


def test_sequential_size_finder_eq_false() -> None:
    assert SequentialSizeFinder() != 42


@pytest.mark.parametrize("module", SEQUENTIAL_MODULES)
def test_sequential_size_finder_find_in_features(
    module: ModuleSizes,
) -> None:
    assert SequentialSizeFinder().find_in_features(module.module) == module.in_features


def test_sequential_size_finder_find_in_features_incorrect() -> None:
    size_finder = SequentialSizeFinder()
    module = nn.Sequential(nn.Identity(), nn.ReLU())
    with pytest.raises(
        SizeNotFoundError,
        match=r"cannot find the input feature sizes because the child modules are not supported",
    ):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", SEQUENTIAL_MODULES)
def test_sequential_size_finder_find_out_features(
    module: ModuleSizes,
) -> None:
    assert SequentialSizeFinder().find_out_features(module.module) == module.out_features


def test_sequential_size_finder_find_out_features_incorrect() -> None:
    size_finder = SequentialSizeFinder()
    module = nn.Sequential(nn.Identity(), nn.ReLU())
    with pytest.raises(
        SizeNotFoundError,
        match=r"cannot find the output feature sizes because the child modules are not supported",
    ):
        size_finder.find_out_features(module)
