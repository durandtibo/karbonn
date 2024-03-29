from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import ModuleListSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

MODULE_LIST_MODULES = [
    ModuleSizes(module=nn.ModuleList([nn.Linear(4, 6)]), in_features=[4], out_features=[6]),
    ModuleSizes(
        module=nn.ModuleList([nn.Linear(4, 6), nn.ReLU(), nn.LSTM(input_size=4, hidden_size=6)]),
        in_features=[4],
        out_features=[6],
    ),
]


##########################################
#     Tests for ModuleListSizeFinder     #
##########################################


def test_sequential_size_finder_repr() -> None:
    assert repr(ModuleListSizeFinder()).startswith("ModuleListSizeFinder(")


def test_sequential_size_finder_str() -> None:
    assert str(ModuleListSizeFinder()).startswith("ModuleListSizeFinder(")


def test_sequential_size_finder_eq_true() -> None:
    assert ModuleListSizeFinder() == ModuleListSizeFinder()


def test_sequential_size_finder_eq_false() -> None:
    assert ModuleListSizeFinder() != 42


@pytest.mark.parametrize("module", MODULE_LIST_MODULES)
def test_sequential_size_finder_find_in_features(
    module: ModuleSizes,
) -> None:
    assert ModuleListSizeFinder().find_in_features(module.module) == module.in_features


def test_sequential_size_finder_find_in_features_no_sizes() -> None:
    size_finder = ModuleListSizeFinder()
    module = nn.ModuleList([nn.Identity(), nn.ReLU()])
    with pytest.raises(
        SizeNotFoundError,
        match="cannot find the input feature sizes because the indexed modules are not supported",
    ):
        size_finder.find_in_features(module)


def test_sequential_size_finder_find_in_features_different_sizes() -> None:
    size_finder = ModuleListSizeFinder()
    module = nn.ModuleList([nn.Linear(4, 6), nn.Linear(6, 4)])
    with pytest.raises(
        SizeNotFoundError,
        match=(
            "cannot find the input feature sizes because the indexed modules have different sizes"
        ),
    ):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", MODULE_LIST_MODULES)
def test_sequential_size_finder_find_out_features(
    module: ModuleSizes,
) -> None:
    assert ModuleListSizeFinder().find_out_features(module.module) == module.out_features


def test_sequential_size_finder_find_out_features_no_sizes() -> None:
    size_finder = ModuleListSizeFinder()
    module = nn.ModuleList([nn.Identity(), nn.ReLU()])
    with pytest.raises(
        SizeNotFoundError,
        match="cannot find the output feature sizes because the indexed modules are not supported",
    ):
        size_finder.find_out_features(module)


def test_sequential_size_finder_find_out_features_different_sizes() -> None:
    size_finder = ModuleListSizeFinder()
    module = nn.ModuleList([nn.Linear(4, 6), nn.Linear(6, 4)])
    with pytest.raises(
        SizeNotFoundError,
        match=(
            "cannot find the output feature sizes because the indexed modules have different sizes"
        ),
    ):
        size_finder.find_out_features(module)
