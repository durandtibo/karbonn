from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import AutoSizeFinder, SequentialSizeFinder, SizeFinderConfig
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.test_linear import ModuleSizes


@pytest.fixture(scope="module")
def config() -> SizeFinderConfig:
    return SizeFinderConfig(size_finder=AutoSizeFinder())


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
    module: ModuleSizes, config: SizeFinderConfig
) -> None:
    assert SequentialSizeFinder().find_in_features(module.module, config) == module.in_features


def test_sequential_size_finder_find_in_features_incorrect(config: SizeFinderConfig) -> None:
    size_finder = SequentialSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute in_features"):
        size_finder.find_in_features(module, config)


@pytest.mark.parametrize("module", SEQUENTIAL_MODULES)
def test_sequential_size_finder_find_out_features(
    module: ModuleSizes, config: SizeFinderConfig
) -> None:
    assert SequentialSizeFinder().find_out_features(module.module, config) == module.out_features


def test_sequential_size_finder_find_out_features_incorrect(config: SizeFinderConfig) -> None:
    size_finder = SequentialSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute out_features"):
        size_finder.find_out_features(module, config)
