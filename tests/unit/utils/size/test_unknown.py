from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import AutoSizeFinder, SizeFinderConfig, UnknownSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.test_linear import ModuleSizes


@pytest.fixture(scope="module")
def config() -> SizeFinderConfig:
    return SizeFinderConfig(size_finder=AutoSizeFinder())


UNKNOWN_MODULES = [
    ModuleSizes(module=nn.Identity(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.ReLU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Sigmoid(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Tanh(), in_features=[], out_features=[]),
]


#######################################
#     Tests for UnknownSizeFinder     #
#######################################


def test_unknown_size_finder_repr() -> None:
    assert repr(UnknownSizeFinder()).startswith("UnknownSizeFinder(")


def test_unknown_size_finder_str() -> None:
    assert str(UnknownSizeFinder()).startswith("UnknownSizeFinder(")


def test_unknown_size_finder_eq_true() -> None:
    assert UnknownSizeFinder() == UnknownSizeFinder()


def test_unknown_size_finder_eq_false() -> None:
    assert UnknownSizeFinder() != 42


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_unknown_size_finder_find_in_features(
    module: ModuleSizes, config: SizeFinderConfig
) -> None:
    size_finder = UnknownSizeFinder()
    with pytest.raises(SizeNotFoundError, match="cannot find the input feature sizes of"):
        size_finder.find_in_features(module.module, config)


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_unknown_size_finder_find_out_features(
    module: ModuleSizes, config: SizeFinderConfig
) -> None:
    size_finder = UnknownSizeFinder()
    with pytest.raises(SizeNotFoundError, match="cannot find the output feature sizes of"):
        size_finder.find_out_features(module.module, config)
