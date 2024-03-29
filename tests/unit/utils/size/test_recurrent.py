from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import AutoSizeFinder, RecurrentSizeFinder, SizeFinderConfig
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes


@pytest.fixture(scope="module")
def config() -> SizeFinderConfig:
    return SizeFinderConfig(size_finder=AutoSizeFinder())


RECURRENT_MODULES = [
    ModuleSizes(module=nn.RNN(input_size=4, hidden_size=6), in_features=[4], out_features=[6]),
    ModuleSizes(module=nn.RNN(input_size=2, hidden_size=8), in_features=[2], out_features=[8]),
    ModuleSizes(module=nn.GRU(input_size=4, hidden_size=6), in_features=[4], out_features=[6]),
    ModuleSizes(module=nn.GRU(input_size=2, hidden_size=8), in_features=[2], out_features=[8]),
    ModuleSizes(module=nn.LSTM(input_size=4, hidden_size=6), in_features=[4], out_features=[6]),
    ModuleSizes(module=nn.LSTM(input_size=2, hidden_size=8), in_features=[2], out_features=[8]),
]


#########################################
#     Tests for RecurrentSizeFinder     #
#########################################


def test_recurrent_size_finder_repr() -> None:
    assert repr(RecurrentSizeFinder()).startswith("RecurrentSizeFinder(")


def test_recurrent_size_finder_str() -> None:
    assert str(RecurrentSizeFinder()).startswith("RecurrentSizeFinder(")


def test_recurrent_size_finder_eq_true() -> None:
    assert RecurrentSizeFinder() == RecurrentSizeFinder()


def test_recurrent_size_finder_eq_false() -> None:
    assert RecurrentSizeFinder() != 42


@pytest.mark.parametrize("module", RECURRENT_MODULES)
def test_recurrent_size_finder_find_in_features(
    module: ModuleSizes, config: SizeFinderConfig
) -> None:
    assert RecurrentSizeFinder().find_in_features(module.module, config) == module.in_features


def test_recurrent_size_finder_find_in_features_incorrect(config: SizeFinderConfig) -> None:
    size_finder = RecurrentSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute input_size"):
        size_finder.find_in_features(module, config)


@pytest.mark.parametrize("module", RECURRENT_MODULES)
def test_recurrent_size_finder_find_out_features(
    module: ModuleSizes, config: SizeFinderConfig
) -> None:
    assert RecurrentSizeFinder().find_out_features(module.module, config) == module.out_features


def test_recurrent_size_finder_find_out_features_incorrect(config: SizeFinderConfig) -> None:
    size_finder = RecurrentSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute hidden_size"):
        size_finder.find_out_features(module, config)
