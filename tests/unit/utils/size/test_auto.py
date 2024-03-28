from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from torch import nn

from karbonn.utils.size import (
    AutoSizeFinder,
    BaseSizeFinder,
    LinearSizeFinder,
    SizeFinderConfig,
)
from tests.unit.utils.size.test_linear import (
    BILINEAR_MODULES,
    LINEAR_MODULES,
    ModuleSizes,
)

MODULES = LINEAR_MODULES + BILINEAR_MODULES


@pytest.fixture(scope="module")
def config() -> SizeFinderConfig:
    return SizeFinderConfig(size_finder=AutoSizeFinder())


####################################
#     Tests for AutoSizeFinder     #
####################################


def test_auto_size_finder_repr() -> None:
    assert repr(AutoSizeFinder()).startswith("AutoSizeFinder(")


def test_auto_size_finder_str() -> None:
    assert str(AutoSizeFinder()).startswith("AutoSizeFinder(")


@pytest.mark.parametrize("module", MODULES)
def test_auto_size_finder_find_in_features(module: ModuleSizes, config: SizeFinderConfig) -> None:
    assert AutoSizeFinder().find_in_features(module.module, config) == module.in_features


@pytest.mark.parametrize("module", MODULES)
def test_auto_size_finder_find_out_features(module: ModuleSizes, config: SizeFinderConfig) -> None:
    assert AutoSizeFinder().find_out_features(module.module, config) == module.out_features


@patch.dict(AutoSizeFinder.registry, {}, clear=True)
def test_auto_size_finder_add_size_finder() -> None:
    assert len(AutoSizeFinder.registry) == 0
    AutoSizeFinder.add_size_finder(nn.Linear, LinearSizeFinder())
    assert AutoSizeFinder.registry[nn.Linear] == LinearSizeFinder()


@patch.dict(AutoSizeFinder.registry, {}, clear=True)
def test_auto_size_finder_add_size_finder_exist_ok_false() -> None:
    assert len(AutoSizeFinder.registry) == 0
    AutoSizeFinder.add_size_finder(nn.Linear, LinearSizeFinder())
    with pytest.raises(
        RuntimeError, match="A size finder .* is already registered for the module type"
    ):
        AutoSizeFinder.add_size_finder(nn.Linear, LinearSizeFinder())


@patch.dict(AutoSizeFinder.registry, {}, clear=True)
def test_auto_size_finder_add_size_finder_exist_ok_true() -> None:
    assert len(AutoSizeFinder.registry) == 0
    AutoSizeFinder.add_size_finder(nn.Linear, Mock(spec=BaseSizeFinder))
    AutoSizeFinder.add_size_finder(nn.Linear, LinearSizeFinder(), exist_ok=True)
    assert AutoSizeFinder.registry[nn.Linear] == LinearSizeFinder()


def test_auto_size_finder_has_size_finder_true() -> None:
    assert AutoSizeFinder.has_size_finder(nn.Linear)


def test_auto_size_finder_has_size_finder_false() -> None:
    assert not AutoSizeFinder.has_size_finder(str)


# def test_auto_size_finder_registered_size_finders() -> None:
#     assert len(SizeFinder.registry) >= 1
#     assert isinstance(SizeFinder.registry["random"], RandomSizeFinder)
