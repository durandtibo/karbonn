from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from torch import nn

from karbonn.utils.size import AutoSizeFinder, BaseSizeFinder, LinearSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.test_attention import MULTIHEAD_ATTENTION_MODULES
from tests.unit.utils.size.test_conv import CONVOLUTION_MODULES
from tests.unit.utils.size.test_embedding import EMBEDDING_MODULES
from tests.unit.utils.size.test_linear import BILINEAR_MODULES, LINEAR_MODULES
from tests.unit.utils.size.test_list import MODULE_LIST_MODULES
from tests.unit.utils.size.test_recurrent import RECURRENT_MODULES
from tests.unit.utils.size.test_sequential import SEQUENTIAL_MODULES
from tests.unit.utils.size.test_transformer import (
    TRANFORMER_LAYERS_MODULES,
    TRANFORMER_MODULES,
)
from tests.unit.utils.size.test_unknown import UNKNOWN_MODULES

if TYPE_CHECKING:
    from tests.unit.utils.size.utils import ModuleSizes

MODULES = (
    LINEAR_MODULES
    + BILINEAR_MODULES
    + CONVOLUTION_MODULES
    + EMBEDDING_MODULES
    + MODULE_LIST_MODULES
    + MULTIHEAD_ATTENTION_MODULES
    + RECURRENT_MODULES
    + SEQUENTIAL_MODULES
    + TRANFORMER_LAYERS_MODULES
    + TRANFORMER_MODULES
)


####################################
#     Tests for AutoSizeFinder     #
####################################


def test_auto_size_finder_repr() -> None:
    assert repr(AutoSizeFinder()).startswith("AutoSizeFinder(")


def test_auto_size_finder_str() -> None:
    assert str(AutoSizeFinder()).startswith("AutoSizeFinder(")


@pytest.mark.parametrize("module", MODULES)
def test_auto_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert AutoSizeFinder().find_in_features(module.module) == module.in_features


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_auto_size_finder_find_in_features_incorrect(
    module: ModuleSizes,
) -> None:
    size_finder = AutoSizeFinder()
    with pytest.raises(SizeNotFoundError, match="cannot find the input feature sizes of"):
        size_finder.find_in_features(module.module)


@pytest.mark.parametrize("module", MODULES)
def test_auto_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert AutoSizeFinder().find_out_features(module.module) == module.out_features


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_auto_size_finder_find_out_features_incorrect(
    module: ModuleSizes,
) -> None:
    size_finder = AutoSizeFinder()
    with pytest.raises(SizeNotFoundError, match="cannot find the output feature sizes of"):
        size_finder.find_out_features(module.module)


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


def test_auto_size_finder_find_size_finder() -> None:
    assert AutoSizeFinder.find_size_finder(nn.Linear) == LinearSizeFinder()


def test_auto_size_finder_find_size_finder_missing() -> None:
    with pytest.raises(TypeError, match="Incorrect module type:"):
        AutoSizeFinder.find_size_finder(str)


# def test_auto_size_finder_registered_size_finders() -> None:
#     assert len(SizeFinder.registry) >= 1
#     assert isinstance(SizeFinder.registry["random"], RandomSizeFinder)
