from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from karbonn.utils.size import find_in_features, find_out_features
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.test_auto import MODULES
from tests.unit.utils.size.test_unknown import UNKNOWN_MODULES

if TYPE_CHECKING:
    from tests.unit.utils.size.utils import ModuleSizes

######################################
#     Tests for find_in_features     #
######################################


@pytest.mark.parametrize("module", MODULES)
def test_find_in_features(module: ModuleSizes) -> None:
    assert find_in_features(module.module) == module.in_features


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_find_in_features_incorrect(module: ModuleSizes) -> None:
    with pytest.raises(SizeNotFoundError, match="cannot find the input feature sizes of"):
        find_in_features(module.module)


#######################################
#     Tests for find_out_features     #
#######################################


@pytest.mark.parametrize("module", MODULES)
def test_find_out_features(module: ModuleSizes) -> None:
    assert find_out_features(module.module) == module.out_features


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_find_out_features_incorrect(module: ModuleSizes) -> None:
    with pytest.raises(SizeNotFoundError, match="cannot find the output feature sizes of"):
        find_out_features(module.module)
