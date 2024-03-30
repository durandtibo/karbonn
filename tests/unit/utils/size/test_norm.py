from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import BatchNormSizeFinder, GroupNormSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

BATCH_NORM_MODULES = [
    ModuleSizes(module=nn.BatchNorm1d(num_features=4), in_features=[4], out_features=[4]),
    ModuleSizes(module=nn.BatchNorm1d(num_features=6), in_features=[6], out_features=[6]),
    ModuleSizes(module=nn.BatchNorm2d(num_features=4), in_features=[4], out_features=[4]),
    ModuleSizes(module=nn.BatchNorm2d(num_features=6), in_features=[6], out_features=[6]),
    ModuleSizes(module=nn.BatchNorm3d(num_features=4), in_features=[4], out_features=[4]),
    ModuleSizes(module=nn.BatchNorm3d(num_features=6), in_features=[6], out_features=[6]),
    ModuleSizes(module=nn.SyncBatchNorm(num_features=4), in_features=[4], out_features=[4]),
    ModuleSizes(module=nn.SyncBatchNorm(num_features=6), in_features=[6], out_features=[6]),
]

GROUP_NORM_MODULES = [
    ModuleSizes(
        module=nn.GroupNorm(num_groups=1, num_channels=4), in_features=[4], out_features=[4]
    ),
    ModuleSizes(
        module=nn.GroupNorm(num_groups=2, num_channels=4), in_features=[4], out_features=[4]
    ),
    ModuleSizes(
        module=nn.GroupNorm(num_groups=1, num_channels=8), in_features=[8], out_features=[8]
    ),
    ModuleSizes(
        module=nn.GroupNorm(num_groups=2, num_channels=8), in_features=[8], out_features=[8]
    ),
]


#########################################
#     Tests for BatchNormSizeFinder     #
#########################################


def test_batch_norm_size_finder_repr() -> None:
    assert repr(BatchNormSizeFinder()).startswith("BatchNormSizeFinder(")


def test_batch_norm_size_finder_str() -> None:
    assert str(BatchNormSizeFinder()).startswith("BatchNormSizeFinder(")


def test_batch_norm_size_finder_eq_true() -> None:
    assert BatchNormSizeFinder() == BatchNormSizeFinder()


def test_batch_norm_size_finder_eq_false() -> None:
    assert BatchNormSizeFinder() != 42


@pytest.mark.parametrize("module", BATCH_NORM_MODULES)
def test_batch_norm_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert BatchNormSizeFinder().find_in_features(module.module) == module.in_features


def test_batch_norm_size_finder_find_in_features_incorrect() -> None:
    size_finder = BatchNormSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute num_features"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", BATCH_NORM_MODULES)
def test_batch_norm_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert BatchNormSizeFinder().find_out_features(module.module) == module.out_features


def test_batch_norm_size_finder_find_out_features_incorrect() -> None:
    size_finder = BatchNormSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute num_features"):
        size_finder.find_out_features(module)


#########################################
#     Tests for GroupNormSizeFinder     #
#########################################


def test_group_norm_size_finder_repr() -> None:
    assert repr(GroupNormSizeFinder()).startswith("GroupNormSizeFinder(")


def test_group_norm_size_finder_str() -> None:
    assert str(GroupNormSizeFinder()).startswith("GroupNormSizeFinder(")


def test_group_norm_size_finder_eq_true() -> None:
    assert GroupNormSizeFinder() == GroupNormSizeFinder()


def test_group_norm_size_finder_eq_false() -> None:
    assert GroupNormSizeFinder() != 42


@pytest.mark.parametrize("module", GROUP_NORM_MODULES)
def test_group_norm_size_finder_find_in_features(module: ModuleSizes) -> None:
    assert GroupNormSizeFinder().find_in_features(module.module) == module.in_features


def test_group_norm_size_finder_find_in_features_incorrect() -> None:
    size_finder = GroupNormSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute num_channels"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", GROUP_NORM_MODULES)
def test_group_norm_size_finder_find_out_features(module: ModuleSizes) -> None:
    assert GroupNormSizeFinder().find_out_features(module.module) == module.out_features


def test_group_norm_size_finder_find_out_features_incorrect() -> None:
    size_finder = GroupNormSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute num_channels"):
        size_finder.find_out_features(module)
