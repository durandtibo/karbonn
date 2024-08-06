from __future__ import annotations

import pytest
from torch import nn

from karbonn.modules import ExU
from karbonn.utils.size import BilinearSizeFinder, LinearSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

LINEAR_MODULES = [
    ModuleSizes(module=nn.Linear(4, 6), in_features=[4], out_features=[6]),
    ModuleSizes(module=nn.Linear(2, 1), in_features=[2], out_features=[1]),
    ModuleSizes(module=ExU(4, 6), in_features=[4], out_features=[6]),
    ModuleSizes(module=ExU(2, 1), in_features=[2], out_features=[1]),
]

BILINEAR_MODULES = [
    ModuleSizes(
        module=nn.Bilinear(in1_features=4, in2_features=6, out_features=7),
        in_features=[4, 6],
        out_features=[7],
    ),
    ModuleSizes(
        module=nn.Bilinear(in1_features=2, in2_features=1, out_features=3),
        in_features=[2, 1],
        out_features=[3],
    ),
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
def test_linear_size_finder_find_out_features(
    module: ModuleSizes,
) -> None:
    assert LinearSizeFinder().find_out_features(module.module) == module.out_features


def test_linear_size_finder_find_out_features_incorrect() -> None:
    size_finder = LinearSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute out_features"):
        size_finder.find_out_features(module)


########################################
#     Tests for BilinearSizeFinder     #
########################################


def test_bilinear_size_finder_repr() -> None:
    assert repr(BilinearSizeFinder()).startswith("BilinearSizeFinder(")


def test_bilinear_size_finder_str() -> None:
    assert str(BilinearSizeFinder()).startswith("BilinearSizeFinder(")


def test_bilinear_size_finder_eq_true() -> None:
    assert BilinearSizeFinder() == BilinearSizeFinder()


def test_bilinear_size_finder_eq_false() -> None:
    assert BilinearSizeFinder() != 42


@pytest.mark.parametrize("module", BILINEAR_MODULES)
def test_bilinear_size_finder_find_in_features(
    module: ModuleSizes,
) -> None:
    assert BilinearSizeFinder().find_in_features(module.module) == module.in_features


def test_bilinear_size_finder_find_in_features_incorrect() -> None:
    size_finder = BilinearSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute in1_features"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", BILINEAR_MODULES)
def test_bilinear_size_finder_find_out_features(
    module: ModuleSizes,
) -> None:
    assert BilinearSizeFinder().find_out_features(module.module) == module.out_features


def test_bilinear_size_finder_find_out_features_incorrect() -> None:
    size_finder = BilinearSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match="module .* does not have attribute out_features"):
        size_finder.find_out_features(module)
