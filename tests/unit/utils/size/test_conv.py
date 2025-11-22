from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import ConvolutionSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

CONVOLUTION_MODULES = [
    ModuleSizes(
        module=nn.Conv1d(in_channels=4, out_channels=6, kernel_size=1),
        in_features=[4],
        out_features=[6],
    ),
    ModuleSizes(
        module=nn.Conv2d(in_channels=4, out_channels=6, kernel_size=1),
        in_features=[4],
        out_features=[6],
    ),
    ModuleSizes(
        module=nn.Conv3d(in_channels=4, out_channels=6, kernel_size=1),
        in_features=[4],
        out_features=[6],
    ),
    ModuleSizes(
        module=nn.ConvTranspose1d(in_channels=4, out_channels=6, kernel_size=1),
        in_features=[4],
        out_features=[6],
    ),
    ModuleSizes(
        module=nn.ConvTranspose2d(in_channels=4, out_channels=6, kernel_size=1),
        in_features=[4],
        out_features=[6],
    ),
    ModuleSizes(
        module=nn.ConvTranspose3d(in_channels=4, out_channels=6, kernel_size=1),
        in_features=[4],
        out_features=[6],
    ),
]


###########################################
#     Tests for ConvolutionSizeFinder     #
###########################################


def test_convolution_size_finder_repr() -> None:
    assert repr(ConvolutionSizeFinder()).startswith("ConvolutionSizeFinder(")


def test_convolution_size_finder_str() -> None:
    assert str(ConvolutionSizeFinder()).startswith("ConvolutionSizeFinder(")


def test_convolution_size_finder_eq_true() -> None:
    assert ConvolutionSizeFinder() == ConvolutionSizeFinder()


def test_convolution_size_finder_eq_false() -> None:
    assert ConvolutionSizeFinder() != 42


@pytest.mark.parametrize("module", CONVOLUTION_MODULES)
def test_convolution_size_finder_find_in_features(
    module: ModuleSizes,
) -> None:
    assert ConvolutionSizeFinder().find_in_features(module.module) == module.in_features


def test_convolution_size_finder_find_in_features_incorrect() -> None:
    size_finder = ConvolutionSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute in_channels"):
        size_finder.find_in_features(module)


@pytest.mark.parametrize("module", CONVOLUTION_MODULES)
def test_convolution_size_finder_find_out_features(
    module: ModuleSizes,
) -> None:
    assert ConvolutionSizeFinder().find_out_features(module.module) == module.out_features


def test_convolution_size_finder_find_out_features_incorrect() -> None:
    size_finder = ConvolutionSizeFinder()
    module = nn.Identity()
    with pytest.raises(SizeNotFoundError, match=r"module .* does not have attribute out_channels"):
        size_finder.find_out_features(module)
