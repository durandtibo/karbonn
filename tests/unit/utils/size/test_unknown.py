from __future__ import annotations

import pytest
from torch import nn

from karbonn.utils.size import UnknownSizeFinder
from karbonn.utils.size.base import SizeNotFoundError
from tests.unit.utils.size.utils import ModuleSizes

UNKNOWN_MODULES = [
    ModuleSizes(module=nn.CELU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Dropout(p=0.2), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Dropout1d(p=0.2), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Dropout2d(p=0.2), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Dropout3d(p=0.2), in_features=[], out_features=[]),
    ModuleSizes(module=nn.ELU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.Identity(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.LeakyReLU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.LogSigmoid(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.MaxPool1d(kernel_size=3), in_features=[], out_features=[]),
    ModuleSizes(module=nn.MaxPool2d(kernel_size=3), in_features=[], out_features=[]),
    ModuleSizes(module=nn.MaxPool3d(kernel_size=3), in_features=[], out_features=[]),
    ModuleSizes(module=nn.MaxUnpool1d(kernel_size=3), in_features=[], out_features=[]),
    ModuleSizes(module=nn.MaxUnpool2d(kernel_size=3), in_features=[], out_features=[]),
    ModuleSizes(module=nn.MaxUnpool3d(kernel_size=3), in_features=[], out_features=[]),
    ModuleSizes(module=nn.PReLU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.PReLU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.ReLU(), in_features=[], out_features=[]),
    ModuleSizes(module=nn.SiLU(), in_features=[], out_features=[]),
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
    module: ModuleSizes,
) -> None:
    size_finder = UnknownSizeFinder()
    with pytest.raises(SizeNotFoundError, match=r"cannot find the input feature sizes of"):
        size_finder.find_in_features(module.module)


@pytest.mark.parametrize("module", UNKNOWN_MODULES)
def test_unknown_size_finder_find_out_features(
    module: ModuleSizes,
) -> None:
    size_finder = UnknownSizeFinder()
    with pytest.raises(SizeNotFoundError, match=r"cannot find the output feature sizes of"):
        size_finder.find_out_features(module.module)
