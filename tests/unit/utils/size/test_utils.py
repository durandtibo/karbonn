from __future__ import annotations

from coola import objects_are_equal
from torch import nn

from karbonn import ExU
from karbonn.utils.size import (
    BilinearSizeFinder,
    LinearSizeFinder,
    UnknownSizeFinder,
    get_karbonn_size_finders,
    get_size_finders,
    get_torch_size_finders,
)

######################################
#     Tests for get_size_finders     #
######################################


def test_get_size_finders() -> None:
    assert objects_are_equal(
        get_size_finders(),
        {
            nn.Module: UnknownSizeFinder(),
            nn.Linear: LinearSizeFinder(),
            nn.Bilinear: BilinearSizeFinder(),
            ExU: LinearSizeFinder(),
        },
    )


##############################################
#     Tests for get_karbonn_size_finders     #
##############################################


def test_get_karbonn_size_finders() -> None:
    assert objects_are_equal(get_karbonn_size_finders(), {ExU: LinearSizeFinder()})


############################################
#     Tests for get_torch_size_finders     #
############################################


def test_get_torch_size_finders() -> None:
    assert objects_are_equal(
        get_torch_size_finders(),
        {
            nn.Module: UnknownSizeFinder(),
            nn.Linear: LinearSizeFinder(),
            nn.Bilinear: BilinearSizeFinder(),
        },
    )
