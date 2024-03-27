from __future__ import annotations

import torch
from torch import nn

from karbonn import ExpSin, Gaussian, Laplacian, MultiQuadratic, Quadratic
from karbonn.utils import is_loss_decreasing_with_adam

############################
#     Tests for ExpSin     #
############################


def test_exp_sin_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_adam(
        module=ExpSin(num_parameters=4),
        criterion=nn.MSELoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
    )


##############################
#     Tests for Gaussian     #
##############################


def test_gaussian_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_adam(
        module=Gaussian(num_parameters=4),
        criterion=nn.MSELoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
    )


###############################
#     Tests for Laplacian     #
###############################


def test_laplacian_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_adam(
        module=Laplacian(num_parameters=4),
        criterion=nn.MSELoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
    )


####################################
#     Tests for MultiQuadratic     #
####################################


def test_multi_quadratic_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_adam(
        module=MultiQuadratic(num_parameters=4),
        criterion=nn.MSELoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
    )


###############################
#     Tests for Quadratic     #
###############################


def test_quadratic_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_adam(
        module=Quadratic(num_parameters=4),
        criterion=nn.MSELoss(),
        feature=torch.randn(8, 4),
        target=torch.randn(8, 4),
    )
