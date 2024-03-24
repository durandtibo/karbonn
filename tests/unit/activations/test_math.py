from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn import Asinh, Exp, Expm1, Log, Log1p, Sinh

###########################
#     Tests for Asinh     #
###########################


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_forward(device: str) -> None:
    device = torch.device(device)
    module = Asinh().to(device=device)
    assert module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)).allclose(
        torch.tensor(
            [-1.4436354637145996, -0.8813735842704773, 0.0, 0.8813735842704773, 1.4436354637145996],
            dtype=torch.float,
            device=device,
        ),
    )


#########################
#     Tests for Exp     #
#########################


@pytest.mark.parametrize("device", get_available_devices())
def test_exp_forward(device: str) -> None:
    device = torch.device(device)
    module = Exp().to(device=device)
    assert module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)).allclose(
        torch.tensor(
            [0.1353352814912796, 0.3678794503211975, 1.0, 2.7182817459106445, 7.389056205749512],
            dtype=torch.float,
            device=device,
        ),
    )


###########################
#     Tests for Expm1     #
###########################


@pytest.mark.parametrize("device", get_available_devices())
def test_expm1_forward(device: str) -> None:
    device = torch.device(device)
    module = Expm1().to(device=device)
    assert module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)).allclose(
        torch.tensor(
            [-0.8646647334098816, -0.6321205496788025, 0.0, 1.718281865119934, 6.389056205749512],
            dtype=torch.float,
            device=device,
        ),
    )


#########################
#     Tests for Log     #
#########################


@pytest.mark.parametrize("device", get_available_devices())
def test_log_forward(device: str) -> None:
    device = torch.device(device)
    module = Log().to(device=device)
    assert module(torch.tensor([1.0, 2.0, 3.0], device=device)).allclose(
        torch.tensor(
            [0.0, 0.6931471805599453, 1.0986122886681098], dtype=torch.float, device=device
        ),
    )


###########################
#     Tests for Log1p     #
###########################


@pytest.mark.parametrize("device", get_available_devices())
def test_log1p_forward(device: str) -> None:
    device = torch.device(device)
    module = Log1p().to(device=device)
    assert module(torch.tensor([0.0, 1.0, 2.0], device=device)).allclose(
        torch.tensor(
            [0.0, 0.6931471805599453, 1.0986122886681098], dtype=torch.float, device=device
        ),
    )


##########################
#     Tests for Sinh     #
##########################


@pytest.mark.parametrize("device", get_available_devices())
def test_sinh_forward(device: str) -> None:
    device = torch.device(device)
    module = Sinh().to(device=device)
    assert module(torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)).allclose(
        torch.tensor(
            [-3.6268603801727295, -1.175201177597046, 0.0, 1.175201177597046, 3.6268603801727295],
            dtype=torch.float,
            device=device,
        ),
    )
