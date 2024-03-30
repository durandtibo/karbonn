from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn.functional import asinh_mse_loss

####################################
#     Tests for asinh_mse_loss     #
####################################


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).allclose(torch.tensor(3.107277599582784, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1
    ).allclose(torch.tensor(1.553638799791392, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="sum",
    ).allclose(torch.tensor(6.214555199165568, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="none",
    ).allclose(
        torch.tensor([[0.0, 3.107277599582784], [3.107277599582784, 0.0]], device=device),
    )


def test_asinh_mse_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match="incorrect is not a valid value for reduction"):
        asinh_mse_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_asinh_mse_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))
