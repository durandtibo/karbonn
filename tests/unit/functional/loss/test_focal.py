from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn.functional import sigmoid_focal_loss

SHAPES = [(2,), (2, 3), (2, 3, 4)]


########################################
#     Tests for sigmoid_focal_loss     #
########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.01132902921115496, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], device=device),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device),
    ).allclose(torch.tensor(0.3509341741420062, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
    ).allclose(torch.tensor(0.20943205058574677, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="sum",
    ).allclose(torch.tensor(1.2565922737121582, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        reduction="none",
    ).allclose(
        torch.tensor(
            [
                [0.005664513111161829, 0.1754670652368954, 0.5264012830471171],
                [0.016993545311148092, 0.005664513111161829, 0.5264012830471171],
            ],
            device=device,
        ),
    )


def test_sigmoid_focal_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction: incorrect"):
        sigmoid_focal_loss(torch.ones(2, 3), torch.ones(2, 3), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("shape", SHAPES)
def test_sigmoid_focal_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).allclose(torch.tensor(0.005664513111161829, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=0.5,
    ).allclose(torch.tensor(0.18113157834805724, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_no_alpha(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        alpha=-1.0,
    ).allclose(torch.tensor(0.3622631566961145, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_sigmoid_focal_loss_gamma_1(device: str) -> None:
    device = torch.device(device)
    assert sigmoid_focal_loss(
        torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]], device=device),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], device=device),
        gamma=1.0,
    ).allclose(torch.tensor(0.2975726744437273, device=device))
