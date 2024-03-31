from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.utils.tensor import get_available_devices
from torch import nn

from karbonn import RelativeLoss, RelativeMSELoss, RelativeSmoothL1Loss
from karbonn.functional.loss.relative import classical_relative_indicator

##################################
#     Tests for RelativeLoss     #
##################################


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(nn.MSELoss(reduction="none"), eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(nn.MSELoss(reduction="none"), reduction="sum", eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(nn.MSELoss(reduction="none"), reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction:"):
        RelativeLoss(nn.MSELoss(reduction="none"), reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        nn.MSELoss(reduction="none"), indicator="arithmetical_mean", reduction="none"
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        nn.MSELoss(reduction="none"), indicator="classical_relative", reduction="none"
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        nn.MSELoss(reduction="none"), indicator="reversed_relative", reduction="none"
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor([[2.0, 0.0, 1e8], [12.0, 3.2, 0.0]], device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_callable(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeLoss(
        nn.MSELoss(reduction="none"), indicator=classical_relative_indicator, reduction="none"
    )
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_incorrect_shapes() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    criterion = RelativeLoss(nn.MSELoss())
    with pytest.raises(RuntimeError, match="loss .* and target .* shapes do not match"):
        criterion(prediction=prediction, target=target)


#####################################
#     Tests for RelativeMSELoss     #
#####################################


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(reduction="sum", eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_mse_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction:"):
        RelativeMSELoss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator="arithmetical_mean", reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator="classical_relative", reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator="reversed_relative", reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor([[2.0, 0.0, 1e8], [12.0, 3.2, 0.0]], device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_indicator_callable(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeMSELoss(indicator=classical_relative_indicator, reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


##########################################
#     Tests for RelativeSmoothL1Loss     #
##########################################


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_allclose(loss, torch.tensor(25000.970703125, device=device), rtol=1e-5)


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(reduction="sum", eps=1e-5)
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(loss, torch.tensor(150005.828125, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[1.5e8, 0.0, 0.5], [1.8333333333333333, 3.5, 0.0]], device=device)
    )


def test_relative_smooth_l1_loss_reduction_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction:"):
        RelativeSmoothL1Loss(reduction="incorrect")


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(indicator="arithmetical_mean", reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss,
        torch.tensor(
            [[1.5, 0.0, 1.0], [1.8333333333333333, 1.1666666666666667, 0.0]], device=device
        ),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(indicator="classical_relative", reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[1.5e8, 0.0, 0.5], [1.8333333333333333, 3.5, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(indicator="reversed_relative", reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[0.75, 0.0, 0.5e8], [1.8333333333333333, 0.7, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_indicator_callable(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    criterion = RelativeSmoothL1Loss(indicator=classical_relative_indicator, reduction="none")
    loss = criterion(prediction=prediction, target=target)
    assert objects_are_equal(
        loss, torch.tensor([[1.5e8, 0.0, 0.5], [1.8333333333333333, 3.5, 0.0]], device=device)
    )
