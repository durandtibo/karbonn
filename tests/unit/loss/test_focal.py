from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose
from coola.utils.tensor import get_available_devices
from torch import nn

from karbonn import BinaryFocalLoss, binary_focal_loss

SIZES = (1, 2)
TOLERANCE = 1e-6


####################################
#     Tests of BinaryFocalLoss     #
####################################


def test_binary_focal_loss_str() -> None:
    assert str(BinaryFocalLoss(loss=nn.BCELoss(reduction="none"))).startswith("BinaryFocalLoss(")


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_binary_focal_loss_valid_alpha(alpha: float) -> None:
    assert BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=alpha)._alpha == alpha


@pytest.mark.parametrize("alpha", [-1, 2])
def test_binary_focal_loss_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="Incorrect parameter alpha"):
        BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=alpha)


@pytest.mark.parametrize("gamma", [0, 0.5, 1])
def test_binary_focal_loss_valid_gamma(gamma: float) -> None:
    assert BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), gamma=gamma)._gamma == gamma


@pytest.mark.parametrize("gamma", [-1, -0.5])
def test_binary_focal_loss_invalid_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match="Incorrect parameter gamma"):
        BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), gamma=gamma)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("num_classes", SIZES)
def test_binary_focal_loss_forward_with_reduction(
    device: str, reduction: str, batch_size: int, num_classes: int
) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(
        loss=nn.BCELoss(reduction="none"), alpha=0.25, reduction=reduction
    ).to(device=device)
    loss = criterion(
        torch.rand(batch_size, num_classes, device=device),
        torch.ones(batch_size, num_classes, device=device),
    )
    assert loss.shape == ()


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("num_classes", SIZES)
def test_binary_focal_loss_forward_without_reduction(
    device: str, batch_size: int, num_classes: int
) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=0.25, reduction="none").to(
        device=device
    )
    loss = criterion(
        torch.rand(batch_size, num_classes, device=device),
        torch.ones(batch_size, num_classes, device=device),
    )
    assert loss.shape == (batch_size, num_classes)


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_without_reduction_alpha_0_25(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=0.25, reduction="none").to(
        device=device
    )
    loss = criterion(
        torch.tensor([[0.9, 0.2, 0.8], [0.7, 0.4, 0.6]], device=device, dtype=torch.float),
        torch.tensor([[1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float),
    )
    assert objects_are_allclose(
        torch.tensor(
            [
                [0.00026340128914456557, 0.006694306539426288, 0.002231435513142096],
                [0.4424600055897814, 0.08246616586867395, 0.24739849760602187],
            ],
            device=device,
            dtype=torch.float,
        ),
        loss,
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_without_reduction_alpha_0_5(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=0.5, reduction="none").to(
        device=device
    )
    loss = criterion(
        torch.tensor([[0.9, 0.2, 0.8], [0.7, 0.4, 0.6]], device=device, dtype=torch.float),
        torch.tensor([[1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float),
    )
    assert objects_are_allclose(
        torch.tensor(
            [
                [0.0005268025782891311, 0.004462871026284192, 0.004462871026284192],
                [0.29497333705985423, 0.1649323317373479, 0.1649323317373479],
            ],
            device=device,
            dtype=torch.float,
        ),
        loss,
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_without_reduction_gamma_1(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(
        loss=nn.BCELoss(reduction="none"), alpha=0.25, gamma=1, reduction="none"
    ).to(device=device)
    loss = criterion(
        torch.tensor([[0.9, 0.2, 0.8], [0.7, 0.4, 0.6]], device=device, dtype=torch.float),
        torch.tensor([[1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float),
    )
    assert objects_are_allclose(
        torch.tensor(
            [
                [0.0026340128914456563, 0.03347153269713145, 0.011157177565710483],
                [0.6320857222711163, 0.13744360978112324, 0.41233082934336973],
            ],
            device=device,
            dtype=torch.float,
        ),
        loss,
        atol=TOLERANCE,
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_sum_reduction_alpha_0_25(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=0.25, reduction="sum").to(
        device=device
    )
    loss = criterion(
        torch.tensor([[0.9, 0.2, 0.8], [0.7, 0.4, 0.6]], device=device, dtype=torch.float),
        torch.tensor([[1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float),
    )
    assert objects_are_allclose(
        loss, torch.tensor(0.7815138124061901, device=device), atol=TOLERANCE
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_binary_focal_loss_forward_mean_reduction_alpha_0_25(device: str) -> None:
    device = torch.device(device)
    criterion = BinaryFocalLoss(loss=nn.BCELoss(reduction="none"), alpha=0.25, reduction="mean").to(
        device=device
    )
    loss = criterion(
        torch.tensor([[0.9, 0.2, 0.8], [0.7, 0.4, 0.6]], device=device, dtype=torch.float),
        torch.tensor([[1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float),
    )
    assert objects_are_allclose(
        loss, torch.tensor(0.13025230206769836, device=device), atol=TOLERANCE
    )


def test_binary_focal_loss_forward_incorrect_reduction() -> None:
    with pytest.raises(ValueError, match="Incorrect reduction: incorrect reduction."):
        BinaryFocalLoss(
            loss=nn.BCELoss(reduction="none"), alpha=0.25, reduction="incorrect reduction"
        )


######################################
#     Tests of binary_focal_loss     #
######################################


def test_binary_focal_loss_logits_false() -> None:
    criterion = binary_focal_loss()
    assert isinstance(criterion, BinaryFocalLoss)
    assert isinstance(criterion.loss, nn.BCELoss)


def test_binary_focal_loss_logits_true() -> None:
    criterion = binary_focal_loss(logits=True)
    assert isinstance(criterion, BinaryFocalLoss)
    assert isinstance(criterion.loss, nn.BCEWithLogitsLoss)
