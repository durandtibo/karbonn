from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.functional import relative_loss
from karbonn.functional.loss.relative import (
    RelativeIndicatorRegistry,
    arithmetical_mean_indicator,
    classical_relative_indicator,
    reversed_relative_indicator,
)

###################################
#     Tests for relative_loss     #
###################################


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        eps=1e-5,
    )
    assert objects_are_equal(loss, torch.tensor(66671.5, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_sum(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        reduction="sum",
        eps=1e-5,
    )
    assert objects_are_equal(loss, torch.tensor(400029.0, device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_reduction_none(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        reduction="none",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_reduction_incorrect() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    with pytest.raises(ValueError, match="Incorrect reduction:"):
        relative_loss(
            loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
            prediction=prediction,
            target=target,
            reduction="incorrect",
        )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_arithmetical_mean(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        reduction="none",
        indicator="arithmetical_mean",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4.0, 0.0, 2.0], [12.0, 5.333333333333333, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_classical_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        reduction="none",
        indicator="classical_relative",
    )
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_reversed_relative(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        reduction="none",
        indicator="reversed_relative",
    )
    assert objects_are_equal(loss, torch.tensor([[2.0, 0.0, 1e8], [12.0, 3.2, 0.0]], device=device))


@pytest.mark.parametrize("device", get_available_devices())
def test_relative_loss_indicator_callable(device: str) -> None:
    prediction = torch.tensor(
        [[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True
    )
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device)
    loss = relative_loss(
        loss=torch.nn.functional.mse_loss(prediction, target, reduction="none"),
        prediction=prediction,
        target=target,
        reduction="none",
        indicator=classical_relative_indicator,
    )
    assert objects_are_equal(
        loss, torch.tensor([[4e8, 0.0, 1.0], [12.0, 16.0, 0.0]], device=device)
    )


def test_relative_loss_incorrect_shapes() -> None:
    prediction = torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]])
    with pytest.raises(RuntimeError, match="loss .* and target .* shapes do not match"):
        relative_loss(
            loss=torch.nn.functional.mse_loss(prediction, target),
            prediction=prediction,
            target=target,
        )


#################################################
#     Tests for arithmetical_mean_indicator     #
#################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_arithmetical_mean_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        arithmetical_mean_indicator(
            torch.tensor([[0.0, 1.0, -1.0], [3.0, 1.0, -1.0]], device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True),
        ),
        torch.tensor([[1.0, 1.0, 0.5], [3.0, 3.0, 1.0]], device=device),
    )


##################################################
#     Tests for classical_relative_indicator     #
##################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_classical_relative_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        classical_relative_indicator(
            torch.ones(2, 3, device=device),
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device, requires_grad=True),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )


#################################################
#     Tests for reversed_relative_indicator     #
#################################################


@pytest.mark.parametrize("device", get_available_devices())
def test_reversed_relative_indicator(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        reversed_relative_indicator(
            torch.tensor([[-2.0, 1.0, 0.0], [-3.0, 5.0, -1.0]], device=device),
            torch.ones(2, 3, device=device, requires_grad=True),
        ),
        torch.tensor([[2.0, 1.0, 0.0], [3.0, 5.0, 1.0]], device=device),
    )


###############################################
#     Tests for RelativeIndicatorRegistry     #
###############################################


def test_relative_indicator_registry_repr() -> None:
    assert repr(RelativeIndicatorRegistry()).startswith("RelativeIndicatorRegistry(")


def test_relative_indicator_registry_str() -> None:
    assert str(RelativeIndicatorRegistry()).startswith("RelativeIndicatorRegistry(")


@patch.dict(RelativeIndicatorRegistry.registry, {}, clear=True)
def test_relative_indicator_registry_add_indicator() -> None:
    registry = RelativeIndicatorRegistry()
    registry.add_indicator("name", classical_relative_indicator)
    assert registry.registry == {"name": classical_relative_indicator}


@patch.dict(RelativeIndicatorRegistry.registry, {}, clear=True)
def test_relative_indicator_registry_add_indicator_exist_ok_false() -> None:
    registry = RelativeIndicatorRegistry()
    registry.add_indicator("name", arithmetical_mean_indicator)
    with pytest.raises(RuntimeError, match="An indicator .* is already registered for the name"):
        registry.add_indicator("name", classical_relative_indicator)


@patch.dict(RelativeIndicatorRegistry.registry, {}, clear=True)
def test_relative_indicator_registry_add_indicator_exist_ok_true() -> None:
    registry = RelativeIndicatorRegistry()
    registry.add_indicator("name", arithmetical_mean_indicator)
    registry.add_indicator("name", classical_relative_indicator, exist_ok=True)
    assert registry.registry == {"name": classical_relative_indicator}


def test_relative_indicator_registry_available_indicators() -> None:
    assert RelativeIndicatorRegistry().available_indicators() == (
        "arithmetical_mean",
        "classical_relative",
        "reversed_relative",
    )


def test_relative_indicator_registry_find_indicator() -> None:
    assert (
        RelativeIndicatorRegistry.find_indicator("classical_relative")
        == classical_relative_indicator
    )


def test_relative_indicator_registry_find_indicator_missing() -> None:
    with pytest.raises(RuntimeError, match="Incorrect name:"):
        RelativeIndicatorRegistry.find_indicator("missing")


def test_relative_indicator_registry_has_indicator_true() -> None:
    assert RelativeIndicatorRegistry.has_indicator("classical_relative")


def test_relative_indicator_registry_has_indicator_false() -> None:
    assert not RelativeIndicatorRegistry.has_indicator("missing")
