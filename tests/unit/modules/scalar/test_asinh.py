from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn.modules import AsinhScalarEncoder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SIZES = (1, 2)

########################################
#     Tests for AsinhScalarEncoder     #
########################################

MODULE_CONSTRUCTORS: tuple[Callable, ...] = (
    AsinhScalarEncoder.create_rand_scale,
    AsinhScalarEncoder.create_linspace_scale,
    AsinhScalarEncoder.create_logspace_scale,
)


def test_asinh_scalar_encoder_str() -> None:
    assert str(
        AsinhScalarEncoder.create_rand_scale(dim=5, min_scale=0.1, max_scale=10.0)
    ).startswith("AsinhScalarEncoder(")


@pytest.mark.parametrize(
    "scale",
    [
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.float),
        [1.0, 2.0, 4.0],
        (1.0, 2.0, 4.0),
    ],
)
def test_asinh_scalar_encoder_scale(scale: torch.Tensor | Sequence[float]) -> None:
    assert AsinhScalarEncoder(scale).scale.equal(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float))


def test_asinh_scalar_encoder_input_size() -> None:
    assert (
        AsinhScalarEncoder.create_rand_scale(dim=5, min_scale=0.1, max_scale=10.0).input_size == 1
    )


def test_asinh_scalar_encoder_output_size() -> None:
    assert (
        AsinhScalarEncoder.create_rand_scale(dim=5, min_scale=0.1, max_scale=10.0).output_size == 5
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_forward_2d(
    device: str, batch_size: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 1, device=device))
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_forward_3d_batch_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, 1, device=device))
    assert out.shape == (batch_size, seq_len, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_forward_3d_seq_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, 1, device=device))
    assert out.shape == (seq_len, batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_backward(
    device: str, batch_size: int, learnable: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0, learnable=learnable).to(
        device=device
    )
    out = module(torch.rand(batch_size, 1, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_dim(dim: int, module_init: Callable) -> None:
    assert module_init(dim=dim, min_scale=0.1, max_scale=10.0).scale.shape == (dim,)


@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_dim_incorrect(dim: int, module_init: Callable) -> None:
    with pytest.raises(ValueError, match="dim has to be greater or equal to 1"):
        module_init(dim=dim, min_scale=0.1, max_scale=10.0)


@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_min_scale_incorrect(module_init: Callable) -> None:
    with pytest.raises(ValueError, match="min_scale has to be greater than 0"):
        module_init(dim=2, min_scale=0, max_scale=1)


@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_max_scale_incorrect(module_init: Callable) -> None:
    with pytest.raises(ValueError, match="max_scale has to be greater than min_scale"):
        module_init(dim=2, min_scale=0.1, max_scale=0.01)


@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_learnable(learnable: bool, module_init: Callable) -> None:
    assert (
        module_init(dim=2, min_scale=0.01, max_scale=1, learnable=learnable).scale.requires_grad
        == learnable
    )


@patch(
    "karbonn.modules.scalar.asinh.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),  # noqa: ARG005
)
def test_asinh_scalar_encoder_create_rand_scale() -> None:
    assert AsinhScalarEncoder.create_rand_scale(dim=3, min_scale=0.2, max_scale=1).scale.data.equal(
        torch.tensor([0.2, 0.6, 1.0])
    )


def test_asinh_scalar_encoder_create_linspace_scale() -> None:
    assert AsinhScalarEncoder.create_linspace_scale(
        dim=3, min_scale=0.2, max_scale=1
    ).scale.data.equal(torch.tensor([0.2, 0.6, 1.0]))


def test_asinh_scalar_encoder_create_logspace_scale() -> None:
    assert AsinhScalarEncoder.create_logspace_scale(
        dim=3, min_scale=0.01, max_scale=1
    ).scale.data.equal(torch.tensor([0.01, 0.1, 1.0]))


def test_asinh_scalar_encoder_forward_scale() -> None:
    module = AsinhScalarEncoder(scale=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float))
    assert module(torch.tensor([[-1], [0], [1]], dtype=torch.float)).allclose(
        torch.tensor(
            [
                [-0.881373587019543, -1.4436354751788103, -1.8184464592320668],
                [0.0, 0.0, 0.0],
                [0.881373587019543, 1.4436354751788103, 1.8184464592320668],
            ],
            dtype=torch.float,
        ),
    )
