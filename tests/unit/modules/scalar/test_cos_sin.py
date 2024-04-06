from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn import CosSinScalarEncoder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SIZES = (1, 2, 3)

#########################################
#     Tests for CosSinScalarEncoder     #
#########################################

COSSIN_MODULE_CONSTRUCTORS: tuple[Callable, ...] = (
    CosSinScalarEncoder.create_rand_frequency,
    CosSinScalarEncoder.create_linspace_frequency,
    CosSinScalarEncoder.create_logspace_frequency,
)


def test_cos_sin_scalar_encoder_str() -> None:
    assert str(
        CosSinScalarEncoder.create_rand_frequency(
            num_frequencies=3, min_frequency=0.1, max_frequency=10.0
        )
    ).startswith("CosSinScalarEncoder(")


@pytest.mark.parametrize(
    "frequency",
    [
        torch.tensor([1.0, 2.0, 4.0]),
        [1.0, 2.0, 4.0],
        (1.0, 2.0, 4.0),
    ],
)
@pytest.mark.parametrize(
    "phase_shift",
    [
        torch.tensor([1.0, 3.0, -2.0]),
        [1.0, 3.0, -2.0],
        (1.0, 3.0, -2.0),
    ],
)
def test_cos_sin_scalar_encoder_frequency_phase_shift(
    frequency: torch.Tensor | Sequence[float],
    phase_shift: torch.Tensor | Sequence[float],
) -> None:
    module = CosSinScalarEncoder(frequency, phase_shift)
    assert module.frequency.equal(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float))
    assert module.phase_shift.equal(torch.tensor([1.0, 3.0, -2.0], dtype=torch.float))


def test_cos_sin_scalar_encoder_frequency_phase_shift_incorrect_dim() -> None:
    with pytest.raises(ValueError, match="Incorrect number of dimensions for frequency"):
        CosSinScalarEncoder(torch.rand(1, 6), torch.rand(6))


def test_cos_sin_scalar_encoder_frequency_phase_shift_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="Incorrect shapes. The shape of frequency"):
        CosSinScalarEncoder(torch.rand(6), torch.rand(4))


def test_cos_sin_scalar_encoder_input_size() -> None:
    assert (
        CosSinScalarEncoder.create_rand_frequency(
            num_frequencies=3, min_frequency=0.1, max_frequency=10.0
        ).input_size
        == 1
    )


def test_cos_sin_scalar_encoder_output_size() -> None:
    assert (
        CosSinScalarEncoder.create_rand_frequency(
            num_frequencies=3, min_frequency=0.1, max_frequency=10.0
        ).output_size
        == 6
    )


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_forward_2d(
    device: str, batch_size: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 1, device=device))
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_forward_3d_batch_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, 1, device=device))
    assert out.shape == (batch_size, seq_len, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("seq_len", SIZES)
@pytest.mark.parametrize("mode", [True, False])
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_forward_3d_seq_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, 1, device=device))
    assert out.shape == (seq_len, batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("batch_size", SIZES)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_backward(
    device: str, batch_size: int, learnable: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(
        num_frequencies=5, min_frequency=0.1, max_frequency=10.0, learnable=learnable
    ).to(device=device)
    out = module(torch.rand(batch_size, 1, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@pytest.mark.parametrize("dim", SIZES)
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_dim(dim: int, module_init: Callable) -> None:
    module = module_init(num_frequencies=dim, min_frequency=0.1, max_frequency=10.0)
    assert module.frequency.shape == (dim * 2,)
    assert module.phase_shift.shape == (dim * 2,)


@pytest.mark.parametrize("num_frequencies", [0, -1])
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_num_frequencies_incorrect(
    num_frequencies: int, module_init: Callable
) -> None:
    with pytest.raises(ValueError, match="num_frequencies has to be greater or equal to 1"):
        module_init(num_frequencies=num_frequencies, min_frequency=0.1, max_frequency=10.0)


@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_min_frequency_incorrect(module_init: Callable) -> None:
    with pytest.raises(ValueError, match="min_frequency has to be greater than 0"):
        module_init(num_frequencies=2, min_frequency=0, max_frequency=1)


@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_max_frequency_incorrect(module_init: Callable) -> None:
    with pytest.raises(ValueError, match="max_frequency has to be greater than min_frequency"):
        module_init(num_frequencies=2, min_frequency=0.1, max_frequency=0.01)


@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_learnable(learnable: bool, module_init: Callable) -> None:
    module = module_init(
        num_frequencies=2, min_frequency=0.01, max_frequency=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


@patch(
    "karbonn.modules.scalar.cos_sin.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),  # noqa: ARG005
)
def test_cos_sin_scalar_encoder_create_rand_frequency() -> None:
    module = CosSinScalarEncoder.create_rand_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([0.2, 0.6, 1.0, 0.2, 0.6, 1.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


@patch(
    "karbonn.modules.scalar.cos_sin.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),  # noqa: ARG005
)
def test_cos_sin_scalar_encoder_create_rand_value_range() -> None:
    module = CosSinScalarEncoder.create_rand_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1
    )
    assert module.frequency.data.equal(torch.tensor([1.0, 3.0, 5.0, 1.0, 3.0, 5.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_linspace_frequency() -> None:
    module = CosSinScalarEncoder.create_linspace_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([0.2, 0.6, 1.0, 0.2, 0.6, 1.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_linspace_value_range() -> None:
    module = CosSinScalarEncoder.create_linspace_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1.0
    )
    assert module.frequency.data.equal(torch.tensor([1.0, 3.0, 5.0, 1.0, 3.0, 5.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_logspace_frequency() -> None:
    module = CosSinScalarEncoder.create_logspace_frequency(
        num_frequencies=3, min_frequency=0.01, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_logspace_value_range() -> None:
    module = CosSinScalarEncoder.create_logspace_value_range(
        num_frequencies=3, min_abs_value=0.01, max_abs_value=1.0
    )
    assert module.frequency.data.equal(torch.tensor([1.0, 10.0, 100.0, 1.0, 10.0, 100.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_forward_frequency_phase_shift() -> None:
    module = CosSinScalarEncoder(
        frequency=torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float),
        phase_shift=torch.zeros(6),
    )
    assert module(torch.tensor([[-1], [0], [1]], dtype=torch.float)).allclose(
        torch.tensor(
            [
                [
                    -0.8414709848078965,
                    -0.9092974268256817,
                    -0.1411200080598672,
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                ],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [
                    0.8414709848078965,
                    0.9092974268256817,
                    0.1411200080598672,
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                ],
            ],
            dtype=torch.float,
        ),
    )
