from __future__ import annotations

import pytest
import torch
from coola import objects_are_allclose, objects_are_equal
from coola.testing import numpy_available, torch_numpy_available
from coola.utils.tensor import get_available_devices

from karbonn.utils.tensor import quantile, quantile_numpy

DTYPES = (torch.float, torch.double, torch.long)
QUANTILE_INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")


##############################
#     Tests for quantile     #
##############################


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_quantile_dtype(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        quantile(torch.arange(11, dtype=dtype), q=torch.tensor([0.1], dtype=dtype)),
        torch.tensor([1.0], dtype=dtype),
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_device(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        quantile(
            torch.arange(11, dtype=torch.float, device=device), q=torch.tensor([0.1], device=device)
        ),
        torch.tensor([1.0], device=device),
    )


def test_quantile_q_multiple() -> None:
    assert objects_are_equal(
        quantile(torch.arange(11, dtype=torch.float), q=torch.tensor([0.1, 0.5, 0.9])),
        torch.tensor([1.0, 5.0, 9.0], dtype=torch.float),
    )


@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATIONS)
def test_quantile_interpolation(interpolation: str) -> None:
    assert objects_are_equal(
        quantile(
            torch.arange(11, dtype=torch.float),
            q=torch.tensor([0.1, 0.5, 0.9]),
            interpolation=interpolation,
        ),
        torch.tensor([1.0, 5.0, 9.0], dtype=torch.float),
    )


@torch_numpy_available
def test_quantile_large() -> None:
    assert objects_are_allclose(
        quantile(torch.arange(20000000, dtype=torch.float), q=torch.tensor([0.1, 0.5, 0.9])),
        torch.tensor([2000000.0, 10000000.0, 18000000.0], dtype=torch.float),
        rtol=1e-5,
    )


@numpy_available
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATIONS)
def test_quantile_compatibility(device: str, interpolation: str) -> None:
    device = torch.device(device)
    values = torch.arange(11, dtype=torch.float, device=device)
    q = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float, device=device)
    assert objects_are_equal(
        quantile(values, q=q, interpolation=interpolation),
        quantile_numpy(values, q=q, interpolation=interpolation),
    )


####################################
#     Tests for quantile_numpy     #
####################################


@torch_numpy_available
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantile_numpy_dtype(dtype: torch.dtype) -> None:
    assert objects_are_equal(
        quantile_numpy(torch.arange(11, dtype=dtype), q=torch.tensor([0.1])),
        torch.tensor([1], dtype=dtype),
    )


@torch_numpy_available
@pytest.mark.parametrize("device", get_available_devices())
def test_quantile_numpy_device(device: str) -> None:
    device = torch.device(device)
    assert objects_are_equal(
        quantile_numpy(torch.arange(11, device=device), q=torch.tensor([0.1])),
        torch.tensor([1], device=device),
    )


@torch_numpy_available
def test_quantile_numpy_q_multiple() -> None:
    assert objects_are_equal(
        quantile_numpy(torch.arange(11), q=torch.tensor([0.1, 0.5, 0.9])),
        torch.tensor([1, 5, 9]),
    )


@torch_numpy_available
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATIONS)
def test_quantile_numpy_interpolation(interpolation: str) -> None:
    assert objects_are_equal(
        quantile_numpy(
            torch.arange(11), q=torch.tensor([0.1, 0.5, 0.9]), interpolation=interpolation
        ),
        torch.tensor([1, 5, 9]),
    )
