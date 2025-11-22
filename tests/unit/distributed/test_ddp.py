from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola import objects_are_equal
from coola.utils.tensor import get_available_devices

from karbonn.distributed import ddp
from karbonn.testing import ignite_available

#################################
#     Tests for sync_reduce     #
#################################


def test_sync_reduce_non_distributed() -> None:
    assert ddp.sync_reduce(35, ddp.SUM) == 35


@ignite_available
@pytest.mark.parametrize("is_distributed", [True, False])
def test_sync_reduce_sum_number(is_distributed: bool) -> None:
    with patch("karbonn.distributed.ddp.is_distributed", lambda: is_distributed):
        assert ddp.sync_reduce(35, ddp.SUM) == 35


@ignite_available
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("is_distributed", [True, False])
def test_sync_reduce_sum_tensor(device: str, is_distributed: bool) -> None:
    with patch("karbonn.distributed.ddp.is_distributed", lambda: is_distributed):
        var_reduced = ddp.sync_reduce(torch.ones(2, 3, device=device), ddp.SUM)
        assert var_reduced.equal(torch.ones(2, 3, device=device))


@ignite_available
@pytest.mark.parametrize("device", get_available_devices())
@patch("karbonn.distributed.ddp.is_distributed", lambda: False)
def test_sync_reduce_sum_tensor_is_not_distributed(device: str) -> None:
    x = torch.ones(2, 3, device=device)
    x_reduced = ddp.sync_reduce(x, ddp.SUM)
    assert x_reduced.equal(x)  # no-op because not distributed


##################################
#     Tests for sync_reduce_     #
##################################


@ignite_available
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("is_distributed", [True, False])
def test_sync_reduce__sum(device: str, is_distributed: bool) -> None:
    with patch("karbonn.distributed.ddp.is_distributed", lambda: is_distributed):
        variable = torch.ones(2, 3, device=device)
        ddp.sync_reduce_(variable, ddp.SUM)
        assert variable.equal(torch.ones(2, 3, device=device))


@ignite_available
@pytest.mark.parametrize("device", get_available_devices())
@patch("karbonn.distributed.ddp.is_distributed", lambda: False)
def test_sync_reduce__sum_is_not_distributed(device: str) -> None:
    variable = torch.ones(2, 3, device=device)
    ddp.sync_reduce_(variable, ddp.SUM)
    assert variable.equal(torch.ones(2, 3, device=device))  # no-op because not distributed


def test_sync_reduce__incorrect_input() -> None:
    with pytest.raises(TypeError, match=r"sync_reduce_ only supports Tensor"):
        ddp.sync_reduce_(1, ddp.SUM)


################################################
#     Tests for all_gather_tensor_varshape     #
################################################


@patch("karbonn.distributed.ddp.is_distributed", lambda: False)
def test_all_gather_tensor_varshape_not_distributed() -> None:
    assert objects_are_equal(ddp.all_gather_tensor_varshape(torch.arange(6)), [torch.arange(6)])


@ignite_available
@patch("karbonn.distributed.ddp.is_distributed", lambda: True)
@patch("karbonn.distributed.ddp.idist.all_gather", lambda tensor: tensor)
def test_all_gather_tensor_varshape_distributed() -> None:
    assert objects_are_equal(ddp.all_gather_tensor_varshape(torch.arange(6)), [torch.arange(6)])
