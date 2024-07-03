from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from coola.utils.tensor import get_available_devices

from karbonn.distributed import ddp

#################################
#     Tests for sync_reduce     #
#################################


@pytest.mark.parametrize("is_distributed", [True, False])
def test_sync_reduce_sum_number(is_distributed: bool) -> None:
    with patch("karbonn.distributed.ddp.is_distributed", lambda: is_distributed):
        assert ddp.sync_reduce(35, ddp.SUM) == 35


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("is_distributed", [True, False])
def test_sync_reduce_sum_tensor(device: str, is_distributed: bool) -> None:
    with patch("karbonn.distributed.ddp.is_distributed", lambda: is_distributed):
        var_reduced = ddp.sync_reduce(torch.ones(2, 3, device=device), ddp.SUM)
        assert var_reduced.equal(torch.ones(2, 3, device=device))


@patch("karbonn.distributed.ddp.get_world_size", lambda: 2)
@patch("karbonn.distributed.ddp.is_distributed", lambda: True)
def test_sync_reduce_avg_number_world_size_2_is_distributed() -> None:
    assert ddp.sync_reduce(8, ddp.AVG) == 4


@patch("karbonn.distributed.ddp.get_world_size", lambda: 2)
@patch("karbonn.distributed.ddp.is_distributed", lambda: False)
def test_sync_reduce_avg_number_world_size_2_is_not_distributed() -> None:
    assert ddp.sync_reduce(8, ddp.AVG) == 8


@pytest.mark.parametrize("device", get_available_devices())
@patch("karbonn.distributed.ddp.get_world_size", lambda: 2)
@patch("karbonn.distributed.ddp.is_distributed", lambda: True)
def test_sync_reduce_avg_tensor_world_size_2_is_distributed(device: str) -> None:
    var_reduced = ddp.sync_reduce(torch.ones(2, 3, device=device), ddp.AVG)
    assert var_reduced.equal(0.5 * torch.ones(2, 3, device=device))


@pytest.mark.parametrize("device", get_available_devices())
@patch("karbonn.distributed.ddp.get_world_size", lambda: 2)
@patch("karbonn.distributed.ddp.is_distributed", lambda: False)
def test_sync_reduce_avg_tensor_world_size_2_is_not_distributed(device: str) -> None:
    x = torch.ones(2, 3, device=device)
    x_reduced = ddp.sync_reduce(x, ddp.AVG)
    assert x_reduced.equal(x)  # no-op because not distributed
