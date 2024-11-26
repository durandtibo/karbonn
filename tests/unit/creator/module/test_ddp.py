from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from torch import distributed as dist
from torch import nn

from karbonn.creator.module.ddp import to_ddp
from karbonn.testing import ignite_available

if TYPE_CHECKING:
    import pytest

############################
#     Tests for to_ddp     #
############################


@ignite_available
@patch("karbonn.creator.module.ddp.isinstance", lambda *args: True)  # noqa: ARG005
def test_to_ddp_already_ddp(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        module = nn.Linear(4, 5)
        assert to_ddp(module) is module
        assert len(caplog.messages) >= 1


@ignite_available
@patch("karbonn.creator.module.ddp.isinstance", lambda *args: False)  # noqa: ARG005
@patch("karbonn.creator.module.ddp.idist.backend", lambda: dist.Backend.GLOO)
def test_to_ddp_gloo() -> None:
    ddp_mock = Mock()
    with patch("karbonn.creator.module.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module)
        ddp_mock.assert_called_once_with(module)


@ignite_available
@patch("karbonn.creator.module.ddp.isinstance", lambda *args: False)  # noqa: ARG005
@patch("karbonn.creator.module.ddp.idist.backend", lambda: dist.Backend.GLOO)
def test_to_ddp_gloo_ddp_kwargs() -> None:
    ddp_mock = Mock()
    with patch("karbonn.creator.module.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module, ddp_kwargs={"find_unused_parameters": True})
        ddp_mock.assert_called_once_with(module, find_unused_parameters=True)


@ignite_available
@patch("karbonn.creator.module.ddp.isinstance", lambda *args: False)  # noqa: ARG005
@patch("karbonn.creator.module.ddp.idist.backend", lambda: dist.Backend.NCCL)
@patch("karbonn.creator.module.ddp.idist.get_local_rank", lambda: 1)
def test_to_ddp_nccl() -> None:
    ddp_mock = Mock()
    with patch("karbonn.creator.module.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module)
        ddp_mock.assert_called_once_with(module, device_ids=[1])


@ignite_available
@patch("karbonn.creator.module.ddp.isinstance", lambda *args: False)  # noqa: ARG005
@patch("karbonn.creator.module.ddp.idist.backend", lambda: dist.Backend.NCCL)
@patch("karbonn.creator.module.ddp.idist.get_local_rank", lambda: 1)
def test_to_ddp_nccl_ddp_kwargs() -> None:
    ddp_mock = Mock()
    with patch("karbonn.creator.module.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module, ddp_kwargs={"find_unused_parameters": True})
        ddp_mock.assert_called_once_with(module, device_ids=[1], find_unused_parameters=True)


@ignite_available
@patch("karbonn.creator.module.ddp.idist.backend", lambda: "UNKNOWN_BACKEND")
def test_to_ddp_unknown_backend() -> None:
    assert isinstance(to_ddp(nn.Linear(4, 6)), nn.Linear)
