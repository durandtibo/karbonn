from __future__ import annotations

from unittest.mock import patch

import pytest
from torch.distributed import Backend

from karbonn.distributed import (
    UnknownBackendError,
    distributed_context,
    is_distributed,
    is_main_process,
)
from karbonn.utils.imports import ignite_available, is_ignite_available

if is_ignite_available():  # pragma: no cover
    from ignite import distributed as idist

####################################
#     Tests for is_distributed     #
####################################


def test_is_distributed_false_not_available() -> None:
    with patch("karbonn.distributed.utils.is_available", lambda: False):
        assert not is_distributed()


def test_is_distributed_false_not_initialized() -> None:
    with (
        patch("karbonn.distributed.utils.is_available", lambda: True),
        patch("karbonn.distributed.utils.is_initialized", lambda: False),
    ):
        assert not is_distributed()


def test_is_distributed_true() -> None:
    with (
        patch("karbonn.distributed.utils.is_available", lambda: True),
        patch("karbonn.distributed.utils.is_initialized", lambda: True),
    ):
        assert is_distributed()


#####################################
#     Tests for is_main_process     #
#####################################


def test_is_main_process_false_distributed() -> None:
    with (
        patch("karbonn.distributed.utils.is_distributed", lambda: True),
        patch("karbonn.distributed.utils.get_rank", lambda: 1),
    ):
        assert not is_main_process()


def test_is_main_process_true_distributed() -> None:
    with (
        patch("karbonn.distributed.utils.is_distributed", lambda: True),
        patch("karbonn.distributed.utils.get_rank", lambda: 0),
    ):
        assert is_main_process()


def test_is_main_process_true_not_distributed() -> None:
    with patch("karbonn.distributed.utils.is_distributed", lambda: False):
        assert is_main_process()


#########################################
#     Tests for distributed_context     #
#########################################


@ignite_available
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO,))
def test_distributed_context_backend() -> None:
    with (
        patch("karbonn.distributed.utils.idist.initialize") as initialize_mock,
        patch("karbonn.distributed.utils.idist.finalize") as finalize_mock,
        distributed_context(Backend.GLOO),
    ):
        pass
    initialize_mock.assert_called_once_with(Backend.GLOO, init_method="env://")
    finalize_mock.assert_called_once_with()


@ignite_available
def test_distributed_context_backend_incorrect() -> None:
    with pytest.raises(UnknownBackendError), distributed_context(backend="incorrect backend"):
        pass


@ignite_available
def test_distributed_context_backend_raise_error() -> None:
    # Test if the `finalize` function is called to release the resources.
    with pytest.raises(RuntimeError), distributed_context(backend=Backend.GLOO):  # noqa: PT012
        msg = "Fake error"
        raise RuntimeError(msg)
    assert idist.backend() is None
