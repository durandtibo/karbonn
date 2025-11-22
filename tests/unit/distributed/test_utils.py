from __future__ import annotations

from unittest.mock import patch

import pytest
from torch.distributed import Backend

from karbonn.distributed import (
    UnknownBackendError,
    auto_backend,
    distributed_context,
    gloocontext,
    is_distributed,
    is_main_process,
    ncclcontext,
    resolve_backend,
)
from karbonn.utils.imports import ignite_available, is_ignite_available

if is_ignite_available():  # pragma: no cover
    pass

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
    with (
        pytest.raises(UnknownBackendError, match=r"Unknown backend"),
        distributed_context(backend="incorrect backend"),
    ):
        pass


@ignite_available
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO,))
def test_distributed_context_backend_raise_error() -> None:
    # Test if the `finalize` function is called to release the resources.
    with (  # noqa: PT012
        pytest.raises(RuntimeError, match=r"Fake error"),
        patch("karbonn.distributed.utils.idist.initialize") as initialize_mock,
        patch("karbonn.distributed.utils.idist.barrier") as barrier_mock,
        patch("karbonn.distributed.utils.idist.finalize") as finalize_mock,
        distributed_context(Backend.GLOO),
    ):
        msg = "Fake error"
        raise RuntimeError(msg)
    initialize_mock.assert_called_once_with(Backend.GLOO, init_method="env://")
    barrier_mock.assert_called_once_with()
    finalize_mock.assert_called_once_with()


##################################
#     Tests for auto_backend     #
##################################


@ignite_available
@pytest.mark.parametrize("cuda_is_available", [True, False])
@patch("karbonn.distributed.utils.idist.available_backends", lambda: ())
def test_auto_backend_no_backend(cuda_is_available: bool) -> None:
    with patch("torch.cuda.is_available", lambda: cuda_is_available):
        assert auto_backend() is None


@ignite_available
@patch("torch.cuda.is_available", lambda: False)
def test_auto_backend_no_gpu() -> None:
    assert auto_backend() == Backend.GLOO


@ignite_available
@patch("torch.cuda.is_available", lambda: False)
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO, Backend.NCCL))
def test_auto_backend_no_gpu_and_nccl() -> None:
    assert auto_backend() == Backend.GLOO


@ignite_available
@patch("torch.cuda.is_available", lambda: True)
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO,))
def test_auto_backend_gpu_and_no_nccl() -> None:
    assert auto_backend() == Backend.GLOO


@ignite_available
@patch("torch.cuda.is_available", lambda: True)
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO, Backend.NCCL))
def test_auto_backend_gpu_and_nccl() -> None:
    assert auto_backend() == Backend.NCCL


#####################################
#     Tests for resolve_backend     #
#####################################


@ignite_available
@pytest.mark.parametrize("backend", [Backend.GLOO, Backend.NCCL, None])
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO, Backend.NCCL))
def test_resolve_backend(backend: str | None) -> None:
    assert resolve_backend(backend) == backend


@ignite_available
@pytest.mark.parametrize("backend", [Backend.GLOO, Backend.NCCL])
def test_resolve_backend_auto(backend: str) -> None:
    with patch("karbonn.distributed.utils.auto_backend", lambda: backend):
        assert resolve_backend("auto") == backend


@ignite_available
def test_resolve_backend_incorrect_backend() -> None:
    with pytest.raises(UnknownBackendError, match=r"Unknown distributed backend 'incorrect'"):
        resolve_backend("incorrect")


#################################
#     Tests for gloocontext     #
#################################


@ignite_available
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.GLOO,))
def test_gloocontext() -> None:
    with patch("karbonn.distributed.utils.distributed_context") as mock, gloocontext():
        mock.assert_called_once_with(Backend.GLOO)


@ignite_available
@patch("karbonn.distributed.utils.idist.available_backends", lambda: ())
def test_gloocontext_no_gloo_backend() -> None:
    with pytest.raises(RuntimeError), gloocontext():
        pass


#################################
#     Tests for ncclcontext     #
#################################


@ignite_available
@patch("torch.cuda.is_available", lambda: True)
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.NCCL,))
@patch("karbonn.distributed.utils.idist.get_local_rank", lambda: 1)
def test_ncclcontext() -> None:
    with (
        patch("karbonn.distributed.utils.distributed_context") as mock,
        patch("karbonn.distributed.utils.torch.cuda.device") as device,
        ncclcontext(),
    ):
        mock.assert_called_once_with(Backend.NCCL)
        device.assert_called_once_with(1)


@ignite_available
@patch("torch.cuda.is_available", lambda: True)
@patch("karbonn.distributed.utils.idist.available_backends", lambda: ())
def test_ncclcontext_no_nccl_backend() -> None:
    with pytest.raises(RuntimeError), ncclcontext():
        pass


@ignite_available
@patch("torch.cuda.is_available", lambda: False)
@patch("karbonn.distributed.utils.idist.available_backends", lambda: (Backend.NCCL,))
def test_ncclcontext_cuda_is_not_available() -> None:
    with pytest.raises(RuntimeError), ncclcontext():
        pass
