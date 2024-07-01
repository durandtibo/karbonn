from __future__ import annotations

from unittest.mock import patch

from karbonn.distributed import is_distributed, is_main_process

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
