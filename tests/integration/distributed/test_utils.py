from __future__ import annotations

from karbonn.distributed import is_distributed, is_main_process

####################################
#     Tests for is_distributed     #
####################################


def test_is_distributed() -> None:
    assert not is_distributed()


#####################################
#     Tests for is_main_process     #
#####################################


def test_is_main_process() -> None:
    assert is_main_process()
