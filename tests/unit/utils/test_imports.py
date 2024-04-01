from __future__ import annotations

from unittest.mock import patch

import pytest

from karbonn.utils.imports import (
    check_objectory,
    check_tabulate,
    is_objectory_available,
    is_tabulate_available,
    objectory_available,
    tabulate_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


#####################
#     objectory     #
#####################


def test_check_objectory_with_package() -> None:
    with patch("karbonn.utils.imports.is_objectory_available", lambda: True):
        check_objectory()


def test_check_objectory_without_package() -> None:
    with (
        patch("karbonn.utils.imports.is_objectory_available", lambda: False),
        pytest.raises(RuntimeError, match="`objectory` package is required but not installed."),
    ):
        check_objectory()


def test_is_objectory_available() -> None:
    assert isinstance(is_objectory_available(), bool)


def test_objectory_available_with_package() -> None:
    with patch("karbonn.utils.imports.is_objectory_available", lambda: True):
        fn = objectory_available(my_function)
        assert fn(2) == 44


def test_objectory_available_without_package() -> None:
    with patch("karbonn.utils.imports.is_objectory_available", lambda: False):
        fn = objectory_available(my_function)
        assert fn(2) is None


def test_objectory_available_decorator_with_package() -> None:
    with patch("karbonn.utils.imports.is_objectory_available", lambda: True):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_objectory_available_decorator_without_package() -> None:
    with patch("karbonn.utils.imports.is_objectory_available", lambda: False):

        @objectory_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


####################
#     tabulate     #
####################


def test_check_tabulate_with_package() -> None:
    with patch("karbonn.utils.imports.is_tabulate_available", lambda: True):
        check_tabulate()


def test_check_tabulate_without_package() -> None:
    with (
        patch("karbonn.utils.imports.is_tabulate_available", lambda: False),
        pytest.raises(RuntimeError, match="`tabulate` package is required but not installed."),
    ):
        check_tabulate()


def test_is_tabulate_available() -> None:
    assert isinstance(is_tabulate_available(), bool)


def test_tabulate_available_with_package() -> None:
    with patch("karbonn.utils.imports.is_tabulate_available", lambda: True):
        fn = tabulate_available(my_function)
        assert fn(2) == 44


def test_tabulate_available_without_package() -> None:
    with patch("karbonn.utils.imports.is_tabulate_available", lambda: False):
        fn = tabulate_available(my_function)
        assert fn(2) is None


def test_tabulate_available_decorator_with_package() -> None:
    with patch("karbonn.utils.imports.is_tabulate_available", lambda: True):

        @tabulate_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_tabulate_available_decorator_without_package() -> None:
    with patch("karbonn.utils.imports.is_tabulate_available", lambda: False):

        @tabulate_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
