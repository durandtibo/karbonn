r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "check_objectory",
    "check_tabulate",
    "is_objectory_available",
    "is_tabulate_available",
    "objectory_available",
    "tabulate_available",
]

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from coola.utils.imports import decorator_package_available

if TYPE_CHECKING:
    from collections.abc import Callable


#####################
#     objectory     #
#####################


def is_objectory_available() -> bool:
    r"""Indicate if the ``objectory`` package is installed or not.

    Returns:
        ``True`` if ``objectory`` is available otherwise ``False``.

    Example usage:

    ```pycon
    >>> from karbonn.utils.imports import is_objectory_available
    >>> is_objectory_available()

    ```
    """
    return find_spec("objectory") is not None


def check_objectory() -> None:
    r"""Check if the ``objectory`` package is installed.

    Raises:
        RuntimeError: if the ``objectory`` package is not installed.

    Example usage:

    ```pycon
    >>> from karbonn.utils.imports import check_objectory
    >>> check_objectory()

    ```
    """
    if not is_objectory_available():
        msg = (
            "`objectory` package is required but not installed. "
            "You can install `objectory` package with the command:\n\n"
            "pip install objectory\n"
        )
        raise RuntimeError(msg)


def objectory_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``objectory``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``objectory`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon
    >>> from karbonn.utils.imports import objectory_available
    >>> @objectory_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_objectory_available)


####################
#     tabulate     #
####################


def is_tabulate_available() -> bool:
    r"""Indicate if the ``tabulate`` package is installed or not.

    Returns:
        ``True`` if ``tabulate`` is available otherwise ``False``.

    Example usage:

    ```pycon
    >>> from karbonn.utils.imports import is_tabulate_available
    >>> is_tabulate_available()

    ```
    """
    return find_spec("tabulate") is not None


def check_tabulate() -> None:
    r"""Check if the ``tabulate`` package is installed.

    Raises:
        RuntimeError: if the ``tabulate`` package is not installed.

    Example usage:

    ```pycon
    >>> from karbonn.utils.imports import check_tabulate
    >>> check_tabulate()

    ```
    """
    if not is_tabulate_available():
        msg = (
            "`tabulate` package is required but not installed. "
            "You can install `tabulate` package with the command:\n\n"
            "pip install tabulate\n"
        )
        raise RuntimeError(msg)


def tabulate_available(fn: Callable[..., Any]) -> Callable[..., Any]:
    r"""Implement a decorator to execute a function only if ``tabulate``
    package is installed.

    Args:
        fn: Specifies the function to execute.

    Returns:
        A wrapper around ``fn`` if ``tabulate`` package is installed,
            otherwise ``None``.

    Example usage:

    ```pycon
    >>> from karbonn.utils.imports import tabulate_available
    >>> @tabulate_available
    ... def my_function(n: int = 0) -> int:
    ...     return 42 + n
    ...
    >>> my_function()

    ```
    """
    return decorator_package_available(fn, is_tabulate_available)
