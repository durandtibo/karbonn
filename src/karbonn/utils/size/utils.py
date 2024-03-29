r"""Contain the default mappings between the module types and their size
finders."""

from __future__ import annotations

__all__ = [
    "get_karbonn_size_finders",
    "get_size_finders",
    "get_torch_size_finders",
]

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from karbonn.utils.size import BaseSizeFinder


def get_size_finders() -> dict[type[nn.Module], BaseSizeFinder]:
    r"""Return the default mappings between the module types and their
    size finders.

    Returns:
        The default mappings between the module types and their size
            finders.

    Example usage:

    ```pycon

    >>> from karbonn.utils.size import get_size_finders
    >>> get_size_finders()
    {<class 'torch.nn.modules.module.Module'>: UnknownSizeFinder(), ...}

    ```
    """
    return get_torch_size_finders() | get_karbonn_size_finders()


def get_karbonn_size_finders() -> dict[type[nn.Module], BaseSizeFinder]:
    r"""Return the default mappings between the module types and their
    size finders.

    Returns:
        The default mappings between the module types and their size
            finders.

    Example usage:

    ```pycon

    >>> from karbonn.utils.size import get_karbonn_size_finders
    >>> get_karbonn_size_finders()
    {...}

    ```
    """
    # Local import to avoid cyclic dependencies
    import karbonn
    from karbonn.utils import size as size_finders

    return {karbonn.ExU: size_finders.LinearSizeFinder()}


def get_torch_size_finders() -> dict[type[nn.Module], BaseSizeFinder]:
    r"""Return the default mappings between the module types and their
    size finders.

    Returns:
        The default mappings between the module types and their size
            finders.

    Example usage:

    ```pycon

    >>> from karbonn.utils.size import get_torch_size_finders
    >>> get_torch_size_finders()
    {<class 'torch.nn.modules.module.Module'>: UnknownSizeFinder(), ...}

    ```
    """
    # Local import to avoid cyclic dependencies
    from karbonn.utils import size as size_finders

    return {
        nn.Module: size_finders.UnknownSizeFinder(),
        nn.Sequential: size_finders.SequentialSizeFinder(),
        nn.Linear: size_finders.LinearSizeFinder(),
        nn.Bilinear: size_finders.BilinearSizeFinder(),
    }
