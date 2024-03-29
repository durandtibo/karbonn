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
from karbonn.utils.size import BatchNormSizeFinder

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
        nn.BatchNorm1d: BatchNormSizeFinder(),
        nn.BatchNorm2d: BatchNormSizeFinder(),
        nn.BatchNorm3d: BatchNormSizeFinder(),
        nn.Bilinear: size_finders.BilinearSizeFinder(),
        nn.Conv1d: size_finders.ConvolutionSizeFinder(),
        nn.Conv2d: size_finders.ConvolutionSizeFinder(),
        nn.Conv3d: size_finders.ConvolutionSizeFinder(),
        nn.ConvTranspose1d: size_finders.ConvolutionSizeFinder(),
        nn.ConvTranspose2d: size_finders.ConvolutionSizeFinder(),
        nn.ConvTranspose3d: size_finders.ConvolutionSizeFinder(),
        nn.Embedding: size_finders.EmbeddingSizeFinder(),
        nn.EmbeddingBag: size_finders.EmbeddingSizeFinder(),
        nn.GRU: size_finders.RecurrentSizeFinder(),
        nn.LSTM: size_finders.RecurrentSizeFinder(),
        nn.Linear: size_finders.LinearSizeFinder(),
        nn.RNN: size_finders.RecurrentSizeFinder(),
        nn.Sequential: size_finders.SequentialSizeFinder(),
        nn.SyncBatchNorm: BatchNormSizeFinder(),
    }
