r"""Contain some math utility functions."""

from __future__ import annotations

__all__ = ["quantile"]

from unittest.mock import Mock

import torch
from coola.utils import check_numpy, is_numpy_available
from torch import Tensor

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()


def quantile(
    tensor: Tensor, q: Tensor, method: str = "linear", dtype: torch.dtype = torch.float
) -> Tensor:
    r"""Return the ``q``-th quantiles.

    This function uses numpy to compute the ``q``-th quantiles
    because PyTorch has a limit to 16M items.
    https://github.com/pytorch/pytorch/issues/64947

    Args:
        tensor: The tensor of values.
        q: The ``q``-values in the range ``[0, 1]``. This input is a
            ``torch.Tensor`` of type float and shape
            ``(num_q_values,)``.
        method: The interpolation method to use when the desired
            quantile lies between two data points. Can be
            ``'linear'``, ``'lower'``, ``'higher'``, ``'midpoint'``
            and ``'nearest'``.
        dtype: The tensor output data type.

    Returns:
        The ``q``-th quantiles as a ``torch.Tensor`` of shape
            ``(num_q_values,)``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.tensor import quantile
    >>> quantile(torch.arange(1001), q=torch.tensor([0.1, 0.9]))
    tensor([100., 900.])

    ```
    """
    check_numpy()
    return torch.from_numpy(
        np.quantile(tensor.detach().cpu().numpy(), q=q.numpy(), method=method)
    ).to(dtype=dtype, device=tensor.device)
