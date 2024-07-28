r"""Contain functionalities for the data distributed parallel
setting."""

from __future__ import annotations

__all__ = ["sync_reduce", "sync_reduce_", "BAND", "BOR", "MAX", "MIN", "PRODUCT", "SUM"]

from typing import overload

import torch
from torch import Tensor

from karbonn.distributed.utils import is_distributed
from karbonn.utils.imports import check_ignite, is_ignite_available

if is_ignite_available():  # pragma: no cover
    from ignite import distributed as idist

# The supported reduction operators
BAND = "AND"  # Bitwise AND (only for integer/long)
BOR = "OR"  # Bitwise OR (only for integer/long)
MAX = "MAX"
MIN = "MIN"
PRODUCT = "PRODUCT"
SUM = "SUM"


@overload
def sync_reduce(variable: Tensor, op: str) -> Tensor: ...  # pragma: no cover


@overload
def sync_reduce(variable: float, op: str) -> float: ...  # pragma: no cover


def sync_reduce(variable: Tensor | float, op: str) -> Tensor | float:
    r"""Synchronize all the processes and then reduce the variable.

    This function is a no-operation function if the distributed mode
    is not activated. It returns the input. If the distributed mode
    is activated, this function does not change the input variable.
    If the input is a tensor, this function will create a copy of the
    tensor before to reduce it. After this function is executed,
    the input variable will contain the value before reduction.
    If you want to do an in-place operation, you can use
    ``sync_reduce_``.

    Args:
        variable: The variable to reduce.
        op: The reduction operation. The available operations are:
            ``AND``, ``OR``, ``MAX``, ``MIN``, ``PRODUCT``,
            and ``SUM``.

    Returns:
        The reduced variable.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.distributed import ddp
    >>> x = torch.ones(2, 3)
    >>> x_reduced = ddp.sync_reduce(x, op=ddp.SUM)
    >>> # for two processes
    >>> x_reduced  # doctest: +SKIP
    tensor([[2., 2., 2.],
            [2., 2., 2.]])

    ```
    """
    if is_distributed():
        check_ignite()
        if torch.is_tensor(variable):
            # Create a copy to not change the values of the input tensor.
            variable = variable.clone()
        variable = idist.all_reduce(variable, op=op)
    return variable


def sync_reduce_(tensor: Tensor, op: str) -> Tensor:
    r"""In-place version of ``sync_reduce`` but it works only for a
    tensor.

    Args:
        tensor: The tensor to reduce in-place.
        op: The reduction operation. The available operations are:
            ``AND``, ``OR``, ``MAX``, ``MIN``, ``PRODUCT``,
            and ``SUM``.

    Returns:
        The reduced tensor which is also the input tensor.

    Raises:
        TypeError: if the input is not a tensor.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn import distributed as dist
    >>> from karbonn.distributed import ddp
    >>> x = torch.ones(2, 3)
    >>> ddp.sync_reduce_(x, op=ddp.SUM)
    >>> # for two processes
    >>> x  # doctest: +SKIP
    tensor([[2., 2., 2.],
            [2., 2., 2.]])

    ```
    """
    if not torch.is_tensor(tensor):
        msg = f"The function `sync_reduce_` only supports Tensor but received {type(tensor)}"
        raise TypeError(msg)

    if is_distributed():
        check_ignite()
        idist.all_reduce(tensor, op=op)
    return tensor
