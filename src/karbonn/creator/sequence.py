r"""Contain object creator implementations."""

from __future__ import annotations

__all__ = ["ListCreator"]

from typing import TYPE_CHECKING, TypeVar

from coola.utils import repr_indent, repr_sequence, str_indent, str_sequence

from karbonn.creator.base import BaseCreator
from karbonn.utils.factory import setup_object

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


class ListCreator(BaseCreator[T]):
    r"""Implement a list object creator.

    Args:
        items: The sequence of objects or their configurations.

    Example usage:

    ```pycon

    >>> from karbonn.creator import ListCreator
    >>> creator = ListCreator(
    ...     [
    ...         {
    ...             "_target_": "torch.nn.Linear",
    ...             "in_features": 4,
    ...             "out_features": 6,
    ...         },
    ...         {"_target_": "torch.nn.Identity"},
    ...     ]
    ... )
    >>> creator
    ListCreator(
      (0): {'_target_': 'torch.nn.Linear', 'in_features': 4, 'out_features': 6}
      (1): {'_target_': 'torch.nn.Identity'}
    )
    >>> creator.create()
    [Linear(in_features=4, out_features=6, bias=True), Identity()]

    ```
    """

    def __init__(self, items: Sequence[T | dict]) -> None:
        self._items = items

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(repr_sequence(self._items))}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._items))}\n)"

    def create(self) -> list[T]:
        return [setup_object(item) for item in self._items]
