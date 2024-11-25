r"""Contain a simple object creator implementation."""

from __future__ import annotations

__all__ = ["Creator"]

from typing import TypeVar

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from karbonn.creator.base import BaseCreator
from karbonn.utils.factory import setup_object

T = TypeVar("T")


class Creator(BaseCreator[T]):
    r"""Implement a simple object creator.

    Args:
        obj_or_config: The object or its configuration.

    Example usage:

    ```pycon

    >>> from karbonn.creator import Creator
    >>> creator = Creator(
    ...     {
    ...         "_target_": "torch.nn.Linear",
    ...         "in_features": 4,
    ...         "out_features": 6,
    ...     }
    ... )
    >>> creator
    Creator(
      (_target_): torch.nn.Linear
      (in_features): 4
      (out_features): 6
    )
    >>> creator.create()
    Linear(in_features=4, out_features=6, bias=True)

    ```
    """

    def __init__(self, obj_or_config: T | dict) -> None:
        self._obj_or_config = obj_or_config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n  "
            f"{repr_indent(repr_mapping(self._obj_or_config))}\n)"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n  "
            f"{str_indent(str_mapping(self._obj_or_config))}\n)"
        )

    def create(self) -> T:
        return setup_object(self._obj_or_config)
