r"""Define the base class to find the input and output feature sizes."""

from __future__ import annotations

__all__ = ["BaseSizeFinder"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


class BaseSizeFinder(ABC):
    r"""Define the base class to find the input or output feature size
    of a module."""

    @abstractmethod
    def find_in_features(self, module: nn.Module) -> list[int]:
        r"""Find the input feature size of a given module.

        Args:
            module: The module.

        Returns:
            The input feature size.

        Raises:
            SizeNotFound: if the input feature size could not be
                found.
        """

    @abstractmethod
    def find_out_features(self, module: nn.Module) -> list[int]:
        r"""Find the output feature size of a given module.

        Args:
            module: The module.

        Returns:
            The output feature size.

        Raises:
            SizeNotFoundError: if the output feature size could not be
                found.
        """


class SizeNotFoundError(Exception):
    r"""Raised if the size could not be found,."""
