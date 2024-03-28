r"""Define the base class to find the input and output feature sizes."""

from __future__ import annotations

__all__ = ["BaseSizeFinder", "SizeFinderConfig"]

from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

from torch import nn

T = TypeVar("T", bound=nn.Module)


class BaseSizeFinder(ABC, Generic[T]):
    r"""Define the base class to find the input or output feature size
    of a module."""

    @abstractmethod
    def find_in_features(self, module: T, config: SizeFinderConfig) -> list[int]:
        r"""Find the input feature sizes of a given module.

        Args:
            module: The module.
            config: The configuration to find the feature sizes.

        Returns:
            The input feature sizes.

        Raises:
            SizeNotFound: if the input feature size could not be
                found.
        """

    @abstractmethod
    def find_out_features(self, module: T, config: SizeFinderConfig) -> list[int]:
        r"""Find the output feature sizes of a given module.

        Args:
            module: The module.
            config: The configuration to find the feature sizes.

        Returns:
            The output feature sizes.

        Raises:
            SizeNotFoundError: if the output feature size could not be
                found.
        """


class SizeNotFoundError(Exception):
    r"""Raised if the size could not be found,."""


class SizeFinderConfig(NamedTuple):
    r"""Define the config to control the size finders.

    Args:
        size_finder: The size finder.
    """

    size_finder: BaseSizeFinder
