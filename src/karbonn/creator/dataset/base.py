r"""Contain the base class to implement a dataset creator."""

from __future__ import annotations

__all__ = ["BaseDatasetCreator"]

import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import AbstractFactory
else:  # pragma: no cover
    AbstractFactory = ABCMeta

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDatasetCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataset creator.

    Example usage:

    ```pycon

    >>> from karbonn.creator.dataset import DatasetCreator
    >>> creator = DatasetCreator(
    ...     {
    ...         "_target_": "karbonn.testing.dummy.DummyDataset",
    ...         "num_examples": 10,
    ...         "feature_size": 4,
    ...     }
    ... )
    >>> creator
    DatasetCreator(
      (dataset): {'_target_': 'karbonn.testing.dummy.DummyDataset', 'num_examples': 10, 'feature_size': 4}
    )
    >>> creator.create()
    DummyDataset(num_examples=10, feature_size=4, rng_seed=14700295087918620795)

    ```
    """

    @abstractmethod
    def create(self) -> Dataset[T]:
        r"""Create a dataset.

        Returns:
            The created dataset.

        Example usage:

        ```pycon

        >>> from karbonn.creator.dataset import DatasetCreator
        >>> creator = DatasetCreator(
        ...     {
        ...         "_target_": "karbonn.testing.dummy.DummyDataset",
        ...         "num_examples": 10,
        ...         "feature_size": 4,
        ...     }
        ... )
        >>> creator.create()
        DummyDataset(num_examples=10, feature_size=4, rng_seed=14700295087918620795)

        ```
        """
