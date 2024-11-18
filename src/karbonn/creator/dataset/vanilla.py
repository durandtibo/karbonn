r"""Contain a simple dataset creator implementation."""

from __future__ import annotations

__all__ = ["DatasetCreator"]

from typing import TYPE_CHECKING, TypeVar

from coola.utils import str_indent, str_mapping

from karbonn.creator.dataset.base import BaseDatasetCreator
from karbonn.utils.factory import setup_object

if TYPE_CHECKING:
    from torch.utils.data import Dataset

T = TypeVar("T")


class DatasetCreator(BaseDatasetCreator[T]):
    r"""Implement a simple dataset creator.

    Args:
        dataset: The dataset or its configuration.

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

    def __init__(self, dataset: Dataset[T] | dict) -> None:
        self._dataset = dataset

    def __repr__(self) -> str:
        config = {"dataset": self._dataset}
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(config))}\n)"

    def create(self) -> Dataset[T]:
        return setup_object(self._dataset)
