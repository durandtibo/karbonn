r"""Contain the base class to implement a metric."""

from __future__ import annotations

__all__ = ["BaseMetric", "EmptyMetricError", "setup_metric"]

import logging
from abc import abstractmethod

from torch.nn import Module

from karbonn.utils.imports import check_objectory, is_objectory_available

if is_objectory_available():
    from objectory import AbstractFactory
else:  # pragma: no cover

    class AbstractFactory: ...


logger = logging.getLogger(__name__)


class BaseMetric(Module, metaclass=AbstractFactory):
    r"""Define the base class to implement a metric.

    This class is used to register the metric using the metaclass
    factory. Child classes must implement the following methods:

        - ``forward``
        - ``reset``
        - ``value``
    """

    # @abstractmethod
    # def forward(self, *args: Any, **kwargs: Any) -> dict | None:
    #     r"""Update the metric given a mini-batch of examples.
    #
    #     Args:
    #         *args: Variable length argument list.
    #         **kwargs: Arbitrary keyword arguments.
    #
    #     Returns:
    #     """

    @abstractmethod
    def reset(self) -> None:
        r"""Reset the metric."""

    @abstractmethod
    def value(self) -> dict:
        r"""Evaluate the metric and return the results given all the
        examples previously seen.

        Returns:
             The results of the metric.
        """


class EmptyMetricError(Exception):
    r"""Raised when you try to evaluate an empty metric."""


def setup_metric(metric: BaseMetric | dict) -> BaseMetric:
    r"""Set up the metric.

    Args:
        metric: The metric or its configuration.

    Returns:
        The instantiated metric.
    """
    if isinstance(metric, dict):
        logger.info("Initializing a metric from its configuration...")
        check_objectory()
        metric = BaseMetric.factory(**metric)
    if not isinstance(metric, Module):
        logger.warning(f"metric is not a 'torch.nn.Module' (received: {type(metric)})")
    return metric
