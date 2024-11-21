r"""Contain a simple optimizer creator implementation."""

from __future__ import annotations

__all__ = ["OptimizerCreator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from karbonn.creator.optimizer.base import BaseOptimizerCreator
from karbonn.utils.factory import setup_optimizer

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer


class OptimizerCreator(BaseOptimizerCreator):
    r"""Implement a simple optimizer creator.

    Args:
        optimizer: The optimizer configuration.

    Example usage:

    ```pycon

    >>> from torch import nn
    >>> from karbonn.creator.optimizer import OptimizerCreator
    >>> linear = nn.Linear(4, 6)
    >>> creator = OptimizerCreator({"_target_": "torch.optim.SGD", "lr": 0.001})
    >>> creator
    OptimizerCreator(
      (optimizer): {'_target_': 'torch.optim.SGD', 'lr': 0.001}
    )
    >>> creator.create(linear)
    SGD (
    Parameter Group 0
        dampening: 0
        differentiable: False
        foreach: None
        fused: None
        lr: 0.001
        maximize: False
        momentum: 0
        nesterov: False
        weight_decay: 0
    )

    ```
    """

    def __init__(self, optimizer: dict) -> None:
        self._optimizer = optimizer

    def __repr__(self) -> str:
        config = {"optimizer": self._optimizer}
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(config))}\n)"

    def create(self, module: Module) -> Optimizer:
        return setup_optimizer(self._optimizer | {"params": module.parameters()})
