r"""Contain functions to instantiate a ``torch.nn.Module`` object from
its configuration."""

from __future__ import annotations

__all__ = ["create_sequential", "is_module_config", "setup_module"]

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

from torch import nn

from karbonn.utils.imports import check_objectory, is_objectory_available

if TYPE_CHECKING:
    from collections.abc import Sequence

if is_objectory_available():
    import objectory
else:  # pragma: no cover
    objectory = Mock()


logger = logging.getLogger(__name__)


def is_module_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``torch.nn.Module``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``torch.nn.Module`` object, otherwise ``False``..

    Example usage:

    ```pycon

    >>> from karbonn.utils import is_module_config
    >>> is_module_config({"_target_": "torch.nn.Identity"})
    True

    ```
    """
    check_objectory()
    return objectory.utils.is_object_config(config, nn.Module)


def setup_module(module: nn.Module | dict) -> nn.Module:
    r"""Set up a ``torch.nn.Module`` object.

    Args:
        module: The module or its configuration.

    Returns:
        The instantiated ``torch.nn.Module`` object.

    Example usage:

    ```pycon

    >>> from karbonn.utils import setup_module
    >>> linear = setup_module(
    ...     {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6}
    ... )
    >>> linear
    Linear(in_features=4, out_features=6, bias=True)

    ```
    """
    if isinstance(module, dict):
        logger.info("Initializing a 'torch.nn.Module' from its configuration... ")
        check_objectory()
        module = objectory.factory(**module)
    if not isinstance(module, nn.Module):
        logger.warning(f"module is not a 'torch.nn.Module' (received: {type(module)})")
    return module


def create_sequential(modules: Sequence[nn.Module | dict]) -> nn.Sequential:
    r"""Create a ``torch.nn.Sequential`` from a sequence of modules.

    Args:
        modules: The sequence of modules or their configuration.

    Returns:
        The instantiated ``torch.nn.Sequential`` object.

    Example usage:

    ```pycon

    >>> from karbonn.utils.factory import create_sequential
    >>> seq = create_sequential(
    ...     [
    ...         {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 6},
    ...         {"_target_": "torch.nn.ReLU"},
    ...         {"_target_": "torch.nn.Linear", "in_features": 6, "out_features": 6},
    ...     ]
    ... )
    >>> seq
    Sequential(
      (0): Linear(in_features=4, out_features=6, bias=True)
      (1): ReLU()
      (2): Linear(in_features=6, out_features=6, bias=True)
    )

    ```
    """
    return nn.Sequential(*[setup_module(module) for module in modules])
