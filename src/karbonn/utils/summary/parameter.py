r"""Contains some functionalities to analyze the parameters of a
``torch.nn.Module``."""

from __future__ import annotations

__all__ = [
    "NO_PARAMETER",
    "PARAMETER_NOT_INITIALIZED",
    "ParameterSummary",
    "get_parameter_summaries",
    "tabulate_parameter_summary",
]

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from coola.nested import convert_to_dict_of_lists
from torch.nn import Module, Parameter, UninitializedParameter

from karbonn.utils.imports import check_tabulate, is_tabulate_available

if TYPE_CHECKING:
    import torch

if is_tabulate_available():
    from tabulate import tabulate

logger = logging.getLogger(__name__)

PARAMETER_NOT_INITIALIZED = "NI"
NO_PARAMETER = "NP"


@dataclass
class ParameterSummary:
    r"""Implement a class to easily manage a parameter summary.

    NI: Not Initialized
    NP: No Parameter
    """

    name: str
    mean: float | str
    median: float | str
    std: float | str
    min: float | str
    max: float | str
    shape: tuple[int, ...] | str
    learnable: bool | str
    device: torch.device | str

    @classmethod
    def from_parameter(
        cls, name: str, parameter: Parameter | UninitializedParameter
    ) -> ParameterSummary:
        r"""Create the parameter summary from the parameter object.

        Args:
            name: The name of the parameter.
            parameter: The parameter object.

        Returns:
            The parameter summary.

        Example usage:

        ```pycon

        >>> import torch
        >>> from karbonn.utils.summary import ParameterSummary
        >>> ParameterSummary.from_parameter("weight", torch.nn.Parameter(torch.ones(6, 4)))
        ParameterSummary(name='weight', mean=1.0, median=1.0, std=0.0, min=1.0, max=1.0, shape=(6, 4), learnable=True, device=device(type='cpu'))

        ```
        """
        if isinstance(parameter, UninitializedParameter):
            return cls(
                name=name,
                mean=PARAMETER_NOT_INITIALIZED,
                median=PARAMETER_NOT_INITIALIZED,
                std=PARAMETER_NOT_INITIALIZED,
                min=PARAMETER_NOT_INITIALIZED,
                max=PARAMETER_NOT_INITIALIZED,
                shape=PARAMETER_NOT_INITIALIZED,
                learnable=parameter.requires_grad,
                device=parameter.device,
            )
        if parameter.numel() == 0:
            return cls(
                name=name,
                mean=NO_PARAMETER,
                median=NO_PARAMETER,
                std=NO_PARAMETER,
                min=NO_PARAMETER,
                max=NO_PARAMETER,
                shape=tuple(parameter.shape),
                learnable=parameter.requires_grad,
                device=parameter.device,
            )
        return cls(
            name=name,
            mean=parameter.mean().item(),
            median=parameter.median().item(),
            std=parameter.std().item(),
            min=parameter.min().item(),
            max=parameter.max().item(),
            shape=tuple(parameter.shape),
            learnable=parameter.requires_grad,
            device=parameter.device,
        )


def get_parameter_summaries(module: Module) -> list[ParameterSummary]:
    r"""Return the parameter summaries of a module.

    Args:
        module: The module with the parameters to summarize.

    Returns:
        The list of parameter summaries.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.summary import get_parameter_summaries
    >>> get_parameter_summaries(torch.nn.Linear(4, 6))
    [ParameterSummary(name='weight', mean=..., median=..., std=..., min=..., max=..., shape=(6, 4), learnable=True, device=device(type='cpu')),
     ParameterSummary(name='bias', mean=..., median=..., std=..., min=..., max=..., shape=(6,), learnable=True, device=device(type='cpu'))]

    ```
    """
    return [
        ParameterSummary.from_parameter(name, parameter)
        for name, parameter in module.named_parameters()
    ]


def tabulate_parameter_summary(
    module: Module, tablefmt: str = "fancy_outline", floatfmt: str = ".6f", **kwargs: Any
) -> str:
    r"""Return a string summary of the model parameters.

    Args:
        module: The module to analyze.
        tablefmt: The table format.
        floatfmt: The float format.
        kwargs: Variable keyword arguments to pass to ``tabulate``.

    Raises:
        RuntimeError: if ``tabulate`` package is not available.

    Example usage:

    ```pycon

    >>> import torch
    >>> from karbonn.utils.summary import tabulate_parameter_summary
    >>> tabulate_parameter_summary(torch.nn.Linear(4, 6))

    ```
    """
    check_tabulate()
    summaries = convert_to_dict_of_lists(
        [asdict(summary) for summary in get_parameter_summaries(module)]
    )
    return tabulate(summaries, headers="keys", tablefmt=tablefmt, floatfmt=floatfmt, **kwargs)
