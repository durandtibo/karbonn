from __future__ import annotations

import pytest
import torch
from coola.utils.tensor import get_available_devices
from torch import nn

from karbonn.utils.summary import (
    NO_PARAMETER,
    PARAMETER_NOT_INITIALIZED,
    ParameterSummary,
    get_parameter_summaries,
)

######################################
#     Tests for ParameterSummary     #
######################################


@pytest.mark.parametrize("device", get_available_devices())
def test_parameter_summary_from_parameter_device(device: str) -> None:
    device = torch.device(device)
    assert ParameterSummary.from_parameter(
        "weight", nn.Parameter(torch.ones(6, 4, device=device))
    ) == ParameterSummary(
        name="weight",
        mean=1.0,
        median=1.0,
        std=0.0,
        min=1.0,
        max=1.0,
        learnable=True,
        shape=(6, 4),
        device=device,
    )


def test_parameter_summary_from_parameter_not_learnable() -> None:
    assert ParameterSummary.from_parameter(
        "weight", nn.Parameter(torch.ones(6, 4), requires_grad=False)
    ) == ParameterSummary(
        name="weight",
        mean=1.0,
        median=1.0,
        std=0.0,
        min=1.0,
        max=1.0,
        learnable=False,
        shape=(6, 4),
        device=torch.device("cpu"),
    )


def test_parameter_summary_from_parameter_uninitialized() -> None:
    assert ParameterSummary.from_parameter(
        "weight", nn.UninitializedParameter()
    ) == ParameterSummary(
        name="weight",
        mean=PARAMETER_NOT_INITIALIZED,
        median=PARAMETER_NOT_INITIALIZED,
        std=PARAMETER_NOT_INITIALIZED,
        min=PARAMETER_NOT_INITIALIZED,
        max=PARAMETER_NOT_INITIALIZED,
        learnable=True,
        shape=PARAMETER_NOT_INITIALIZED,
        device=torch.device("cpu"),
    )


def test_parameter_summary_from_parameter_no_parameters() -> None:
    assert ParameterSummary.from_parameter(
        "weight", nn.Parameter(torch.ones(0, 4))
    ) == ParameterSummary(
        name="weight",
        mean=NO_PARAMETER,
        median=NO_PARAMETER,
        std=NO_PARAMETER,
        min=NO_PARAMETER,
        max=NO_PARAMETER,
        learnable=True,
        shape=(0, 4),
        device=torch.device("cpu"),
    )


#############################################
#     Tests for get_parameter_summaries     #
#############################################


def test_get_parameter_summaries_linear() -> None:
    linear = nn.Linear(4, 6)
    nn.init.ones_(linear.weight)
    nn.init.zeros_(linear.bias)
    assert get_parameter_summaries(linear) == [
        ParameterSummary(
            name="weight",
            mean=1.0,
            median=1.0,
            std=0.0,
            min=1.0,
            max=1.0,
            learnable=True,
            shape=(6, 4),
            device=torch.device("cpu"),
        ),
        ParameterSummary(
            name="bias",
            mean=0.0,
            median=0.0,
            std=0.0,
            min=0.0,
            max=0.0,
            learnable=True,
            shape=(6,),
            device=torch.device("cpu"),
        ),
    ]


def test_get_parameter_summaries_lazy_linear() -> None:
    assert get_parameter_summaries(nn.LazyLinear(6)) == [
        ParameterSummary(
            name="weight",
            mean=PARAMETER_NOT_INITIALIZED,
            median=PARAMETER_NOT_INITIALIZED,
            std=PARAMETER_NOT_INITIALIZED,
            min=PARAMETER_NOT_INITIALIZED,
            max=PARAMETER_NOT_INITIALIZED,
            learnable=True,
            shape=PARAMETER_NOT_INITIALIZED,
            device=torch.device("cpu"),
        ),
        ParameterSummary(
            name="bias",
            mean=PARAMETER_NOT_INITIALIZED,
            median=PARAMETER_NOT_INITIALIZED,
            std=PARAMETER_NOT_INITIALIZED,
            min=PARAMETER_NOT_INITIALIZED,
            max=PARAMETER_NOT_INITIALIZED,
            learnable=True,
            shape=PARAMETER_NOT_INITIALIZED,
            device=torch.device("cpu"),
        ),
    ]
