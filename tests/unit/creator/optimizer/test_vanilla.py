from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import SGD, Optimizer

from karbonn.creator.optimizer import OptimizerCreator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"

######################################
#     Tests for OptimizerCreator     #
######################################


def test_optimizer_creator_repr() -> None:
    assert repr(
        OptimizerCreator(optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001})
    ).startswith("OptimizerCreator")


def test_optimizer_creator_str() -> None:
    assert str(
        OptimizerCreator(optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001})
    ).startswith("OptimizerCreator")


@objectory_available
def test_optimizer_creator_create() -> None:
    linear = nn.Linear(in_features=4, out_features=6)
    assert isinstance(
        OptimizerCreator(optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.001}).create(linear),
        SGD,
    )


@pytest.mark.parametrize(
    ("config", "cls"),
    [
        ({OBJECT_TARGET: "torch.optim.Adadelta"}, torch.optim.Adadelta),
        ({OBJECT_TARGET: "torch.optim.Adafactor"}, torch.optim.Adafactor),
        ({OBJECT_TARGET: "torch.optim.Adagrad"}, torch.optim.Adagrad),
        ({OBJECT_TARGET: "torch.optim.Adam"}, torch.optim.Adam),
        ({OBJECT_TARGET: "torch.optim.AdamW"}, torch.optim.AdamW),
        ({OBJECT_TARGET: "torch.optim.SparseAdam"}, torch.optim.SparseAdam),
        ({OBJECT_TARGET: "torch.optim.Adamax"}, torch.optim.Adamax),
        ({OBJECT_TARGET: "torch.optim.ASGD"}, torch.optim.ASGD),
        ({OBJECT_TARGET: "torch.optim.LBFGS"}, torch.optim.LBFGS),
        ({OBJECT_TARGET: "torch.optim.NAdam"}, torch.optim.NAdam),
        ({OBJECT_TARGET: "torch.optim.RAdam"}, torch.optim.RAdam),
        ({OBJECT_TARGET: "torch.optim.RMSprop"}, torch.optim.RMSprop),
        ({OBJECT_TARGET: "torch.optim.Rprop"}, torch.optim.Rprop),
        ({OBJECT_TARGET: "torch.optim.SGD"}, torch.optim.SGD),
    ],
)
@objectory_available
def test_optimizer_creator_create_compatibility(config: dict, cls: type[Optimizer]) -> None:
    linear = nn.Linear(in_features=4, out_features=6)
    assert isinstance(
        OptimizerCreator(optimizer=config).create(linear),
        cls,
    )
