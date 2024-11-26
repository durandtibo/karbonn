from __future__ import annotations

import torch.nn
from torch import nn

from karbonn.creator import Creator, DictCreator
from karbonn.testing import objectory_available
from karbonn.utils.imports import is_objectory_available

if is_objectory_available():
    from objectory import OBJECT_TARGET
else:  # pragma: no cover
    OBJECT_TARGET = "_target_"


#################################
#     Tests for DictCreator     #
#################################


@objectory_available
def test_dict_creator_repr() -> None:
    assert repr(
        DictCreator(
            creators={
                "key1": {
                    OBJECT_TARGET: "karbonn.creator.Creator",
                    "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
                },
                "key2": Creator(torch.nn.Identity()),
            },
        )
    ).startswith("DictCreator")


@objectory_available
def test_dict_creator_str() -> None:
    assert str(
        DictCreator(
            creators={
                "key1": {
                    OBJECT_TARGET: "karbonn.creator.Creator",
                    "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
                },
                "key2": Creator(torch.nn.Identity()),
            },
        )
    ).startswith("DictCreator")


@objectory_available
def test_dict_creator_create_one_item() -> None:
    obj = DictCreator(
        creators={
            "key1": {
                OBJECT_TARGET: "karbonn.creator.Creator",
                "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
            },
        },
    ).create()
    assert isinstance(obj, dict)
    assert len(obj) == 1
    assert isinstance(obj["key1"], nn.Linear)


@objectory_available
def test_dict_creator_create_two_creators() -> None:
    obj = DictCreator(
        creators={
            "key1": {
                OBJECT_TARGET: "karbonn.creator.Creator",
                "obj_or_config": torch.nn.Linear(in_features=4, out_features=6),
            },
            "key2": Creator(torch.nn.Identity()),
        },
    ).create()
    assert isinstance(obj, dict)
    assert len(obj) == 2
    assert isinstance(obj["key1"], nn.Linear)
    assert isinstance(obj["key2"], nn.Identity)
