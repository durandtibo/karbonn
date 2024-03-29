from __future__ import annotations

__all__ = ["MODULES", "ModuleSizes", "MULTIHEAD_ATTENTION_MODULES", "IN3_OUT1_3D_MODULES"]

from typing import NamedTuple

from torch import nn


class ModuleSizes(NamedTuple):
    module: nn.Module
    in_features: list[int]
    out_features: list[int]


MULTIHEAD_ATTENTION_MODULES = [
    ModuleSizes(
        module=nn.MultiheadAttention(embed_dim=4, num_heads=1), in_features=[4], out_features=[4]
    ),
    ModuleSizes(
        module=nn.MultiheadAttention(embed_dim=8, num_heads=2), in_features=[8], out_features=[8]
    ),
]

IN3_OUT1_3D_MODULES = MULTIHEAD_ATTENTION_MODULES

MODULES = IN3_OUT1_3D_MODULES
