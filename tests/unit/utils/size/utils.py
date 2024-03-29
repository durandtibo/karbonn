from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch import nn


class ModuleSizes(NamedTuple):
    module: nn.Module
    in_features: list[int]
    out_features: list[int]
