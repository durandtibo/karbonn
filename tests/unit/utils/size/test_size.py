from __future__ import annotations

import pytest
import torch

from tests.unit.utils.size.utils import IN3_OUT1_3D_MODULES, ModuleSizes


@pytest.mark.parametrize("module", IN3_OUT1_3D_MODULES)
def test_in3_out1_3d_sizes(module: ModuleSizes) -> None:
    x1 = torch.rand(2, 3, module.in_features[0])
    x2 = torch.rand(2, 3, module.in_features[1])
    x3 = torch.rand(2, 3, module.in_features[2])
    output = module.module(x1, x2, x3)
    assert output.shape == (2, 3, module.out_features[0])
