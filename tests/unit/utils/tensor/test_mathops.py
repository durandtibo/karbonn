from __future__ import annotations

import pytest
import torch

from karbonn.utils.tensor import quantile

DTYPES = (torch.float, torch.long)


##############################
#     Tests for quantile     #
##############################


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantile_dtype(dtype: torch.dtype) -> None:
    assert quantile(torch.arange(11).to(dtype=dtype), q=torch.tensor([0.1])).equal(
        torch.tensor([1], dtype=torch.float),
    )


def test_quantile_q_multiple() -> None:
    assert quantile(torch.arange(11), q=torch.tensor([0.1, 0.5, 0.9])).equal(
        torch.tensor([1, 5, 9], dtype=torch.float),
    )
