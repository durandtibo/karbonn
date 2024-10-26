from __future__ import annotations

__all__ = ["PiecewiseLinearNumericalEncoder"]

import math
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from karbonn.modules.numerical.sine import prepare_tensor_param

if TYPE_CHECKING:
    from torch import Tensor


class PiecewiseLinearNumericalEncoder(Module):

    def __init__(self, bins: Tensor) -> None:
        super().__init__()
        bins = prepare_tensor_param(bins, name="bins")
        n_bins = bins.shape[1]
        self.register_buffer("boundary", bins[:, :-1] if n_bins > 1 else bins)
        self.register_buffer("width", bins.diff() if n_bins > 1 else torch.ones_like(bins))

    @property
    def input_size(self) -> int:
        r"""Return the input feature size i.e. the number of scalar
        values."""
        return self.boundary.shape[0]

    @property
    def output_size(self) -> int:
        r"""Return the output feature size i.e. the number of bins minus
        one."""
        return self.boundary.shape[1]

    def extra_repr(self) -> str:
        return f"n_features={self.input_size}, n_bins={self.output_size}"

    def forward(self, scalar: Tensor) -> Tensor:
        x = (scalar[..., None] - self.boundary) / self.width
        n_bins = x.shape[-1]
        if n_bins == 1:
            return x
        return torch.cat(
            [
                x[..., :1].clamp_max(1.0),
                *([] if n_bins == 2 else [x[..., 1:-1].clamp(0.0, 1.0)]),
                x[..., -1:].clamp_min(0.0),
            ],
            dim=-1,
        )


class PiecewiseLinearEncodingImpl(Module):
    # NOTE
    # 1. DO NOT USE THIS CLASS DIRECTLY (ITS OUTPUT CONTAINS INFINITE VALUES).
    # 2. This implementation is not memory efficient for cases when there are many
    #    features with low number of bins and only few features
    #    with high number of bins. If this becomes a problem,
    #    just split features into groups and encode the groups separately.

    # The output of this module has the shape (*batch_dims, n_features, max_n_bins),
    # where max_n_bins = max(map(len, bins)) - 1.
    # If the i-th feature has the number of bins less than max_n_bins,
    # then its piecewise-linear representation is padded with inf as follows:
    # [x_1, x_2, ..., x_k, inf, ..., inf]
    # where:
    #            x_1 <= 1.0
    #     0.0 <= x_i <= 1.0 (for i in range(2, k))
    #     0.0 <= x_k
    #     k == len(bins[i]) - 1  (the number of bins for the i-th feature)

    # If all features have the same number of bins, then there are no infinite values.

    edges: Tensor
    width: Tensor
    mask: Tensor

    def __init__(self, bins: list[Tensor]) -> None:
        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer("edges", edges[:, :-1])
        self.register_buffer("width", edges.diff())
        # mask is false for the padding values.
        self.register_buffer(
            "mask",
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(max_n_edges - 1, dtype=torch.bool, device=x.device),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: Tensor) -> Tensor:
        # See Equation 1 in the paper.
        x = (x[..., None] - self.edges) / self.width

        # If the number of bins is greater than 1, then, the following rules must
        # be applied to a piecewise-linear encoding of a single feature:
        # - the leftmost value can be negative, but not greater than 1.0.
        # - the rightmost value can be greater than 1.0, but not negative.
        # - the intermediate values must stay within [0.0, 1.0].
        n_bins = x.shape[-1]
        if n_bins > 1:
            if self._same_bin_count:
                x = torch.cat(
                    [
                        x[..., :1].clamp_max(1.0),
                        *([] if n_bins == 2 else [x[..., 1:-1].clamp(0.0, 1.0)]),
                        x[..., -1:].clamp_min(0.0),
                    ],
                    dim=-1,
                )
            else:
                # In this case, the rightmost values for all features are located
                # in different columns.
                x = torch.stack(
                    [
                        (
                            x[..., i, :]
                            if count == 1
                            else torch.cat(
                                [
                                    x[..., i, :1].clamp_max(1.0),
                                    *(
                                        []
                                        if n_bins == 2
                                        else [x[..., i, 1 : count - 1].clamp(0.0, 1.0)]
                                    ),
                                    x[..., i, count - 1 : count].clamp_min(0.0),
                                    x[..., i, count:],
                                ],
                                dim=-1,
                            )
                        )
                        for i, count in enumerate(self._bin_counts)
                    ],
                    dim=-2,
                )
        return x


if __name__ == "__main__":
    x = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    m1 = PiecewiseLinearEncodingImpl(
        [torch.tensor([0.0, 1.0, 2.0, 4.0]), torch.tensor([2.0, 4.0, 6.0, 8.0])]
    )

    o1 = m1(x)

    m2 = PiecewiseLinearNumericalEncoder(
        bins=torch.tensor([[0.0, 1.0, 2.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    )

    o2 = m2(x)
