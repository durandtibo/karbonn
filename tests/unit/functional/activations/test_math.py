from __future__ import annotations

import torch

from karbonn.functional import safe_exp, safe_log

##############################
#     Tests for safe_exp     #
##############################


def test_safe_exp_max_value_default() -> None:
    assert safe_exp(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float)).allclose(
        torch.tensor([0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 485165184.0]),
    )


def test_safe_exp_max_value_1() -> None:
    assert safe_exp(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float), max=1.0).equal(
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 2.7182817459106445, 2.7182817459106445]
        ),
    )


##############################
#     Tests for safe_log     #
##############################


def test_safe_log_min_value_default() -> None:
    assert safe_log(torch.tensor([-1, 0, 1, 2], dtype=torch.float)).allclose(
        torch.tensor([-18.420680743952367, -18.420680743952367, 0.0, 0.6931471805599453]),
    )


def test_safe_log_min_value_1() -> None:
    assert safe_log(torch.tensor([-1, 0, 1, 2], dtype=torch.float), min=1.0).equal(
        torch.tensor([0.0, 0.0, 0.0, 0.6931471805599453]),
    )
