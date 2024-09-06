from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from karbonn.testing import tabulate_available
from karbonn.utils.format import str_table


@pytest.fixture
def tabular_data() -> list:
    return [["col1", "col2", "col3"], [10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]]


###############################
#     Tests for str_table     #
###############################


@tabulate_available
def test_str_table_tabulate(tabular_data: Any) -> None:
    assert str_table(tabular_data, tablefmt="heavy_grid") == (
        "┏━━━━━━┳━━━━━━┳━━━━━━┓\n"
        "┃ col1 ┃ col2 ┃ col3 ┃\n"
        "┣━━━━━━╋━━━━━━╋━━━━━━┫\n"
        "┃ 10   ┃ 20   ┃ 30   ┃\n"
        "┣━━━━━━╋━━━━━━╋━━━━━━┫\n"
        "┃ 11   ┃ 21   ┃ 31   ┃\n"
        "┣━━━━━━╋━━━━━━╋━━━━━━┫\n"
        "┃ 12   ┃ 22   ┃ 32   ┃\n"
        "┣━━━━━━╋━━━━━━╋━━━━━━┫\n"
        "┃ 13   ┃ 23   ┃ 33   ┃\n"
        "┗━━━━━━┻━━━━━━┻━━━━━━┛"
    )


def test_str_table_no_tabulate(tabular_data: Any) -> None:
    with patch("karbonn.utils.format.is_tabulate_available", lambda: False):
        assert (
            str_table(tabular_data)
            == "[['col1', 'col2', 'col3'], [10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]]"
        )
