from __future__ import annotations

import importlib
import logging

import torch
from torch import nn

from karbonn import ResidualBlock
from karbonn.utils import setup_module
from karbonn.utils.imports import is_objectory_available, objectory_available

if is_objectory_available():
    import objectory

logger = logging.getLogger(__name__)


def check_imports() -> None:
    logger.info("Checking imports...")
    objects_to_import = [
        "karbonn.ResidualBlock",
        "karbonn.utils.setup_module",
    ]
    for a in objects_to_import:
        module_path, name = a.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        obj = getattr(module, name)
        assert obj is not None


def check_modules() -> None:
    logger.info("Checking modules...")
    module = ResidualBlock(residual=nn.Linear(4, 4))
    out = module(torch.rand(6, 4))
    assert out.shape == (6, 4)
    assert out.dtype == torch.float


@objectory_available
def check_utils() -> None:
    logger.info("Checking utils...")
    assert isinstance(setup_module({objectory.OBJECT_TARGET: "torch.nn.ReLU"}), nn.ReLU)


def main() -> None:
    check_imports()
    check_modules()
    check_utils()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
