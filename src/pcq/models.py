"""pcq.models — compatibility facade for reference example model atoms.

The reference implementations live in :mod:`pcq.examples.models`.

This module remains for v2.x compatibility with older examples and tests that
import ``pcq.models.mlp`` or pass ``Trainer(model="small_cnn")``. New docs should
prefer ``pcq.examples.models`` for contract examples and project-local atoms for
production code.
"""
from __future__ import annotations

from pcq.examples.models import (
    deeplab_v3,
    mlp,
    resnet18,
    small_cnn,
    text_classifier,
    unet,
)

__all__ = [
    "deeplab_v3",
    "mlp",
    "resnet18",
    "small_cnn",
    "text_classifier",
    "unet",
]
