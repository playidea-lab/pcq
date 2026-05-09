"""pcq.datasets — compatibility facade for reference example dataset atoms.

The reference implementations live in :mod:`pcq.examples.datasets`.

This module remains for v2.x compatibility with older examples and tests that
import ``pcq.datasets.fake`` or pass ``Trainer(dataset="fake")``. New docs should
prefer ``pcq.examples.datasets`` for contract examples and project-local atoms
for production data.
"""
from __future__ import annotations

from pcq.examples.datasets import (
    cifar10,
    fake,
    fake_seg,
    fake_text,
    mnist,
    voc_seg,
)

__all__ = [
    "cifar10",
    "fake",
    "fake_seg",
    "fake_text",
    "mnist",
    "voc_seg",
]
