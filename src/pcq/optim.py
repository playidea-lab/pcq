"""pcq.optim — compatibility facade for reference example optimizer atoms.

The reference implementations live in :mod:`pcq.examples.optim`.

This module remains for v2.x compatibility with older examples and tests that
import ``pcq.optim.adamw``. New docs should prefer ``pcq.examples.optim`` for
contract examples and project-local atoms for production optimizers.
"""
from __future__ import annotations

from pcq.examples.optim import adamw

__all__ = ["adamw"]
