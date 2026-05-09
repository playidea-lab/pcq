"""pcq.sched — compatibility facade for reference example scheduler atoms.

The reference implementations live in :mod:`pcq.examples.sched`.

This module remains for v2.x compatibility with older examples and tests that
import ``pcq.sched.cosine``. New docs should prefer ``pcq.examples.sched`` for
contract examples and project-local atoms for production schedulers.
"""
from __future__ import annotations

from pcq.examples.sched import cosine

__all__ = ["cosine"]
