"""pcq.examples — reference example atoms.

Explicit namespace for contract examples. Model, dataset, optimizer, and
scheduler implementations live in ``pcq.examples.{models,datasets,optim,sched}``;
loss and metric categories currently mirror ``pcq.{loss, metric}`` for
compatibility.

The atoms exposed here exist for:

  1. **Contract verification** — registry / Trainer / Experiment / CQ contract
     keeps working when new atoms are registered with the same surface.
  2. **Onboarding** — concrete reference for users writing project-local atoms
     (`pcq.register_*`, `pcq atoms scaffold`).
  3. **Smoke tests** — CI baseline that catches framework breakage.

They are **NOT** a production model catalog. Real research atoms belong in
project-local `atoms/` via `pcq.register_*`.

```python
import pcq.examples as examples

# Contract examples — not production catalog entries.
examples.models.mlp(in_dim=784, hidden=[128], out_dim=10)
examples.datasets.fake(num_samples=8)
examples.optim.adamw(model.parameters())
examples.loss.cross_entropy(ignore_index=-1)
```

`pcq.models.mlp(...)`, `pcq.datasets.fake(...)`, `pcq.optim.adamw(...)`, and
`pcq.sched.cosine(...)` remain as v2 compatibility facades for their
`pcq.examples.*` counterparts. Prefer project-local atoms for real work.
"""
from __future__ import annotations

from importlib import import_module


def __getattr__(name: str):
    if name in {"datasets", "models", "optim", "sched"}:
        return import_module(f"pcq.examples.{name}")
    if name in {"loss", "metric"}:
        return import_module(f"pcq.{name}")
    raise AttributeError(f"module 'pcq.examples' has no attribute {name!r}")


__all__ = [
    "datasets",
    "loss",
    "metric",
    "models",
    "optim",
    "sched",
]
