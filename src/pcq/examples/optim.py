"""pcq.examples.optim — reference example optimizer atoms (factories).

Contract example (``adamw``) for the optim atom kind. NOT a production
optimizer catalog. New optimizers should be registered as project atoms via
``pcq.register_optim``.

``pcq.optim`` re-exports this function as a v2 compatibility facade.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW


def adamw(
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> AdamW:
    """AdamW factory."""
    return AdamW(params, lr=lr, weight_decay=weight_decay)


# Registry 자동 등록 (string-name lookup 용)
from pcq._registry import optims as _registry  # noqa: E402

# v1.9: optim 도 metadata-aware. factory 는 (params, **kwargs) signature 유지 —
# build("adamw", model.parameters(), lr=1e-3) 형태로 호출됨.
# 검증: AtomRef.params 에는 lr/weight_decay 만 있음 (model.parameters 는 fit 시점).
_registry.register(
    "adamw",
    factory=adamw,
    meta={
        "tasks": [],  # task-agnostic
        "params": {
            "lr": {"type": "float", "default": 1e-3, "min": 0.0},
            "weight_decay": {
                "type": "float",
                "default": 0.01,
                "min": 0.0,
            },
        },
        "smoke_safe": True,
        "description": (
            "AdamW optimizer factory. Requires model parameters at build time. "
            "[reference example — common baseline; register project atoms for "
            "Lion / Lookahead / etc.]"
        ),
    },
)
