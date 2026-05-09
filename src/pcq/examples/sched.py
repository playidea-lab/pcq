"""pcq.examples.sched — reference example LR scheduler atoms.

Contract example (``cosine``) for the sched atom kind. NOT a production
scheduler catalog. New schedulers should be registered as project atoms via
``pcq.register_sched``.

``pcq.sched`` re-exports this function as a v2 compatibility facade.
"""
from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def cosine(optimizer: Optimizer, T_max: int, warmup: int = 0) -> LRScheduler:
    """Cosine annealing with optional linear warmup.

    Args:
        T_max: total steps (excluding warmup)
        warmup: linear warmup steps from start_factor=1e-3 to 1.0
    """
    if warmup <= 0:
        return CosineAnnealingLR(optimizer, T_max=T_max)
    # warmup 구간 → cosine 구간 순차 적용
    warmup_sched = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
    )
    cosine_sched = CosineAnnealingLR(optimizer, T_max=T_max)
    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )


# Registry 자동 등록 (string-name lookup 용)
from pcq._registry import scheds as _registry  # noqa: E402

# v1.9: sched 도 metadata-aware. factory signature: (optimizer, T_max, warmup=0).
# T_max 는 required — recipe 에 명시 필수.
_registry.register(
    "cosine",
    factory=cosine,
    meta={
        "tasks": [],
        "params": {
            "T_max": {
                "type": "int",
                "required": True,
                "min": 1,
                "description": (
                    "total cosine annealing steps (excluding warmup)"
                ),
            },
            "warmup": {
                "type": "int",
                "default": 0,
                "min": 0,
                "description": (
                    "linear warmup steps from start_factor=1e-3"
                ),
            },
        },
        "smoke_safe": True,
        "description": (
            "Cosine annealing with optional linear warmup. "
            "Requires optimizer at build. "
            "[reference example — register project atoms for OneCycle / Plateau "
            "/ custom schedules]"
        ),
    },
)
