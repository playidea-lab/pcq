"""Recipe: CIFAR-10 + SmallCNN baseline.

v1.9: RecipeSpec metadata-first 형태. smoke_overrides 는 fake 32x32x3 fallback.
"""
from __future__ import annotations

import pcq
from pcq import datasets
from pcq.agent.schema import RecipeSpec


SPEC = RecipeSpec(
    name="vision/cifar10_smallcnn_baseline",
    task="classification",
    description="SmallCNN baseline on CIFAR-10 — tutorial recipe.",
    metrics=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"],
    monitor_candidates=[
        {"name": "eval_acc", "mode": "max"},
        {"name": "eval_loss", "mode": "min"},
    ],
    requires_extras=["vision"],
    smoke_safe=False,
    smoke_overrides={
        # CIFAR-10 다운로드 우회 — 32x32x3 fake
        "dataset_train": lambda _split: datasets.fake(
            num_samples=32, num_classes=10, image_size=32, channels=3,
        ),
        "dataset_eval": lambda _split: datasets.fake(
            num_samples=16, num_classes=10, image_size=32, channels=3,
        ),
    },
    atoms={
        "model": pcq.model_ref(
            "small_cnn", {"in_channels": 3, "num_classes": 10},
        ),
        "dataset_train": pcq.dataset_ref(
            "cifar10", {"root": "data", "train": True, "download": True},
        ),
        "dataset_eval": pcq.dataset_ref(
            "cifar10", {"root": "data", "train": False, "download": True},
        ),
        "loss": pcq.loss_ref("cross_entropy"),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3}),
        "sched": pcq.sched_ref("cosine", {"T_max": 10, "warmup": 500}),
    },
    defaults={"epochs": 10, "batch_size": 64},
)


def cifar10_smallcnn_baseline() -> dict:
    """CIFAR-10 + SmallCNN baseline recipe."""
    return SPEC.build()
