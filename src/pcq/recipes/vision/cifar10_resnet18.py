"""Recipe: CIFAR-10 + ResNet-18 baseline (from scratch).

v1.9: RecipeSpec metadata-first 형태. smoke_overrides 는 fake 32x32x3.
"""
from __future__ import annotations

import pcq
from pcq import datasets
from pcq.agent.schema import RecipeSpec


SPEC = RecipeSpec(
    name="vision/cifar10_resnet18",
    task="classification",
    description="ResNet-18 from-scratch baseline on CIFAR-10.",
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
            num_samples=16, num_classes=10, image_size=32, channels=3,
        ),
        "dataset_eval": lambda _split: datasets.fake(
            num_samples=8, num_classes=10, image_size=32, channels=3,
        ),
    },
    atoms={
        "model": pcq.model_ref(
            "resnet18", {"num_classes": 10, "pretrained": False},
        ),
        "dataset_train": pcq.dataset_ref(
            "cifar10", {"root": "data", "train": True, "download": True},
        ),
        "dataset_eval": pcq.dataset_ref(
            "cifar10", {"root": "data", "train": False, "download": True},
        ),
        "loss": pcq.loss_ref("cross_entropy"),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3, "weight_decay": 5e-4}),
        "sched": pcq.sched_ref("cosine", {"T_max": 30, "warmup": 500}),
    },
    defaults={"epochs": 30, "batch_size": 128},
)


def cifar10_resnet18() -> dict:
    """CIFAR-10 + ResNet-18 baseline recipe (from scratch)."""
    return SPEC.build()
