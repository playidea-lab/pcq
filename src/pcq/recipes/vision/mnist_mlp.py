"""Recipe: MNIST + MLP baseline.

v1.9: RecipeSpec metadata-first 형태. smoke_overrides 는 callable 로 유지
(테스트 호환 — recipe_smoke 가 callable 인자 사용).
"""
from __future__ import annotations

import pcq
from pcq import datasets
from pcq.agent.schema import RecipeSpec
from pcq.examples import models as example_models


SPEC = RecipeSpec(
    name="vision/mnist_mlp",
    task="classification",
    description="MLP on MNIST — common simple baseline.",
    metrics=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"],
    monitor_candidates=[
        {"name": "eval_acc", "mode": "max"},
        {"name": "eval_loss", "mode": "min"},
    ],
    requires_extras=["vision"],
    smoke_safe=False,
    smoke_overrides={
        # MNIST 다운로드 우회 — 28x28 grayscale fake
        "dataset_train": lambda _split: datasets.fake(
            num_samples=32, num_classes=10, image_size=28, channels=1,
        ),
        "dataset_eval": lambda _split: datasets.fake(
            num_samples=16, num_classes=10, image_size=28, channels=1,
        ),
        # MLP input dim = 28*28*1 = 784 (channels 1 일치)
        "model": example_models.mlp(28 * 28, [16], 10),
    },
    atoms={
        "model": pcq.model_ref(
            "mlp", {"in_dim": 28 * 28, "hidden": [128, 64], "out_dim": 10},
        ),
        "dataset_train": pcq.dataset_ref(
            "mnist", {"root": "data", "train": True, "download": True},
        ),
        "dataset_eval": pcq.dataset_ref(
            "mnist", {"root": "data", "train": False, "download": True},
        ),
        "loss": pcq.loss_ref("cross_entropy"),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3}),
        "sched": pcq.sched_ref("cosine", {"T_max": 10}),
    },
    defaults={"epochs": 10, "batch_size": 128},
)


def mnist_mlp() -> dict:
    """MNIST + MLP baseline recipe."""
    return SPEC.build()
