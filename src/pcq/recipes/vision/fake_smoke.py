"""Recipe: fake dataset + MLP — fastest CI smoke.

v1.9: RecipeSpec metadata-first 형태. 외부 dep 0, smoke_safe=True.
"""
from __future__ import annotations

import pcq
from pcq.agent.schema import RecipeSpec


SPEC = RecipeSpec(
    name="vision/fake_smoke",
    task="classification",
    description=(
        "MLP on synthetic 32x32 RGB images — torch-only smoke."
    ),
    metrics=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"],
    monitor_candidates=[
        {"name": "eval_loss", "mode": "min"},
        {"name": "eval_acc", "mode": "max"},
    ],
    requires_extras=[],
    smoke_safe=True,
    atoms={
        "model": pcq.model_ref(
            "mlp",
            {"in_dim": 3 * 32 * 32, "hidden": [64], "out_dim": 10},
        ),
        "dataset_train": pcq.dataset_ref(
            "fake",
            {"num_samples": 128, "num_classes": 10, "image_size": 32},
        ),
        "dataset_eval": pcq.dataset_ref(
            "fake",
            {"num_samples": 64, "num_classes": 10, "image_size": 32},
        ),
        "loss": pcq.loss_ref("cross_entropy"),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3}),
    },
    defaults={"epochs": 2, "batch_size": 32},
)


def fake_smoke() -> dict:
    """Fake + MLP smoke recipe — 외부 dep 0, CI 최단."""
    return SPEC.build()
