"""Recipe: fake seg smoke — UNet on random masks. CI 용 (외부 dep 0).

v1.8: RecipeSpec metadata-first 형태. 기존 dict factory 호환 유지.
"""
from __future__ import annotations

import pcq
from pcq.agent.schema import RecipeSpec


SPEC = RecipeSpec(
    name="vision/seg/fake_seg_smoke",
    task="segmentation",
    description=(
        "UNet on synthetic segmentation masks — fastest CI smoke for seg."
    ),
    metrics=["epoch", "train_loss", "train_iou", "eval_loss", "eval_iou"],
    monitor_candidates=[
        {"name": "eval_iou", "mode": "max"},
        {"name": "eval_loss", "mode": "min"},
    ],
    requires_extras=[],
    smoke_safe=True,
    atoms={
        "model": pcq.model_ref(
            "unet", {"in_channels": 3, "num_classes": 10, "base_ch": 16}
        ),
        # split 별 별도 dataset 인스턴스 (build_ref 가 즉시 build).
        "dataset_train": pcq.dataset_ref(
            "fake_seg",
            {"num_samples": 32, "num_classes": 10, "image_size": 32},
        ),
        "dataset_eval": pcq.dataset_ref(
            "fake_seg",
            {"num_samples": 16, "num_classes": 10, "image_size": 32},
        ),
        "loss": pcq.loss_ref("cross_entropy"),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3}),
    },
    defaults={"epochs": 2, "batch_size": 8},
)


def fake_seg_smoke() -> dict:
    """Fake seg + tiny UNet — torchvision 없이 segmentation 경로 검증."""
    return SPEC.build()
