"""Recipe: Pascal VOC 2012 + UNet. pcq[vision] extras (torchvision) 필요.

v1.8: RecipeSpec metadata-first 형태. smoke_overrides 는 callable 직접 (테스트 호환).
"""
from __future__ import annotations

import pcq
from pcq import datasets
from pcq.agent.schema import RecipeSpec
from pcq.examples import models as example_models


SPEC = RecipeSpec(
    name="vision/seg/voc_unet",
    task="segmentation",
    description="Pascal VOC 2012 segmentation with UNet baseline.",
    metrics=["epoch", "train_loss", "train_iou", "eval_loss", "eval_iou"],
    monitor_candidates=[
        {"name": "eval_iou", "mode": "max"},
        {"name": "eval_loss", "mode": "min"},
    ],
    requires_extras=["vision"],
    smoke_safe=False,
    smoke_overrides={
        # VOC 다운로드 우회 — fake_seg 32x32 + 작은 UNet
        "dataset_train": lambda _split: datasets.fake_seg(
            num_samples=16, num_classes=21, image_size=32
        ),
        "dataset_eval": lambda _split: datasets.fake_seg(
            num_samples=8, num_classes=21, image_size=32
        ),
        "model": example_models.unet(in_channels=3, num_classes=21, base_ch=16),
    },
    atoms={
        "model": pcq.model_ref(
            "unet", {"in_channels": 3, "num_classes": 21, "base_ch": 32}
        ),
        "dataset_train": pcq.dataset_ref(
            "voc_seg",
            {
                "root": "data",
                "image_set": "train",
                "download": True,
                "image_size": 256,
            },
        ),
        "dataset_eval": pcq.dataset_ref(
            "voc_seg",
            {
                "root": "data",
                "image_set": "val",
                "download": True,
                "image_size": 256,
            },
        ),
        # ignore_index=-1 — voc_seg 가 void(255) 를 -1 로 변환하여 emit
        "loss": pcq.loss_ref("cross_entropy", {"ignore_index": -1}),
        "optim": pcq.optim_ref("adamw", {"lr": 1e-3, "weight_decay": 1e-4}),
        "sched": pcq.sched_ref("cosine", {"T_max": 50, "warmup": 500}),
    },
    defaults={"epochs": 50, "batch_size": 16},
)


def voc_unet() -> dict:
    """Pascal VOC 2012 segmentation — UNet baseline + cosine warmup."""
    return SPEC.build()
