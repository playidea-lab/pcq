"""Registry / 확장성 테스트."""
from __future__ import annotations

import pytest

import pcq
from pcq import Trainer


def test_register_model_via_function():
    pcq.register_model("smoke_mlp", lambda: pcq.models.mlp(3 * 8 * 8, [8], 2))
    assert "smoke_mlp" in Trainer.list_models()


def test_register_model_via_decorator():
    @pcq.register_model("smoke_mlp_dec")
    def factory():
        return pcq.models.mlp(3 * 8 * 8, [8], 2)

    assert "smoke_mlp_dec" in Trainer.list_models()


def test_unknown_model_clear_error():
    with pytest.raises(ValueError, match="unknown model"):
        Trainer(task="classification", dataset="fake", model="does_not_exist")


def test_register_dataset_via_function(tmp_path):
    pcq.register_dataset(
        "smoke_fake_ds",
        lambda _split: pcq.datasets.fake(num_samples=16, num_classes=2, image_size=8),
    )
    pcq.register_model(
        "smoke_mlp_for_ds", lambda: pcq.models.mlp(3 * 8 * 8, [8], 2)
    )
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 8}
    Trainer(
        task="classification",
        dataset="smoke_fake_ds",
        model="smoke_mlp_for_ds",
        cfg=cfg,
    ).fit()
    assert (tmp_path / "model.pt").exists()


def test_register_metric_via_decorator():
    @pcq.register_metric("custom_zero")
    def zero(logits, labels):
        import torch

        return torch.tensor(0.0)

    assert "custom_zero" in Trainer.list_metrics()


def test_list_models_includes_builtins():
    models = Trainer.list_models()
    for builtin in (
        "mlp",
        "small_cnn",
        "resnet18",
        "text_classifier",
        "unet",
        "deeplab_v3",
    ):
        assert builtin in models, f"missing builtin: {builtin}"


def test_list_datasets_includes_builtins():
    datasets = Trainer.list_datasets()
    for builtin in ("fake", "fake_text", "cifar10", "mnist", "fake_seg", "voc_seg"):
        assert builtin in datasets, f"missing builtin: {builtin}"


def test_list_metrics_includes_builtins():
    metrics = Trainer.list_metrics()
    for builtin in (
        "accuracy",
        "top_5_accuracy",
        "mse",
        "mae",
        "iou",
        "dice_score",
        "pixel_accuracy",
    ):
        assert builtin in metrics, f"missing builtin: {builtin}"
