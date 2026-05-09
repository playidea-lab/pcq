"""Recipe catalog smoke tests — 각 recipe import + dict 구조 검증."""
from __future__ import annotations

import pytest

from pcq import Trainer
from pcq.trainer import _import_recipe

# minimum recipe contract — 모든 recipe 가 갖춰야 하는 필수 키
_REQUIRED_KEYS = {"model", "loss"}

# 7 종 recipe 풀세트 (Trainer.list_presets() 기대값) — v1.3 부터 segmentation 2종 추가
_EXPECTED_PRESETS = {
    "vision/cifar10_smallcnn_baseline",
    "vision/fake_smoke",
    "vision/mnist_mlp",
    "vision/cifar10_resnet18",
    "nlp/fake_text_classifier",
    "vision/seg/fake_seg_smoke",
    "vision/seg/voc_unet",
}


def test_list_presets_has_all_recipes():
    presets = set(Trainer.list_presets())
    missing = _EXPECTED_PRESETS - presets
    assert not missing, f"missing recipes: {missing}"


@pytest.mark.parametrize(
    "recipe_name",
    [
        "vision/fake_smoke",
        "vision/mnist_mlp",
        "vision/cifar10_resnet18",
        "vision/cifar10_smallcnn_baseline",
        "nlp/fake_text_classifier",
    ],
)
def test_recipe_returns_valid_dict(recipe_name):
    """각 recipe 가 callable + dict 반환 + model/loss/dataset 키 보유."""
    # mnist/cifar10/resnet18 은 torchvision 필요 — 없으면 skip
    if recipe_name in (
        "vision/mnist_mlp",
        "vision/cifar10_resnet18",
        "vision/cifar10_smallcnn_baseline",
    ):
        pytest.importorskip("torchvision")

    fn = _import_recipe(recipe_name)
    d = fn()
    assert isinstance(d, dict)
    assert _REQUIRED_KEYS.issubset(d.keys()), (
        f"{recipe_name} missing keys: {_REQUIRED_KEYS - d.keys()}"
    )
    # dataset 키: 공용 'dataset' 또는 split 별 'dataset_train'
    assert "dataset" in d or "dataset_train" in d


def test_fake_smoke_recipe_actually_trains(tmp_path):
    """vision/fake_smoke 는 외부 dep 없으니 실제로 1 epoch 학습 가능."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    Trainer(preset="vision/fake_smoke", cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()


def test_nlp_fake_text_recipe_actually_trains(tmp_path):
    """nlp/fake_text_classifier 도 외부 dep 없이 학습 가능 (int64 token)."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    Trainer(preset="nlp/fake_text_classifier", cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()


def test_print_recipe_for_each(capsys):
    for name in [
        "vision/fake_smoke",
        "nlp/fake_text_classifier",
    ]:
        Trainer.print_recipe(name)
    out = capsys.readouterr().out
    assert "vision/fake_smoke" in out
    assert "nlp/fake_text_classifier" in out


# ── v1.3: Segmentation recipes ──────────────────────────────────────────────
def test_seg_recipes_in_preset_list():
    """3-level nested recipe (vision/seg/...) 도 list_presets 에 포함."""
    presets = Trainer.list_presets()
    assert "vision/seg/fake_seg_smoke" in presets
    assert "vision/seg/voc_unet" in presets


def test_fake_seg_smoke_recipe_returns_valid_dict():
    fn = _import_recipe("vision/seg/fake_seg_smoke")
    d = fn()
    assert d.get("task") == "segmentation"
    assert "model" in d and "loss" in d
    assert "dataset_train" in d


def test_fake_seg_smoke_recipe_actually_trains(tmp_path):
    """fake_seg + UNet — 외부 dep 없이 실제 학습 가능."""
    cfg = {
        "output_dir": str(tmp_path),
        "epochs": 1,
        "batch_size": 4,
        "device": "cpu",
    }
    Trainer(preset="vision/seg/fake_seg_smoke", cfg=cfg).fit()
    assert (tmp_path / "model.pt").exists()
    # iou 메트릭이 stdout 에 송출되었는지는 별도로 확인 가능


def test_voc_unet_recipe_returns_valid_dict():
    """voc_unet 은 torchvision 필요 — recipe import 자체는 dep 없이 동작."""
    fn = _import_recipe("vision/seg/voc_unet")
    d = fn()
    assert d.get("task") == "segmentation"
    assert "sched_factory" in d
