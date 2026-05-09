"""Tests for pcq.Trainer high-level API (T-CQPY-005)."""
from __future__ import annotations

import pytest

import pcq
from pcq import Trainer


def test_trainer_atom_only_no_preset(tmp_path):
    """Case C: atom names only — Trainer composes a working experiment from defaults."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    t = Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg)
    t.fit()
    assert len(t.history) == 1


def test_trainer_atom_objects_passed_directly(tmp_path):
    """Case D: pass atom objects/factories directly via overrides."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    fake_ds = pcq.datasets.fake(num_samples=32, num_classes=3, image_size=8)
    t = Trainer(
        model=pcq.models.mlp(3 * 8 * 8, [16], 3),
        dataset_train=lambda split: fake_ds,
        dataset_eval=lambda split: fake_ds,
        loss=pcq.loss.cross_entropy(),
        optim_factory=lambda p: pcq.optim.adamw(p, lr=1e-2),
        cfg=cfg,
    )
    t.fit()
    assert len(t.history) == 1


def _fake_cifar_like(split):
    # cifar10 대체용 fake 데이터셋 (네트워크/디스크 없이 preset smoke 테스트)
    return pcq.datasets.fake(
        num_samples=32, num_classes=10, image_size=32, channels=3
    )


def test_trainer_with_preset_smoke(tmp_path):
    """Case A: preset only — substitute fake for cifar10 to skip download."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    t = Trainer(
        preset="vision/cifar10_smallcnn_baseline",
        dataset_train=_fake_cifar_like,
        dataset_eval=_fake_cifar_like,
        cfg=cfg,
    )
    t.fit()
    assert len(t.history) == 1


def test_trainer_override_replaces_atom(tmp_path):
    """Case B: preset + override — override replaces preset's value."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    custom_sched_called: list[bool] = []

    def custom_sched(o):
        custom_sched_called.append(True)
        return pcq.sched.cosine(o, T_max=20, warmup=10)

    t = Trainer(
        preset="vision/cifar10_smallcnn_baseline",
        dataset_train=_fake_cifar_like,
        dataset_eval=_fake_cifar_like,
        sched_factory=custom_sched,
        cfg=cfg,
    )
    t.fit()
    assert custom_sched_called  # 사용자 override 가 실제 호출됨


def test_trainer_list_presets_contains_baseline():
    presets = Trainer.list_presets()
    assert "vision/cifar10_smallcnn_baseline" in presets


def test_trainer_print_recipe_shows_dict_keys(capsys):
    Trainer.print_recipe("vision/cifar10_smallcnn_baseline")
    out = capsys.readouterr().out
    # 헤더 + 주요 atom key 들이 보여야 함
    assert "vision/cifar10_smallcnn_baseline" in out
    assert "model" in out
    assert "loss" in out


def test_trainer_unknown_preset_raises():
    with pytest.raises(ValueError, match="recipe"):
        Trainer(preset="vision/does_not_exist")


def test_trainer_unknown_model_raises():
    with pytest.raises(ValueError, match="model"):
        Trainer(task="classification", dataset="fake", model="not_a_model")
