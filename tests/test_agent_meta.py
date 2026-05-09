"""Agent metadata primitives — recipe_meta, diff_recipes, dry_run, provenance."""
from __future__ import annotations

import json
import time

import pytest

import pcq


def test_recipe_meta_returns_required_keys():
    meta = pcq.recipe_meta("vision/fake_smoke")
    for key in (
        "name",
        "task",
        "declared_metrics",
        "requires_extras",
        "smoke_safe",
        "has_smoke_overrides",
        "atoms",
        "epochs",
        "batch_size",
    ):
        assert key in meta, f"missing key: {key}"


def test_recipe_meta_atom_summary_includes_model_class():
    meta = pcq.recipe_meta("vision/seg/fake_seg_smoke")
    assert "model" in meta["atoms"]
    # _UNet (torch.nn.Module subclass) 인스턴스
    assert "UNet" in meta["atoms"]["model"]


def test_recipe_meta_no_side_effects_fast():
    """recipe_meta 는 학습 안 함 — 빠르게 끝나야 (sec 단위 budget)."""
    t0 = time.time()
    pcq.recipe_meta("vision/fake_smoke")
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"recipe_meta too slow: {elapsed:.2f}s"


def test_recipe_meta_handles_missing_extras_gracefully():
    """torchvision 미설치 시 import_error 필드만 있는 degraded meta 반환."""
    try:
        import torchvision  # noqa: F401
    except ImportError:
        meta = pcq.recipe_meta("vision/cifar10_resnet18")
        assert "import_error" in meta
        assert "torchvision" in meta["import_error"]
    else:
        pytest.skip("torchvision installed — degradation path not testable")


def test_diff_recipes_classification_vs_segmentation():
    diff = pcq.diff_recipes("vision/fake_smoke", "vision/seg/fake_seg_smoke")
    assert "task" in diff["diff"]
    assert diff["diff"]["task"]["a"] == "classification"
    assert diff["diff"]["task"]["b"] == "segmentation"


def test_diff_recipes_same_recipe_empty_diff():
    diff = pcq.diff_recipes("vision/fake_smoke", "vision/fake_smoke")
    assert diff["diff"] == {}


def test_dry_run_returns_plan_without_training(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 5, "batch_size": 32}
    trainer = pcq.Trainer(preset="vision/fake_smoke", cfg=cfg)
    plan = trainer.dry_run()
    assert plan["epochs"] == 5
    assert plan["batch_size"] == 32
    assert "atoms" in plan
    assert "model" in plan["atoms"]
    assert "expected_artifacts" in plan
    assert "model.pt" in plan["expected_artifacts"]
    # 학습 X — output_dir 에 model.pt 가 없어야 한다
    assert not (tmp_path / "model.pt").exists()


def test_dry_run_records_preset_and_overrides(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    trainer = pcq.Trainer(
        preset="vision/fake_smoke",
        sched_factory=lambda o: pcq.sched.cosine(o, T_max=5),
        cfg=cfg,
    )
    plan = trainer.dry_run()
    assert plan["preset"] == "vision/fake_smoke"
    assert "sched_factory" in plan["overrides"]


def test_provenance_recorded_in_config_json(tmp_path):
    """config.json 에 _recipe / _pcq_version / _git_sha 자동 기록."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(preset="vision/fake_smoke", cfg=cfg).fit()
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved.get("_recipe") == "vision/fake_smoke"
    assert "_pcq_version" in saved
    assert saved["_pcq_version"] == pcq.__version__
    assert "_git_sha" in saved


def test_provenance_overrides_recorded_in_config(tmp_path):
    """override 한 atom 키 목록이 config.json 의 _overrides 로 기록."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(
        preset="vision/fake_smoke",
        sched_factory=lambda o: pcq.sched.cosine(o, T_max=5),
        cfg=cfg,
    ).fit()
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "_overrides" in saved
    assert "sched_factory" in saved["_overrides"]


def test_provenance_no_preset_no_recipe_field(tmp_path):
    """preset 없이 atom-only 모드 — _recipe 필드는 없어야 한다."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(
        task="classification", dataset="fake", model="mlp", cfg=cfg
    ).fit()
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "_recipe" not in saved
    # task/dataset/model 은 override 로 추적
    assert "_overrides" in saved
    assert set(saved["_overrides"]) >= {"model", "dataset", "task"}


def test_list_meta_returns_all_registered_recipes():
    metas = pcq.agent.list_meta()
    assert len(metas) >= 7  # v1.3 까지 7개 recipe
    names = {m["name"] for m in metas}
    assert "vision/fake_smoke" in names
    assert "vision/seg/fake_seg_smoke" in names
    assert "nlp/fake_text_classifier" in names


def test_list_meta_each_has_name():
    for m in pcq.agent.list_meta():
        assert "name" in m
        assert isinstance(m["name"], str)
