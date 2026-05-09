"""init-experiment scaffolding — v1.10."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from pcq.agent import init_experiment


def test_init_creates_files(tmp_path):
    result = init_experiment(tmp_path, preset="vision/fake_smoke")
    assert "cq.yaml" in result.files_created
    assert "train.py" in result.files_created
    assert "recipes/local.py" in result.files_created
    assert (tmp_path / "cq.yaml").exists()
    assert (tmp_path / "train.py").exists()
    assert (tmp_path / "recipes" / "local.py").exists()


def test_init_skips_existing_without_force(tmp_path):
    (tmp_path / "cq.yaml").write_text("# pre-existing\n")
    result = init_experiment(tmp_path, preset="vision/fake_smoke")
    assert "cq.yaml" in result.files_skipped
    assert "cq.yaml" not in result.files_created


def test_init_force_overwrites(tmp_path):
    (tmp_path / "cq.yaml").write_text("# pre-existing\n")
    result = init_experiment(tmp_path, preset="vision/fake_smoke", force=True)
    assert "cq.yaml" in result.files_created
    text = (tmp_path / "cq.yaml").read_text()
    assert "vision/fake_smoke" in text


def test_init_cq_yaml_contains_preset(tmp_path):
    init_experiment(tmp_path, preset="vision/fake_smoke")
    text = (tmp_path / "cq.yaml").read_text()
    assert "vision/fake_smoke" in text
    # cmd 는 train.py 호출
    assert "train.py" in text


def test_init_default_name_from_preset(tmp_path):
    result = init_experiment(tmp_path, preset="vision/fake_smoke")
    # / 가 - 로
    assert result.name == "vision-fake_smoke"


def test_init_custom_name(tmp_path):
    result = init_experiment(
        tmp_path, preset="vision/fake_smoke", name="my-experiment",
    )
    assert result.name == "my-experiment"
    text = (tmp_path / "cq.yaml").read_text()
    assert "my-experiment" in text


def test_init_to_dict_is_json_safe(tmp_path):
    result = init_experiment(tmp_path, preset="vision/fake_smoke")
    json.dumps(result.to_dict())


def test_init_then_train_actually_runs(tmp_path):
    """Init 한 프로젝트가 실제로 학습 가능 — Trainer.from_cfg 통합 검증."""
    init_experiment(tmp_path, preset="vision/fake_smoke", force=True)
    # 작은 cfg 로 학습 (epochs=1)
    cfg_path = tmp_path / "smoke_cfg.json"
    cfg_path.write_text(
        json.dumps({
            "preset": "vision/fake_smoke",
            "output_dir": str(tmp_path / "out"),
            "epochs": 1,
            "batch_size": 8,
            "seed": 42,
            "_metrics_declared": [
                "epoch", "train_loss", "train_acc", "eval_loss", "eval_acc",
            ],
        })
    )
    env = {**os.environ, "CQ_CONFIG_JSON": str(cfg_path)}
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"train.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (tmp_path / "out" / "model.pt").exists()
