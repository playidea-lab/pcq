"""init-experiment --style {trainer|experiment|script} 분기 검증."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from pcq.agent import init_experiment


def test_default_style_is_trainer(tmp_path):
    """style 미지정 시 trainer (preset 필수)."""
    result = init_experiment(tmp_path, preset="vision/fake_smoke")
    assert result.style == "trainer"
    assert (tmp_path / "pcq_atoms.py").exists()
    text = (tmp_path / "train.py").read_text()
    assert "Trainer.from_cfg" in text


def test_experiment_style(tmp_path):
    """experiment style — Experiment subclass + atom infra."""
    result = init_experiment(
        tmp_path, preset=None, style="experiment", force=True,
    )
    assert result.style == "experiment"
    text = (tmp_path / "train.py").read_text()
    assert "class MyExperiment" in text
    assert "Experiment" in text
    # atom infra 도 생성 (experiment 도 atom 사용 가능)
    assert (tmp_path / "pcq_atoms.py").exists()
    assert (tmp_path / "atoms" / "__init__.py").exists()


def test_script_style_no_preset(tmp_path):
    """script style — preset 없이도 init 가능, atom infra 미생성."""
    result = init_experiment(
        tmp_path, preset=None, style="script", force=True,
    )
    assert result.style == "script"
    train_text = (tmp_path / "train.py").read_text()
    assert "pcq.config()" in train_text
    assert "pcq.save_all" in train_text
    # script style 은 atoms / recipes 디렉터리 미생성
    assert not (tmp_path / "pcq_atoms.py").exists()
    assert not (tmp_path / "recipes").exists()
    assert not (tmp_path / "atoms").exists()


def test_unknown_style_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown style"):
        init_experiment(tmp_path, preset=None, style="bogus", force=True)


def test_script_style_train_runnable(tmp_path):
    """script style train.py 가 실제 실행 가능 (placeholder acc=0.5)."""
    init_experiment(tmp_path, preset=None, style="script", force=True)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({
        "output_dir": str(tmp_path / "out"),
        "seed": 42,
        "_metrics_declared": ["epoch", "eval_acc"],
    }))
    env = {**os.environ, "CQ_CONFIG_JSON": str(cfg_path)}
    proc = subprocess.run(
        [sys.executable, "train.py"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"script train.py failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    out_dir = tmp_path / "out"
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_summary.json").exists()


def test_init_with_pyproject_generates_file(tmp_path):
    """v2.1: --with-pyproject flag → pyproject.toml 생성 + pcq dep + preset extras.

    v3.0.0: PyPI distribution, Python import, and CLI command are all `pcq`.
    """
    result = init_experiment(
        tmp_path, preset="vision/mnist_mlp", style="trainer",
        force=True, with_pyproject=True,
    )
    assert "pyproject.toml" in result.files_created
    py = (tmp_path / "pyproject.toml").read_text()
    assert "pcq>=" in py
    # vision/mnist_mlp는 vision extras 필요 (torchvision)
    assert "pcq[vision]" in py
    # v2.1.1: non-package mode — hatchling wheel build 없이 uv lock/sync 가능
    assert "[tool.uv]" in py
    assert "package = false" in py
    # v2.1.1: hatchling [build-system] 블록 없음 — fresh user `uv sync` 빌드 실패 방지
    assert "[build-system]" not in py
    assert "hatchling" not in py


def test_init_with_pyproject_is_valid_toml(tmp_path):
    """v2.1.1: 생성된 pyproject.toml이 실제로 파싱 가능 + uv가 받아들이는 구조."""
    import tomllib

    init_experiment(
        tmp_path, preset="vision/fake_smoke", style="trainer",
        force=True, with_pyproject=True,
    )
    text = (tmp_path / "pyproject.toml").read_text()
    parsed = tomllib.loads(text)
    assert parsed["project"]["name"]
    # v3.0.0: PyPI distribution, Python import, and CLI command are all `pcq`.
    assert "pcq" in str(parsed["project"]["dependencies"])
    # uv non-package mode
    assert parsed.get("tool", {}).get("uv", {}).get("package") is False
    # v3.0.1: pcq is on PyPI; generated template no longer needs a git source.
    assert "sources" not in parsed.get("tool", {}).get("uv", {})


def test_init_without_pyproject_default(tmp_path):
    """default는 pyproject.toml 안 만듦 (사용자 본인 pyproject 있을 가능성)."""
    result = init_experiment(
        tmp_path, preset="vision/fake_smoke", style="trainer",
        force=True,
    )
    assert "pyproject.toml" not in result.files_created
    assert not (tmp_path / "pyproject.toml").exists()


def test_init_with_pyproject_no_extras_for_torch_only_recipe(tmp_path):
    """vision/fake_smoke는 requires_extras=[]. pcq[]는 안 들어감."""
    init_experiment(
        tmp_path, preset="vision/fake_smoke", style="trainer",
        force=True, with_pyproject=True,
    )
    py = (tmp_path / "pyproject.toml").read_text()
    # v3.0.0: distribution renamed to `pcq`.
    assert "pcq>=" in py
    assert "pcq[" not in py    # extras 라인 없음


def test_init_with_pyproject_skip_existing_without_force(tmp_path):
    (tmp_path / "pyproject.toml").write_text("# existing\n")
    result = init_experiment(
        tmp_path, preset="vision/fake_smoke", style="trainer",
        with_pyproject=True,
    )
    assert "pyproject.toml" in result.files_skipped
