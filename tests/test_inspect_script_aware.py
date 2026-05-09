"""inspect: detected_imports + cq_calls — v1.13 script-aware AST 분석."""
from __future__ import annotations

from pcq.agent import inspect_project


def _write_cq_yaml(tmp_path) -> None:
    (tmp_path / "cq.yaml").write_text(
        "name: test\n"
        "cmd: uv run python train.py\n"
        "configs:\n  output_dir: output\n"
        "metrics:\n  - epoch\n",
        encoding="utf-8",
    )


def test_inspect_detects_sklearn_imports(tmp_path):
    """sklearn import + pcq.config + pcq.save_all 를 감지."""
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "import joblib\n"
        "cfg = pcq.config()\n"
        "pcq.log(epoch=0, eval_acc=0.9)\n"
        "pcq.save_all(history=[{'epoch': 0, 'eval_acc': 0.9}])\n",
        encoding="utf-8",
    )
    insp = inspect_project(tmp_path)
    assert insp.entrypoint is not None
    assert insp.entrypoint.kind == "script"
    assert "sklearn" in insp.entrypoint.detected_imports
    assert "joblib" in insp.entrypoint.detected_imports
    assert "pcq.config" in insp.entrypoint.cq_calls
    assert "pcq.log" in insp.entrypoint.cq_calls
    assert "pcq.save_all" in insp.entrypoint.cq_calls


def test_inspect_detects_xgboost_with_save_all(tmp_path):
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\n"
        "import xgboost as xgb\n"
        "cfg = pcq.config()\n"
        "pcq.save_all(history=[])\n",
        encoding="utf-8",
    )
    insp = inspect_project(tmp_path)
    assert insp.entrypoint is not None
    assert insp.entrypoint.kind == "script"
    assert "xgboost" in insp.entrypoint.detected_imports


def test_inspect_trainer_still_detected(tmp_path):
    """v1.13 변경 후에도 trainer kind 가 우선 — preset 추출 그대로."""
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\n"
        "import torch\n"
        "cfg = pcq.config()\n"
        "pcq.Trainer(preset='vision/fake_smoke').fit()\n",
        encoding="utf-8",
    )
    insp = inspect_project(tmp_path)
    assert insp.entrypoint is not None
    assert insp.entrypoint.kind == "trainer"
    assert insp.entrypoint.preset == "vision/fake_smoke"
    assert "torch" in insp.entrypoint.detected_imports


def test_inspect_literal_preset_still_wins_over_cfg(tmp_path):
    """Trainer(preset='X') literal이 있으면 그게 우선 (cq.yaml.configs.preset 무시)."""
    (tmp_path / "cq.yaml").write_text("""\
name: t
cmd: uv run python train.py
configs:
  preset: vision/should_not_use_this
metrics:
  - epoch
artifacts:
  - output/
""")
    (tmp_path / "train.py").write_text("""\
import pcq

pcq.Trainer(preset="vision/fake_smoke").fit()
""")
    insp = inspect_project(tmp_path)
    assert insp.entrypoint.preset == "vision/fake_smoke"
