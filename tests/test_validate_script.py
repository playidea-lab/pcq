"""validate: script-aware gates — v1.13."""
from __future__ import annotations

from pcq.agent import validate_project


def _write_cq_yaml(tmp_path) -> None:
    (tmp_path / "cq.yaml").write_text(
        "name: test\n"
        "cmd: uv run python train.py\n"
        "configs:\n  output_dir: output\n"
        "metrics:\n  - epoch\n  - eval_acc\n"
        "artifacts:\n  - output/\n",
        encoding="utf-8",
    )


def test_script_missing_cq_config_fails(tmp_path):
    """contract script 가 pcq.config() 없이는 blocking fail."""
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\nprint('hello')\n",
        encoding="utf-8",
    )
    report = validate_project(tmp_path)
    # script 로 분류되도록 pcq.config 가 없어도 cq import 됨 → kind=script
    config_check = next(
        (c for c in report.checks if c.id == "cq_config_called"),
        None,
    )
    assert config_check is not None
    assert config_check.status == "fail"
    assert config_check.severity == "blocking"


def test_script_missing_log_warns(tmp_path):
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\n"
        "cfg = pcq.config()\n"
        "pcq.save_all(history=[])\n",
        encoding="utf-8",
    )
    report = validate_project(tmp_path)
    log_check = next(
        (c for c in report.checks if c.id == "cq_log_called"),
        None,
    )
    assert log_check is not None
    assert log_check.status == "warn"


def test_script_missing_save_warns(tmp_path):
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\n"
        "cfg = pcq.config()\n"
        "pcq.log(epoch=0, eval_acc=0.5)\n",
        encoding="utf-8",
    )
    report = validate_project(tmp_path)
    save_check = next(
        (c for c in report.checks if c.id == "standard_artifacts_helper"),
        None,
    )
    assert save_check is not None
    assert save_check.status == "warn"


def test_script_complete_passes(tmp_path):
    """모든 contract 호출 다 있으면 — script gate 들 pass."""
    _write_cq_yaml(tmp_path)
    (tmp_path / "train.py").write_text(
        "import pcq\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "cfg = pcq.config()\n"
        "pcq.log(epoch=0, eval_acc=0.9)\n"
        "pcq.save_all(history=[{'epoch': 0, 'eval_acc': 0.9}])\n",
        encoding="utf-8",
    )
    report = validate_project(tmp_path)
    # 3 개 핵심 script gate 모두 pass
    config_check = next(c for c in report.checks if c.id == "cq_config_called")
    log_check = next(c for c in report.checks if c.id == "cq_log_called")
    save_check = next(
        c for c in report.checks if c.id == "standard_artifacts_helper"
    )
    assert config_check.status == "pass"
    assert log_check.status == "pass"
    assert save_check.status == "pass"
    # detected_frameworks info
    fw_check = next(
        c for c in report.checks if c.id == "detected_frameworks"
    )
    assert "sklearn" in fw_check.detail
