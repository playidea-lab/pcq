"""summarize_run + build_run_summary 테스트.

v4.0: Trainer 제거. pcq.save_run_summary / save_metrics 로 직접 작성.
"""
from __future__ import annotations

import json

import pcq
from pcq.agent import summarize_run


def _build_completed_run(tmp_path, epochs: int = 1, monitor: str = "eval_loss"):
    """contract artifact 세트 생성."""
    import os
    import json as _json
    history = [
        {"epoch": e, monitor: 1.0 - 0.1 * e, "eval_acc": 0.5 + 0.1 * e}
        for e in range(epochs)
    ]
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(_json.dumps({
        "output_dir": str(tmp_path),
        "monitor": monitor,
        "mode": "min",
        "epochs": epochs,
        "seed": 42,
    }))
    prev = os.environ.get("CQ_CONFIG_JSON")
    os.environ["CQ_CONFIG_JSON"] = str(cfg_path)
    try:
        pcq.save_metrics(history)
        pcq.save_run_summary(history=history, status="completed")
        pcq.save_manifest()
        pcq.save_config_snapshot()
    finally:
        if prev is None:
            os.environ.pop("CQ_CONFIG_JSON", None)
        else:
            os.environ["CQ_CONFIG_JSON"] = prev


def test_run_summary_json_written(tmp_path):
    """save_run_summary 후 output/run_summary.json 생성."""
    _build_completed_run(tmp_path, epochs=2)
    summary_path = tmp_path / "run_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["schema_version"] == 1
    assert data["status"] == "completed"
    assert data["last"]["epoch"] == 1


def test_run_summary_in_manifest(tmp_path):
    """manifest.json 의 files 에 run_summary.json kind=summary 포함."""
    _build_completed_run(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    paths = {f["path"]: f["kind"] for f in manifest["files"]}
    assert "run_summary.json" in paths
    assert paths["run_summary.json"] == "summary"


def test_summarize_run_reads_existing_summary(tmp_path):
    """run_summary.json 있으면 그대로 로드."""
    _build_completed_run(tmp_path)
    summary = summarize_run(tmp_path)
    assert summary.status == "completed"
    assert summary.last is not None


def test_summarize_run_synthesizes_when_no_summary_file(tmp_path):
    """run_summary.json 없으면 metrics.json + config.json 에서 합성."""
    _build_completed_run(tmp_path)
    (tmp_path / "run_summary.json").unlink()
    summary = summarize_run(tmp_path)
    assert any("synthesized" in w for w in summary.warnings)
    assert summary.last is not None


def test_summarize_run_missing_metrics_failure(tmp_path):
    """output_dir 있지만 metrics.json 없음 → status=failed."""
    summary = summarize_run(tmp_path)
    assert summary.status == "failed"
    assert summary.failure is not None


def test_summarize_run_missing_output_dir_unknown(tmp_path):
    """output_dir 자체가 없음 → status=unknown."""
    summary = summarize_run(tmp_path / "nonexistent")
    assert summary.status == "unknown"


def test_run_summary_to_dict_round_trip(tmp_path):
    """run_summary.json 의 dict 가 그대로 RunSummary 로 다시 로드 가능."""
    _build_completed_run(tmp_path)
    summary = summarize_run(tmp_path)
    d = summary.to_dict()
    assert d["schema_version"] == 1
    assert d["status"] == "completed"
