"""summarize_run + build_run_summary 테스트."""
from __future__ import annotations

import json

import pcq
from pcq.agent import summarize_run


def test_run_summary_json_written_by_fit(tmp_path):
    """fit() 종료 후 output/run_summary.json 자동 생성."""
    cfg = {"output_dir": str(tmp_path), "epochs": 2, "batch_size": 16, "seed": 42}
    pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()
    summary_path = tmp_path / "run_summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["schema_version"] == 1
    assert data["status"] == "completed"
    assert data["last"]["epoch"] == 1


def test_run_summary_in_manifest(tmp_path):
    """manifest.json 의 files 에 run_summary.json kind=summary 포함."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    paths = {f["path"]: f["kind"] for f in manifest["files"]}
    assert "run_summary.json" in paths
    assert paths["run_summary.json"] == "summary"


def test_summarize_run_reads_existing_summary(tmp_path):
    """run_summary.json 있으면 그대로 로드."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()
    summary = summarize_run(tmp_path)
    assert summary.status == "completed"
    assert summary.last is not None


def test_summarize_run_synthesizes_when_no_summary_file(tmp_path):
    """run_summary.json 없으면 metrics.json + config.json 에서 합성."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()
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


def test_run_summary_includes_provenance(tmp_path):
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(preset="vision/fake_smoke", cfg=cfg).fit()
    data = json.loads((tmp_path / "run_summary.json").read_text())
    assert data["recipe"] == "vision/fake_smoke"
    assert "provenance" in data
    assert data["provenance"].get("pcq_version") == pcq.__version__


def test_run_summary_to_dict_round_trip(tmp_path):
    """run_summary.json 의 dict 가 그대로 RunSummary 로 다시 로드 가능."""
    cfg = {"output_dir": str(tmp_path), "epochs": 1, "batch_size": 16}
    pcq.Trainer(task="classification", dataset="fake", model="mlp", cfg=cfg).fit()
    summary = summarize_run(tmp_path)
    d = summary.to_dict()
    assert d["schema_version"] == 1
    assert d["status"] == "completed"
