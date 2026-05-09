"""describe_run — compact RunRecord summary (v1.17)."""
import json
from pathlib import Path

import pcq
from pcq.agent.describe import RunDescription, describe_run


def _setup_cfg(tmp_path: Path, **extra) -> Path:
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg.update(extra)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_describe_run_with_no_record_returns_no_record_status(tmp_path):
    desc = describe_run(tmp_path)
    assert desc.status == "no_record"
    assert isinstance(desc, RunDescription)


def test_describe_run_with_corrupted_json_returns_corrupted(tmp_path):
    (tmp_path / "run_record.json").write_text("{invalid json")
    desc = describe_run(tmp_path)
    assert desc.status == "corrupted"


def test_describe_run_extracts_basic_fields(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(tmp_path, monitor="eval_iou", mode="max")),
    )
    history = [
        {"epoch": 0, "eval_iou": 0.5},
        {"epoch": 1, "eval_iou": 0.7},
    ]
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history, status="completed")
    pcq.save_manifest()
    pcq.finalize_run(history=history)

    desc = describe_run(tmp_path)
    assert desc.status == "completed"
    assert desc.target_metric == "eval_iou"
    assert desc.mode == "max"
    assert desc.best == {
        "epoch": 1,
        "value": 0.7,
        "metrics": {"eval_iou": 0.7},
        "checkpoint": "best.ckpt",
    }
    assert desc.best_value == 0.7
    assert desc.best_epoch == 1
    assert desc.last == {
        "epoch": 1,
        "value": 0.7,
        "metrics": {"eval_iou": 0.7},
        "checkpoint": "last.ckpt",
    }
    assert desc.last_value == 0.7
    assert desc.last_epoch == 1
    assert desc.epochs_completed == 2
    assert desc.python   # populated from environment
    assert desc.platform
    assert desc.decision_facts["run_completed"] is True
    assert desc.decision_facts["has_best"] is True
    assert desc.decision_facts["has_target_metric"] is True


def test_describe_run_to_dict_drops_empty_optional_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])

    desc = describe_run(tmp_path)
    out = desc.to_dict()
    # 빈 list / dict / None 은 to_dict 에서 제거.
    assert "inputs_summary" not in out
    assert "artifacts_summary" not in out or isinstance(
        out.get("artifacts_summary"), dict
    )
    assert "failure" not in out
    # always-keep keys 은 유지 (status, schema_version 등).
    assert "status" in out
    assert "schema_version" in out


def test_describe_run_inputs_summary_from_cq_yaml(tmp_path, monkeypatch):
    """cq.yaml 의 inputs section 을 inputs_summary 로 요약."""
    (tmp_path / "cq.yaml").write_text(
        """
name: t
cmd: c
configs: {}
metrics: [eval_acc]
artifacts: [output/]
inputs:
  dataset:
    name: dental
    version: v12
  pretrained:
    name: dinov3
"""
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))
    pcq.save_metrics([])
    pcq.save_run_summary(history=[])
    pcq.save_manifest()
    pcq.finalize_run(history=[])

    desc = describe_run(out_dir)
    assert "dataset:dental@v12" in desc.inputs_summary
    assert "pretrained:dinov3" in desc.inputs_summary


def test_describe_run_artifacts_summary_counts_by_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    (tmp_path / "model.pt").write_bytes(b"fake")
    (tmp_path / "best.ckpt").write_bytes(b"ck")
    (tmp_path / "last.ckpt").write_bytes(b"ck")
    pcq.save_metrics([{"epoch": 0}])
    pcq.save_run_summary(history=[{"epoch": 0}])
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0}])

    desc = describe_run(tmp_path)
    # model.pt → 'model', best.ckpt + last.ckpt → 'checkpoint' x2
    assert desc.artifacts_summary.get("model", 0) >= 1
    assert desc.artifacts_summary.get("checkpoint", 0) >= 2


def test_describe_run_validation_status_propagated(tmp_path, monkeypatch):
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    pcq.save_metrics([{"epoch": 0, "loss": 0.1}])
    pcq.save_run_summary(history=[{"epoch": 0, "loss": 0.1}])
    pcq.save_manifest()
    pcq.finalize_run(history=[{"epoch": 0, "loss": 0.1}])

    desc = describe_run(tmp_path)
    assert desc.validation_status in ("pass", "warn", "fail")


def test_describe_run_failure_extracted_from_run_summary(tmp_path, monkeypatch):
    """run_summary.json 의 failure dict 가 description.failure 에 들어감."""
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(tmp_path)))
    failure = {"category": "oom", "message": "CUDA out of memory"}
    pcq.save_metrics([])
    pcq.save_run_summary(history=[], status="failed", failure=failure)
    pcq.save_manifest()
    pcq.finalize_run(history=[], status="failed", failure=failure)

    desc = describe_run(tmp_path)
    assert desc.failure is not None
    assert desc.failure.get("category") == "oom"
    assert desc.failure.get("error_code") == "ERR_OUT_OF_MEMORY"
    assert desc.decision_facts["has_failure"] is True


def test_describe_run_includes_decision_evidence(tmp_path, monkeypatch):
    """agent 가 다음 판단에 필요한 사실들을 한 번에 읽을 수 있어야 한다."""
    (tmp_path / "cq.yaml").write_text(
        """
name: decision-demo
cmd: uv run python train.py
configs:
  seed: 42
  strictness: 3
metrics:
  eval_acc:
    mode: max
artifacts:
  - output/
inputs:
  dataset:
    uri: cq://datasets/digits
    sha256: abc123
"""
    )
    (tmp_path / "uv.lock").write_text("# lock\n", encoding="utf-8")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "CQ_CONFIG_JSON",
        str(_setup_cfg(
            out_dir,
            monitor="eval_acc",
            mode="max",
            strictness=3,
            _parent_run_id="run_parent",
            _parent_run_path="../parent/output",
        )),
    )
    history = [
        {"epoch": 0, "eval_acc": 0.6, "loss": 1.0},
        {"epoch": 1, "eval_acc": 0.8, "loss": 0.5},
    ]
    (out_dir / "model.pt").write_bytes(b"model")
    pcq.save_metrics(history)
    pcq.save_run_summary(history=history, status="completed")
    pcq.save_manifest()
    pcq.finalize_run(history=history)

    desc = describe_run(out_dir)
    data = desc.to_dict()

    assert data["target_metric"] == "eval_acc"
    assert data["mode"] == "max"
    assert data["parent_run_id"] == "run_parent"
    assert data["parent_run_path"] == "../parent/output"
    assert data["metrics_declared"] == [{"name": "eval_acc", "mode": "max"}]
    assert data["artifacts"]
    assert data["reproducibility_evidence"]["environment"]["lockfile"] == "uv.lock"
    assert data["reproducibility_evidence"]["config"]["seed"] == 42
    assert data["reproducibility_evidence"]["config"]["strictness"] == 3
    assert data["reproducibility_evidence"]["inputs"]["count"] == 1
    assert data["decision_facts"]["has_parent"] is True
    assert data["decision_facts"]["artifact_count"] == len(data["artifacts"])
    assert data["decision_facts"]["metric_count"] == 1
    assert data["decision_facts"]["input_count"] == 1
