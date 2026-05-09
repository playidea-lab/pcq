"""pcq.agent.validate_run — post-run validation gates (v1.16)."""
import json
from pathlib import Path

from pcq.agent.validate_run import validate_run


def _make_artifacts(
    tmp_path: Path,
    history=None,
    manifest_files=None,
    with_run_record=True,
):
    history = history if history is not None else [{"epoch": 0, "eval_acc": 0.5}]
    (tmp_path / "metrics.json").write_text(json.dumps({"history": history}))
    rs = {
        "schema_version": 1,
        "status": "completed",
        "best": {"epoch": history[0]["epoch"]} if history else None,
        "last": {"epoch": history[-1]["epoch"]} if history else None,
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(rs))
    files = manifest_files if manifest_files is not None else [
        {"path": "metrics.json", "kind": "metrics"}
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files})
    )
    if with_run_record:
        rr = {
            "schema_version": 1,
            "run": {"status": "completed"},
            "execution": {},
            "source": {},
            "environment": {},
            "metrics": {"declared": [], "history_path": "metrics.json"},
            "artifacts": files,
        }
        (tmp_path / "run_record.json").write_text(json.dumps(rr))


def test_validate_run_pass(tmp_path):
    _make_artifacts(tmp_path)
    report = validate_run(tmp_path)
    assert report.status in ("pass", "warn")


def test_validate_run_missing_metrics_fails(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    report = validate_run(tmp_path)
    assert report.status == "fail"
    assert any(c.id == "metrics_present" for c in report.checks)


def test_validate_run_summary_inconsistent_fails(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0}]})
    )
    rs = {"best": {"epoch": 99}, "last": {"epoch": 50}}    # mismatch
    (tmp_path / "run_summary.json").write_text(json.dumps(rs))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    report = validate_run(tmp_path)
    assert any(
        c.id == "summary_metrics_consistent" and c.status == "fail"
        for c in report.checks
    )


def test_validate_run_missing_run_record_warns(tmp_path):
    _make_artifacts(tmp_path, with_run_record=False)
    report = validate_run(tmp_path)
    rr_check = next(
        (c for c in report.checks if c.id == "run_record_present"), None
    )
    assert rr_check is not None
    assert rr_check.status == "warn"


def test_validate_run_reports_strictness_level(tmp_path):
    _make_artifacts(tmp_path)
    report = validate_run(tmp_path, strictness=1)
    d = report.to_dict()
    assert d["strictness"] == 1
    assert d["strictness_name"] == "static"
    strict = next(c for c in report.checks if c.id == "strictness_level")
    assert strict.evidence["level"] == 1


def test_validate_run_missing_run_record_fails_at_strictness3(tmp_path):
    _make_artifacts(tmp_path, with_run_record=False)
    report = validate_run(tmp_path, strictness=3)
    rr_check = next(c for c in report.checks if c.id == "run_record_present")
    assert report.status == "fail"
    assert rr_check.status == "fail"
    assert rr_check.severity == "blocking"


def test_validate_run_strictness3_requires_reproducibility_evidence(tmp_path):
    _make_artifacts(tmp_path)
    report = validate_run(tmp_path, strictness=3)
    assert report.status == "fail"
    assert any(
        c.id == "run_record_execution_identity" and c.status == "fail"
        for c in report.checks
    )
    assert any(
        c.id == "source_reproducibility" and c.status == "fail"
        for c in report.checks
    )


def test_validate_run_strictness4_requires_service_evidence(tmp_path):
    _make_artifacts(tmp_path)
    rr = {
        "schema_version": 1,
        "run": {"name": "t", "status": "completed"},
        "execution": {"cmd": "uv run python train.py"},
        "source": {"git_sha": "abc123", "dirty": False},
        "environment": {"python": "3.12", "platform": "test"},
        "inputs": {"dataset": {"name": "local"}},
        "metrics": {
            "declared": [{"name": "eval_acc"}],
            "history_path": "metrics.json",
        },
        "artifacts": [{"path": "metrics.json", "kind": "metrics"}],
    }
    (tmp_path / "run_record.json").write_text(json.dumps(rr))
    report = validate_run(tmp_path, strictness=4)
    assert report.status == "fail"
    assert any(
        c.id == "service_input_identity" and c.status == "fail"
        for c in report.checks
    )
    assert any(
        c.id == "service_metric_schema" and c.status == "fail"
        for c in report.checks
    )
    assert any(
        c.id == "service_hardware_evidence" and c.status == "fail"
        for c in report.checks
    )


def test_validate_run_manifest_sha_verify(tmp_path):
    """schema v2 manifest sha256 mismatch → fail."""
    content = b"hello"
    (tmp_path / "model.pt").write_bytes(content)
    files = [
        {
            "path": "model.pt",
            "kind": "model",
            "sha256": "0" * 64,
            "size_bytes": 5,
        }
    ]
    _make_artifacts(tmp_path, manifest_files=files)
    # manifest schema v2 로 다시 작성
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "files": files})
    )
    report = validate_run(tmp_path)
    assert any(
        c.id == "manifest_evidence" and c.status == "fail"
        for c in report.checks
    )


def test_validate_run_manifest_v2_sha_match_passes(tmp_path):
    """schema v2 + 정확한 sha256 → pass."""
    import hashlib

    content = b"hello world"
    (tmp_path / "model.pt").write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    files = [
        {
            "path": "model.pt",
            "kind": "model",
            "sha256": sha,
            "size_bytes": len(content),
        }
    ]
    _make_artifacts(tmp_path, manifest_files=files)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "files": files})
    )
    report = validate_run(tmp_path)
    evidence = next(
        c for c in report.checks if c.id == "manifest_evidence"
    )
    assert evidence.status == "pass"


def test_validate_run_run_record_missing_keys_fails(tmp_path):
    """run_record 에 필수 키 빠지면 fail."""
    _make_artifacts(tmp_path, with_run_record=False)
    # 의도적으로 부족한 run_record 작성
    (tmp_path / "run_record.json").write_text(
        json.dumps({"schema_version": 1})
    )
    report = validate_run(tmp_path)
    assert any(
        c.id == "run_record_complete" and c.status == "fail"
        for c in report.checks
    )


def test_validate_run_metrics_invalid_json_fails(tmp_path):
    (tmp_path / "metrics.json").write_text("{not valid json")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": []})
    )
    (tmp_path / "run_summary.json").write_text(json.dumps({}))
    report = validate_run(tmp_path)
    assert report.status == "fail"
    assert any(
        c.id == "metrics_well_formed" and c.status == "fail"
        for c in report.checks
    )


def test_validate_run_manifest_invalid_json_fails(tmp_path):
    (tmp_path / "manifest.json").write_text("{bad")
    (tmp_path / "metrics.json").write_text(
        json.dumps({"history": [{"epoch": 0}]})
    )
    (tmp_path / "run_summary.json").write_text(json.dumps({}))
    report = validate_run(tmp_path)
    assert any(
        c.id == "manifest_parseable" and c.status == "fail"
        for c in report.checks
    )
