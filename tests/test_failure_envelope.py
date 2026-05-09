"""FailureInfo + structured failure envelope (v2.11)."""
from __future__ import annotations

import json
from pathlib import Path

import pcq
from pcq.agent.run_record import (
    ERROR_CODES,
    FailureInfo,
    category_to_error_code,
)
from pcq.contract import _classify_exception, _normalize_failure


def _setup_cfg(out_dir: Path, **extra) -> Path:
    cfg = {"output_dir": str(out_dir), "seed": 42}
    cfg.update(extra)
    p = out_dir.parent / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_failure_info_roundtrip():
    """to_dict ↔ from_dict 보존."""
    fi = FailureInfo(
        error_code="ERR_OUT_OF_MEMORY",
        category="oom",
        message="CUDA OOM at batch 17",
        evidence={"batch_size": 32, "free_gb": 0.2},
        suggested_fix="reduce batch size to 16",
    )
    d = fi.to_dict()
    fi2 = FailureInfo.from_dict(d)
    assert fi2.error_code == "ERR_OUT_OF_MEMORY"
    assert fi2.category == "oom"
    assert fi2.evidence == {"batch_size": 32, "free_gb": 0.2}
    assert fi2.suggested_fix == "reduce batch size to 16"


def test_failure_to_dict_omits_empty():
    """빈 값은 직렬화에서 제외 — backward compat."""
    fi = FailureInfo(error_code="ERR_RUNTIME", message="boom")
    d = fi.to_dict()
    assert "evidence" not in d
    assert "suggested_fix" not in d
    assert "category" not in d


def test_failure_backward_compat_category_only():
    """old shape (category 만) → error_code derive."""
    old = {"category": "oom", "message": "CUDA OOM"}
    fi = FailureInfo.from_dict(old)
    assert fi.error_code == "ERR_OUT_OF_MEMORY"
    assert fi.category == "oom"


def test_failure_backward_compat_unknown_category():
    """알 수 없는 category → ERR_RUNTIME (catch-all)."""
    fi = FailureInfo.from_dict({"category": "weird_thing", "message": "x"})
    assert fi.error_code == "ERR_RUNTIME"


def test_failure_explicit_error_code_preserved():
    """이미 error_code 있으면 derive 하지 않음."""
    fi = FailureInfo.from_dict({
        "category": "oom",
        "error_code": "ERR_INVALID_CONFIG",   # 의도적 mismatch — 명시값 보존.
        "message": "test",
    })
    assert fi.error_code == "ERR_INVALID_CONFIG"


def test_category_to_error_code_mapping():
    """category → error_code 매핑이 ERROR_CODES 안의 값만 발급."""
    for cat in (
        "missing_dependency", "config_error", "dataset_missing",
        "oom", "timeout", "unknown_exception",
    ):
        code = category_to_error_code(cat)
        assert code in ERROR_CODES


def test_error_codes_set_immutable():
    """ERROR_CODES 가 frozenset (의도적 enum)."""
    assert isinstance(ERROR_CODES, frozenset)
    assert "ERR_RUNTIME" in ERROR_CODES
    assert "ERR_MISSING_DEPENDENCY" in ERROR_CODES


def test_classify_exception_import_error():
    """ImportError → ERR_MISSING_DEPENDENCY + evidence.module."""
    exc = ImportError("No module named 'torchvision'")
    exc.name = "torchvision"
    code, category, evidence = _classify_exception(exc)
    assert code == "ERR_MISSING_DEPENDENCY"
    assert category == "missing_dependency"
    assert evidence["module"] == "torchvision"


def test_classify_exception_memory_error():
    """MemoryError → ERR_OUT_OF_MEMORY."""
    code, category, evidence = _classify_exception(MemoryError())
    assert code == "ERR_OUT_OF_MEMORY"
    assert category == "oom"
    assert evidence["exception_type"] == "MemoryError"


def test_classify_exception_timeout_error():
    """TimeoutError → ERR_TIMEOUT."""
    code, category, _ = _classify_exception(TimeoutError("deadline"))
    assert code == "ERR_TIMEOUT"
    assert category == "timeout"


def test_classify_exception_file_not_found():
    """FileNotFoundError → ERR_DATASET_UNAVAILABLE + path evidence."""
    exc = FileNotFoundError(2, "No such file", "/data/missing.csv")
    code, category, evidence = _classify_exception(exc)
    assert code == "ERR_DATASET_UNAVAILABLE"
    assert category == "dataset_missing"
    assert evidence["path"] == "/data/missing.csv"


def test_classify_exception_runtime_fallback():
    """모르는 exception → ERR_RUNTIME catch-all."""
    code, category, evidence = _classify_exception(ValueError("oops"))
    assert code == "ERR_RUNTIME"
    assert category == "unknown_exception"
    assert evidence["exception_type"] == "ValueError"


def test_normalize_failure_derives_error_code_from_category():
    """category 만 있고 error_code 없으면 derive + evidence={} 보장."""
    out = _normalize_failure({"category": "oom", "message": "boom"})
    assert out["error_code"] == "ERR_OUT_OF_MEMORY"
    assert out["category"] == "oom"
    assert out["evidence"] == {}


def test_normalize_failure_explicit_overrides():
    """명시 error_code + evidence 는 그대로 보존."""
    out = _normalize_failure({
        "category": "oom",
        "error_code": "ERR_OUT_OF_MEMORY",
        "message": "CUDA OOM",
        "evidence": {"batch_size": 64},
        "suggested_fix": "reduce bs",
    })
    assert out["error_code"] == "ERR_OUT_OF_MEMORY"
    assert out["evidence"] == {"batch_size": 64}
    assert out["suggested_fix"] == "reduce bs"


def test_normalize_failure_none():
    """None 입력 → None 출력 (pass-through)."""
    assert _normalize_failure(None) is None
    assert _normalize_failure({}) == {}


def test_save_run_summary_failure_evidence_persists(tmp_path, monkeypatch):
    """save_run_summary 가 failure dict 의 error_code/evidence 를 보존."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_run_summary(
        history=[{"epoch": 0, "loss": float("nan")}],
        status="failed",
        failure={
            "category": "nan_loss",
            "error_code": "ERR_RUNTIME",
            "message": "loss became NaN",
            "evidence": {"epoch": 0, "step": 142},
            "suggested_fix": "check learning rate",
        },
    )
    rs = json.loads((out_dir / "run_summary.json").read_text())
    f = rs["failure"]
    assert f["error_code"] == "ERR_RUNTIME"
    assert f["evidence"] == {"epoch": 0, "step": 142}
    assert f["category"] == "nan_loss"
    assert f["suggested_fix"] == "check learning rate"


def test_save_run_summary_failure_derives_error_code(tmp_path, monkeypatch):
    """category 만 있으면 save_run_summary 가 error_code 자동 derive."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("CQ_CONFIG_JSON", str(_setup_cfg(out_dir)))

    pcq.save_run_summary(
        history=[],
        status="failed",
        failure={"message": "ImportError: No module"},
    )
    rs = json.loads((out_dir / "run_summary.json").read_text())
    f = rs["failure"]
    # classify_failure → category=missing_dependency → derive ERR_MISSING_DEPENDENCY.
    assert f["category"] == "missing_dependency"
    assert f["error_code"] == "ERR_MISSING_DEPENDENCY"


def test_failure_info_loads_old_run_summary(tmp_path):
    """v2.10 이전 RunRecord (category-only) 도 정상 read."""
    old_failure = {
        "category": "config_error",
        "message": "CQ_CONFIG_JSON missing",
        "suggested_fix": "set the env var",
    }
    fi = FailureInfo.from_dict(old_failure)
    assert fi.error_code == "ERR_INVALID_CONFIG"
    assert fi.evidence == {}
    assert fi.suggested_fix == "set the env var"
