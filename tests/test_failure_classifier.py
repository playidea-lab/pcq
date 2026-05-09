"""failure_classifier — 자동 카테고리 추론 (v1.17)."""
import json

import pcq
from pcq.agent.failure_classifier import classify_failure, enrich_failure


def test_classify_oom_message():
    assert classify_failure("CUDA out of memory") == "oom"
    assert classify_failure("RuntimeError: cuda out of memory") == "oom"


def test_classify_nan_loss():
    assert classify_failure("loss is NaN at epoch 5") == "nan_loss"
    assert classify_failure("training diverged: nan_loss observed") == "nan_loss"


def test_classify_missing_dependency():
    assert classify_failure("ModuleNotFoundError: No module named 'timm'") == "missing_dependency"
    assert classify_failure("ImportError: No module named foo") == "missing_dependency"


def test_classify_dataset_missing():
    assert classify_failure("FileNotFoundError: data/train.csv") == "dataset_missing"


def test_classify_dataset_shape():
    assert classify_failure("size mismatch for layer.weight") == "dataset_shape"
    assert classify_failure("shape mismatch: expected (3,224) got (1,224)") == "dataset_shape"


def test_classify_unknown_falls_back():
    assert classify_failure("some unrelated error") == "unknown_exception"
    assert classify_failure("") == "unknown_exception"


def test_enrich_failure_fills_missing_category():
    out = enrich_failure({"message": "CUDA out of memory"})
    assert out is not None
    assert out["category"] == "oom"


def test_enrich_failure_keeps_explicit_category():
    """이미 명시된 카테고리(unknown 아님)는 보존."""
    given = {"category": "label_contract", "message": "CUDA out of memory"}
    out = enrich_failure(given)
    assert out is not None
    assert out["category"] == "label_contract"


def test_enrich_failure_overrides_unknown_exception():
    given = {"category": "unknown_exception", "message": "ModuleNotFoundError: timm"}
    out = enrich_failure(given)
    assert out is not None
    assert out["category"] == "missing_dependency"


def test_enrich_failure_handles_none():
    assert enrich_failure(None) is None


def test_save_run_summary_auto_classifies_failure(tmp_path, monkeypatch):
    """save_run_summary 가 failure category 를 자동으로 enrich 한다."""
    cfg = {"output_dir": str(tmp_path), "seed": 42}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))

    failure = {"message": "ModuleNotFoundError: No module named 'timm'"}
    pcq.save_run_summary(history=[], status="failed", failure=failure)
    rs = json.loads((tmp_path / "run_summary.json").read_text())
    assert rs["failure"]["category"] == "missing_dependency"
