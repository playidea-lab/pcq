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


# ── 4 신규 카테고리 (T-FC-1) ──────────────────────────────────────────


def test_classify_accuracy_below_threshold():
    """accuracy_below_threshold: 검증 정확도가 목표치 미달인 경우."""
    # Arrange
    msg = "ValueError: Validation accuracy 0.65 below target 0.80"
    # Act
    result = classify_failure(msg)
    # Assert
    assert result == "accuracy_below_threshold"


def test_classify_user_interrupted():
    """user_interrupted: KeyboardInterrupt / SIGINT 로 중단된 경우."""
    # Arrange
    msg = "KeyboardInterrupt\n  File train.py, line 42\n  ..."
    # Act
    result = classify_failure(msg)
    # Assert
    assert result == "user_interrupted"


def test_classify_disk_full():
    """disk_full: 디스크 공간 부족으로 쓰기 실패한 경우."""
    # Arrange
    msg = "OSError: [Errno 28] No space left on device"
    # Act
    result = classify_failure(msg)
    # Assert
    assert result == "disk_full"


def test_classify_model_load_failed():
    """model_load_failed: safetensors/체크포인트 파일 손상으로 로드 실패."""
    # Arrange
    msg = "RuntimeError: safetensors corrupt header at offset 0"
    # Act
    result = classify_failure(msg)
    # Assert
    assert result == "model_load_failed"


def test_no_false_positive_below_average():
    """accuracy_below_threshold: 'below average' 같은 일반 표현은 매칭하지 않아야 한다."""
    # Arrange
    msg = "Training below average accuracy on validation"
    # Act
    result = classify_failure(msg)
    # Assert — 'target/threshold' 없으므로 unknown_exception 이어야 함
    assert result == "unknown_exception", f"got {result}, should not match accuracy_below_threshold"


# ── 기존 카테고리 sanity check ─────────────────────────────────────────


def test_sanity_oom_still_works():
    """기존 oom 패턴이 신규 패턴 추가 후에도 그대로 동작하는지 확인."""
    assert classify_failure("RuntimeError: CUDA out of memory") == "oom"


def test_sanity_nan_loss_still_works():
    """기존 nan_loss 패턴이 신규 패턴 추가 후에도 그대로 동작하는지 확인."""
    assert classify_failure("loss is NaN at epoch 10") == "nan_loss"


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
