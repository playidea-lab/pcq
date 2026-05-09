"""ignore_index mismatch validation gate (Step 6 부분 구현)."""
from __future__ import annotations

import pcq
from pcq.agent.validate import _validate_label_contracts


def test_ignore_index_mismatch_detected():
    """voc_seg 가 -1 (void→-1 변환), loss 가 -100 (default) → mismatch."""
    recipe_dict = {
        "dataset_train": pcq.dataset_ref("voc_seg", {"image_size": 256}),
        "loss": pcq.loss_ref("cross_entropy", {"ignore_index": -100}),
    }
    check = _validate_label_contracts(recipe_dict)
    assert check.status == "fail"
    assert check.severity == "blocking"
    assert "ignore_index" in check.detail
    assert check.suggested_fix is not None
    assert "-1" in check.suggested_fix


def test_ignore_index_match_passes():
    """voc_seg 와 loss 둘 다 -1 → pass."""
    recipe_dict = {
        "dataset_train": pcq.dataset_ref("voc_seg", {"image_size": 256}),
        "loss": pcq.loss_ref("cross_entropy", {"ignore_index": -1}),
    }
    check = _validate_label_contracts(recipe_dict)
    assert check.status == "pass"


def test_ignore_index_no_dataset_label_contract_passes():
    """fake_seg 는 ignore_index label_contract 없음 → 검사 skip (pass)."""
    recipe_dict = {
        "dataset_train": pcq.dataset_ref("fake_seg", {"num_classes": 21}),
        "loss": pcq.loss_ref("cross_entropy", {"ignore_index": -1}),
    }
    check = _validate_label_contracts(recipe_dict)
    # fake_seg 에는 ignore_index 가 label_contract 에 없으므로 mismatch 검사 안 함
    assert check.status == "pass"


def test_ignore_index_legacy_dict_skipped():
    """AtomRef 가 아닌 callable 등 legacy 형태는 skip."""
    recipe_dict = {
        "dataset_train": lambda _split: None,  # callable, not AtomRef
        "loss": object(),  # not AtomRef
    }
    check = _validate_label_contracts(recipe_dict)
    # 둘 다 AtomRef 아님 → ignore_index 추출 불가 → pass (둘 다 None)
    assert check.status == "pass"


# ── v1.9: model-dataset channel 일치 gate ───────────────────────────────────
def test_model_dataset_channels_mismatch_detected():
    """small_cnn(in_channels=3) + fake(channels=1) → fail."""
    from pcq.agent.validate import _validate_model_dataset_channels

    recipe_dict = {
        "model": pcq.model_ref(
            "small_cnn", {"in_channels": 3, "num_classes": 10},
        ),
        "dataset_train": pcq.dataset_ref(
            "fake",
            {"num_samples": 8, "num_classes": 10, "channels": 1},
        ),
    }
    check = _validate_model_dataset_channels(recipe_dict)
    assert check is not None
    assert check.status == "fail"
    assert check.severity == "blocking"
    assert "channels" in check.detail.lower()


def test_model_dataset_channels_match():
    """small_cnn(in_channels=3) + fake(channels=3) → pass."""
    from pcq.agent.validate import _validate_model_dataset_channels

    recipe_dict = {
        "model": pcq.model_ref(
            "small_cnn", {"in_channels": 3, "num_classes": 10},
        ),
        "dataset_train": pcq.dataset_ref(
            "fake",
            {"num_samples": 8, "num_classes": 10, "channels": 3},
        ),
    }
    check = _validate_model_dataset_channels(recipe_dict)
    assert check is not None
    assert check.status == "pass"


def test_model_dataset_channels_legacy_skipped():
    """AtomRef 아닌 객체는 None 반환 (skip)."""
    from pcq.agent.validate import _validate_model_dataset_channels

    recipe_dict = {
        "model": object(),
        "dataset_train": lambda _split: None,
    }
    check = _validate_model_dataset_channels(recipe_dict)
    assert check is None


# ── v1.9: optional extras availability gate ────────────────────────────────
def test_optional_extras_no_extras_required_passes():
    """requires_extras 없으면 pass."""
    from pcq.agent.validate import _validate_optional_extras

    check = _validate_optional_extras({"requires_extras": []})
    assert check.status == "pass"


def test_optional_extras_unknown_module_warn():
    """알 수 없는 extra 가 import 안되면 warn."""
    from pcq.agent.validate import _validate_optional_extras

    check = _validate_optional_extras(
        {"requires_extras": ["__definitely_no_such_extra__"]}
    )
    assert check.status == "warn"
    assert "not installed" in check.detail


def test_optional_extras_torchvision_present():
    """torchvision 설치된 환경(테스트 dep) → pass."""
    from pcq.agent.validate import _validate_optional_extras

    check = _validate_optional_extras({"requires_extras": ["vision"]})
    # torchvision 이 설치돼 있다고 가정 — 안 돼있으면 warn 도 허용
    assert check.status in ("pass", "warn")


# ── v1.9: monitor_candidates declared gate ─────────────────────────────────
def test_monitor_candidates_all_declared():
    """SPEC.monitor_candidates 의 metric 이 모두 SPEC.metrics 에 있으면 pass."""
    from pcq.agent.schema import RecipeSpec
    from pcq.agent.validate import _validate_monitor_in_metrics

    spec = RecipeSpec(
        name="x",
        task="classification",
        metrics=["epoch", "eval_loss", "eval_acc"],
        monitor_candidates=[
            {"name": "eval_loss", "mode": "min"},
            {"name": "eval_acc", "mode": "max"},
        ],
    )
    check = _validate_monitor_in_metrics(spec, list(spec.metrics))
    assert check is not None
    assert check.status == "pass"


def test_monitor_candidates_missing_warn():
    """SPEC.monitor_candidates 의 metric 이 SPEC.metrics 에 없으면 warn."""
    from pcq.agent.schema import RecipeSpec
    from pcq.agent.validate import _validate_monitor_in_metrics

    spec = RecipeSpec(
        name="x",
        task="classification",
        metrics=["epoch", "eval_loss"],
        monitor_candidates=[{"name": "f1_score", "mode": "max"}],
    )
    check = _validate_monitor_in_metrics(spec, list(spec.metrics))
    assert check is not None
    assert check.status == "warn"
    assert "f1_score" in check.detail


def test_monitor_candidates_empty_returns_none():
    """monitor_candidates 비면 None (skip)."""
    from pcq.agent.schema import RecipeSpec
    from pcq.agent.validate import _validate_monitor_in_metrics

    spec = RecipeSpec(
        name="x", task="classification", metrics=["eval_loss"],
        monitor_candidates=[],
    )
    check = _validate_monitor_in_metrics(spec, list(spec.metrics))
    assert check is None
