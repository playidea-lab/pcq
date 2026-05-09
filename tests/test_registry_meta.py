"""Metadata-aware registry 동작 테스트 — register/build/build_ref/validate_ref."""
from __future__ import annotations

import pytest

import pcq
from pcq import registry


def test_register_with_meta_explicit():
    pcq.register_loss(
        "test_loss_meta",
        factory=lambda: None,
        meta={
            "tasks": ["classification"],
            "params": {"x": {"type": "int", "default": 1}},
        },
    )
    spec = registry.losses.get("test_loss_meta")
    assert spec.metadata_status == "explicit"
    assert spec.tasks == ["classification"]


def test_register_old_style_inferred():
    # 기존 형태 (meta 미지정) → metadata_status="inferred"
    pcq.register_loss("test_loss_old", lambda: None)
    spec = registry.losses.get("test_loss_old")
    assert spec.metadata_status == "inferred"


def test_build_ref_resolves_with_params():
    pcq.register_loss(
        "test_ce_resolve",
        factory=lambda ignore_index=-100: ignore_index,
        meta={"params": {"ignore_index": {"type": "int", "default": -100}}},
    )
    ref = pcq.loss_ref("test_ce_resolve", {"ignore_index": -1})
    out = registry.losses.build_ref(ref)
    assert out == -1


def test_validate_ref_unknown_param():
    pcq.register_loss(
        "test_ce_validate",
        factory=lambda **k: None,
        meta={"params": {"a": {"type": "int", "default": 0}}},
    )
    ref = pcq.loss_ref("test_ce_validate", {"unknown_param": 1})
    errors = registry.losses.validate_ref(ref)
    assert any("unknown" in e for e in errors)


def test_validate_ref_kind_mismatch():
    ref = pcq.model_ref("anything")
    errors = registry.losses.validate_ref(ref)
    assert any("kind mismatch" in e for e in errors)


def test_validate_ref_unknown_atom():
    ref = pcq.loss_ref("does_not_exist_xyz")
    errors = registry.losses.validate_ref(ref)
    assert any("unknown loss" in e for e in errors)


def test_validate_ref_inferred_skips_strict():
    """inferred metadata 는 legacy 호환 — strict param check 안 함."""
    pcq.register_loss("test_inferred_legacy", lambda: None)
    ref = pcq.loss_ref("test_inferred_legacy", {"any_param": 0})
    errors = registry.losses.validate_ref(ref)
    assert errors == []


def test_register_decorator_form_with_meta():
    @pcq.register_metric("custom_zero_v18", meta={"tasks": ["any"]})
    def factory():
        import torch

        return lambda logits, labels: torch.tensor(0.0)

    spec = registry.metrics.get("custom_zero_v18")
    assert spec.tasks == ["any"]
    assert spec.metadata_status == "explicit"


def test_build_with_param_validation_rejects():
    pcq.register_loss(
        "test_strict_param",
        factory=lambda lr=0.1: lr,
        meta={"params": {"lr": {"type": "float", "min": 0.0, "max": 1.0}}},
    )
    with pytest.raises(ValueError, match="param errors"):
        registry.losses.build("test_strict_param", lr=2.0)


def test_meta_returns_json_safe_dict():
    import json

    spec_dict = registry.losses.meta("cross_entropy")
    json.dumps(spec_dict)
    assert spec_dict["name"] == "cross_entropy"
    assert spec_dict["metadata_status"] == "explicit"


def test_5_explicit_atoms_have_full_metadata():
    """v1.8 에서 5 atom 에 명시적 metadata 부여."""
    cases = [
        (registry.losses, "cross_entropy"),
        (registry.models, "unet"),
        (registry.datasets, "fake_seg"),
        (registry.datasets, "voc_seg"),
        (registry.metrics, "iou"),
    ]
    for reg, name in cases:
        spec = reg.get(name)
        assert spec.metadata_status == "explicit", (
            f"{reg.kind}/{name} should have explicit metadata"
        )
