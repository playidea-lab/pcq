"""AtomSpec, ParamSpec, AtomRef 직렬화/검증 테스트."""
from __future__ import annotations

import json

from pcq.registry.spec import AtomRef, AtomSpec, ParamSpec


def test_param_spec_validate_ok():
    p = ParamSpec(type="int", default=10, min=0, max=100)
    ok, _ = p.validate(50)
    assert ok


def test_param_spec_validate_type_mismatch():
    p = ParamSpec(type="int")
    ok, msg = p.validate("hello")
    assert not ok and "type mismatch" in msg


def test_param_spec_validate_choices():
    p = ParamSpec(type="str", choices=["train", "val"])
    ok, _ = p.validate("train")
    assert ok
    ok, msg = p.validate("test")
    assert not ok and "choices" in msg


def test_param_spec_validate_range_max():
    p = ParamSpec(type="float", min=0.0, max=1.0)
    ok, msg = p.validate(2.0)
    assert not ok and "max" in msg


def test_param_spec_validate_range_min():
    p = ParamSpec(type="int", min=1)
    ok, msg = p.validate(0)
    assert not ok and "min" in msg


def test_param_spec_required_missing_returns_error():
    p = ParamSpec(type="int", required=True)
    ok, msg = p.validate(None)
    assert not ok and "required" in msg


def test_param_spec_optional_none_ok():
    p = ParamSpec(type="int", required=False)
    ok, _ = p.validate(None)
    assert ok


def test_param_spec_to_dict_minimal():
    p = ParamSpec(type="int", default=1)
    d = p.to_dict()
    assert d == {"type": "int", "default": 1}


def test_param_spec_to_dict_full():
    p = ParamSpec(
        type="int", default=5, required=True, min=0, max=100,
        description="num samples",
    )
    d = p.to_dict()
    assert d["type"] == "int"
    assert d["min"] == 0
    assert d["max"] == 100
    assert d["required"] is True
    assert d["description"] == "num samples"


def test_atom_spec_from_meta_inferred():
    spec = AtomSpec.from_meta("model", "x", lambda: None, None)
    assert spec.metadata_status == "inferred"
    assert spec.params == {}


def test_atom_spec_from_meta_explicit():
    spec = AtomSpec.from_meta(
        "loss", "ce", lambda: None,
        {
            "tasks": ["classification"],
            "params": {"ignore_index": {"type": "int", "default": -100}},
        },
    )
    assert spec.metadata_status == "explicit"
    assert "ignore_index" in spec.params
    assert spec.params["ignore_index"].default == -100


def test_atom_spec_to_dict_excludes_factory():
    spec = AtomSpec.from_meta(
        "loss", "ce", lambda: None, {"tasks": ["a"]}
    )
    d = spec.to_dict()
    assert "factory" not in d
    assert d["tasks"] == ["a"]
    json.dumps(d)  # JSON-safe


def test_atom_spec_validate_params_required_missing():
    spec = AtomSpec.from_meta(
        "opt", "x", lambda: None,
        {"params": {"lr": {"type": "float", "required": True}}},
    )
    errors = spec.validate_params({})
    assert any("required" in e for e in errors)


def test_atom_spec_validate_params_unknown():
    spec = AtomSpec.from_meta(
        "opt", "x", lambda: None,
        {"params": {"lr": {"type": "float"}}},
    )
    errors = spec.validate_params({"lr": 1e-3, "unknown_param": 0})
    assert any("unknown" in e for e in errors)


def test_atom_spec_validate_params_clean_pass():
    spec = AtomSpec.from_meta(
        "loss", "ce", lambda: None,
        {"params": {"ignore_index": {"type": "int", "default": -100}}},
    )
    errors = spec.validate_params({"ignore_index": -1})
    assert errors == []


def test_atom_ref_to_from_dict_round_trip():
    ref = AtomRef(
        kind="loss", name="cross_entropy", params={"ignore_index": -1}
    )
    d = ref.to_dict()
    json.dumps(d)
    ref2 = AtomRef.from_dict(d)
    assert ref2 == ref


def test_atom_ref_default_params_empty():
    ref = AtomRef.from_dict({"kind": "model", "name": "unet"})
    assert ref.params == {}
