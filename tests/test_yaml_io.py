"""yaml_io round-trip 테스트 — v1.10 minimal YAML reader/writer + v1.11 ruamel."""
from __future__ import annotations

import pytest

from pcq.agent.yaml_io import _RUAMEL_YAML, read_yaml, write_yaml


def test_round_trip_minimal(tmp_path):
    data = {"name": "test", "cmd": "uv run python train.py"}
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["name"] == "test"
    assert loaded["cmd"] == "uv run python train.py"


def test_round_trip_with_configs(tmp_path):
    data = {
        "name": "smoke",
        "cmd": "uv run python train.py",
        "configs": {
            "preset": "vision/fake_smoke",
            "epochs": 2,
            "batch_size": 16,
            "lr": 0.001,
            "seed": 42,
        },
        "metrics": ["epoch", "train_loss"],
        "artifacts": ["output/"],
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["configs"]["epochs"] == 2
    # float 표현 확인 (lr=0.001 round-trip)
    assert abs(loaded["configs"]["lr"] - 0.001) < 1e-9
    assert loaded["metrics"] == ["epoch", "train_loss"]
    assert loaded["artifacts"] == ["output/"]


def test_round_trip_metrics_list_with_underscores(tmp_path):
    """_metrics_declared 같은 underscore prefix 키 → list 보존."""
    data = {
        "name": "n",
        "cmd": "c",
        "configs": {
            "preset": "x/y",
            "_metrics_declared": [
                "epoch", "train_loss", "train_acc", "eval_loss", "eval_acc",
            ],
        },
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    md = loaded["configs"]["_metrics_declared"]
    assert md == ["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"]


def test_round_trip_nested_overrides_data_atomref_dict(tmp_path):
    """_overrides_data.<atom> = {kind, name, params: {...}} 의 nested dict round-trip."""
    data = {
        "name": "n",
        "cmd": "c",
        "configs": {
            "preset": "vision/fake_smoke",
            "epochs": 5,
            "_overrides_data": {
                "loss": {
                    "kind": "loss",
                    "name": "cross_entropy",
                    "params": {"ignore_index": -1},
                },
            },
        },
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    od = loaded["configs"].get("_overrides_data", {})
    assert "loss" in od
    loss_ref = od["loss"]
    # 최소: kind/name 보존
    assert loss_ref["kind"] == "loss"
    assert loss_ref["name"] == "cross_entropy"
    # params 보존 (3-level deep)
    assert loss_ref["params"]["ignore_index"] == -1


def test_quoted_strings_for_special_chars(tmp_path):
    """공백 포함 문자열도 round-trip 보장 (writer 별 표현 차이 허용).

    minimal writer 는 JSON-quote, ruamel 은 plain — 둘 다 read 가 똑같이
    복원해야 한다.
    """
    data = {"name": "with-dash_test", "cmd": "uv run python -m pcq.cli"}
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["cmd"] == "uv run python -m pcq.cli"
    assert loaded["name"] == "with-dash_test"


def test_minimal_writer_quotes_strings_with_spaces(tmp_path, monkeypatch):
    """minimal writer 는 공백 포함 문자열을 JSON-quote 한다 (writer-specific)."""
    import pcq.agent.yaml_io as mod

    monkeypatch.setattr(mod, "_RUAMEL_YAML", None)
    data = {"name": "n", "cmd": "uv run python -m pcq.cli"}
    p = tmp_path / "cq.yaml"
    mod.write_yaml(data, p)
    text = p.read_text()
    assert '"uv run python -m pcq.cli"' in text


def test_minimal_writer_top_level_key_order(tmp_path, monkeypatch):
    """minimal writer 는 cq.yaml convention 순서로 재배치 (writer-specific).

    ruamel 은 입력 dict 의 키 순서를 보존 — 사용자가 손으로 쓴 순서 우선.
    """
    import pcq.agent.yaml_io as mod

    monkeypatch.setattr(mod, "_RUAMEL_YAML", None)
    data = {
        "artifacts": ["output/"],
        "metrics": ["loss"],
        "configs": {"epochs": 1},
        "cmd": "x",
        "name": "n",
    }
    p = tmp_path / "cq.yaml"
    mod.write_yaml(data, p)
    text = p.read_text()
    # name 이 cmd 보다 먼저, configs 가 metrics 보다 먼저
    name_idx = text.find("name:")
    cmd_idx = text.find("cmd:")
    configs_idx = text.find("configs:")
    metrics_idx = text.find("metrics:")
    artifacts_idx = text.find("artifacts:")
    assert name_idx < cmd_idx < configs_idx < metrics_idx < artifacts_idx


def test_empty_dict_and_list_emit(tmp_path):
    data = {"name": "n", "configs": {}, "metrics": []}
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    # 빈 dict/list 가 None 으로 깨지지 않아야
    assert loaded.get("configs", {}) == {} or loaded.get("configs") is None
    assert loaded.get("metrics", []) == [] or loaded.get("metrics") is None


def test_negative_int_values_round_trip(tmp_path):
    """ignore_index=-1 같은 음수 round-trip."""
    data = {
        "name": "n",
        "cmd": "c",
        "configs": {"epochs": 3, "ignore_index": -1, "neg_float": -0.5},
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["configs"]["ignore_index"] == -1
    assert abs(loaded["configs"]["neg_float"] - (-0.5)) < 1e-9


def test_bool_values_round_trip(tmp_path):
    data = {"name": "n", "configs": {"smoke": True, "amp": False}}
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["configs"]["smoke"] is True
    assert loaded["configs"]["amp"] is False


# ─────────────────────────────────────────────────────────────────────
# v1.11: ruamel.yaml optional — comment preservation
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(_RUAMEL_YAML is None, reason="ruamel.yaml not installed")
def test_ruamel_preserves_comments_round_trip(tmp_path):
    """ruamel 설치 시 주석이 read → modify → write 순환에서 보존된다."""
    yaml_text = (
        "# Top-level comment\n"
        "name: test\n"
        "configs:\n"
        "  # config comment\n"
        "  epochs: 5\n"
    )
    p = tmp_path / "cq.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    data = read_yaml(p)
    data["configs"]["epochs"] = 10
    write_yaml(data, p)
    text2 = p.read_text(encoding="utf-8")
    assert "Top-level comment" in text2
    assert "config comment" in text2
    assert "epochs: 10" in text2


@pytest.mark.skipif(_RUAMEL_YAML is None, reason="ruamel.yaml not installed")
def test_ruamel_round_trip_dict_compatible(tmp_path):
    """ruamel CommentedMap 도 isinstance(x, dict) 통과한다."""
    data = {
        "name": "n",
        "cmd": "c",
        "configs": {"epochs": 3, "lr": 0.001},
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert isinstance(loaded, dict)
    assert isinstance(loaded["configs"], dict)
    assert loaded["configs"]["epochs"] == 3


def test_minimal_writer_works_when_ruamel_absent(tmp_path, monkeypatch):
    """ruamel 미사용 강제 → minimal writer/reader fallback."""
    import pcq.agent.yaml_io as mod

    monkeypatch.setattr(mod, "_RUAMEL_YAML", None)
    data = {"name": "t", "cmd": "c", "configs": {"epochs": 3}}
    p = tmp_path / "cq.yaml"
    mod.write_yaml(data, p)
    loaded = mod.read_yaml(p)
    assert loaded["configs"]["epochs"] == 3


def test_minimal_writer_overrides_data_round_trip_when_ruamel_absent(
    tmp_path, monkeypatch,
):
    """v1.10 fallback 의 _overrides_data round-trip 보장."""
    import pcq.agent.yaml_io as mod

    monkeypatch.setattr(mod, "_RUAMEL_YAML", None)
    data = {
        "name": "n",
        "cmd": "c",
        "configs": {
            "preset": "vision/fake_smoke",
            "_overrides_data": {
                "loss": {
                    "kind": "loss",
                    "name": "cross_entropy",
                    "params": {"ignore_index": -1},
                },
            },
        },
    }
    p = tmp_path / "cq.yaml"
    mod.write_yaml(data, p)
    loaded = mod.read_yaml(p)
    od = loaded["configs"]["_overrides_data"]
    assert od["loss"]["name"] == "cross_entropy"
    assert od["loss"]["params"]["ignore_index"] == -1


# ─────────────────────────────────────────────────────────────────────
# v1.15: structured cq.yaml — dict-style metrics + inputs section
# ─────────────────────────────────────────────────────────────────────


def test_round_trip_dict_metrics(tmp_path):
    data = {
        "name": "t", "cmd": "c",
        "metrics": {
            "eval_iou": {"mode": "max", "split": "val"},
            "eval_loss": {"mode": "min"},
        },
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["metrics"]["eval_iou"]["mode"] == "max"
    assert loaded["metrics"]["eval_iou"]["split"] == "val"
    assert loaded["metrics"]["eval_loss"]["mode"] == "min"


def test_round_trip_inputs_section(tmp_path):
    data = {
        "name": "t", "cmd": "c",
        "inputs": {
            "dataset": {
                "name": "dental",
                "version": "v12",
                "uri": "cq://datasets/dental/v12",
            },
        },
    }
    p = tmp_path / "cq.yaml"
    write_yaml(data, p)
    loaded = read_yaml(p)
    assert loaded["inputs"]["dataset"]["name"] == "dental"
    assert loaded["inputs"]["dataset"]["uri"] == "cq://datasets/dental/v12"


def test_round_trip_dict_metrics_minimal_writer(tmp_path, monkeypatch):
    """minimal writer (ruamel 없음) 에서도 dict-style metrics round-trip."""
    import pcq.agent.yaml_io as mod

    monkeypatch.setattr(mod, "_RUAMEL_YAML", None)
    data = {
        "name": "t", "cmd": "c",
        "metrics": {
            "eval_iou": {"mode": "max", "split": "val", "aggregation": "macro"},
        },
    }
    p = tmp_path / "cq.yaml"
    mod.write_yaml(data, p)
    loaded = mod.read_yaml(p)
    assert loaded["metrics"]["eval_iou"]["mode"] == "max"
    assert loaded["metrics"]["eval_iou"]["aggregation"] == "macro"


def test_round_trip_inputs_minimal_writer(tmp_path, monkeypatch):
    """minimal writer 에서도 inputs 3-level dict round-trip."""
    import pcq.agent.yaml_io as mod

    monkeypatch.setattr(mod, "_RUAMEL_YAML", None)
    data = {
        "name": "t", "cmd": "c",
        "inputs": {
            "dataset": {
                "name": "dental",
                "version": "v12",
                "uri": "cq://datasets/dental/v12",
            },
        },
    }
    p = tmp_path / "cq.yaml"
    mod.write_yaml(data, p)
    loaded = mod.read_yaml(p)
    assert loaded["inputs"]["dataset"]["name"] == "dental"
    # cq URI 는 opaque — quoting 차이가 있어도 round-trip.
    assert loaded["inputs"]["dataset"]["uri"] == "cq://datasets/dental/v12"
