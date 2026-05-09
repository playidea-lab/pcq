"""tests/test_core.py — pcq.core 5개 공개 함수의 컨트랙트 테스트."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import pcq
from pcq import core


@pytest.fixture(autouse=True)
def reset_module_state():
    """테스트 간 모듈 레벨 상태 (미선언 메트릭 + declared cache) 초기화."""
    core._undeclared_warned.clear()
    core._undeclared_count.clear()
    core._reset_declared_cache()
    yield
    core._undeclared_warned.clear()
    core._undeclared_count.clear()
    core._reset_declared_cache()


@pytest.fixture
def cfg_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """임시 JSON config 파일을 만들고 CQ_CONFIG_JSON으로 가리킨다."""
    cfg = {
        "epochs": 3,
        "batch_size": 32,
        "lr": 0.001,
        "output_dir": str(tmp_path / "out"),
        "inputs": {"train": str(tmp_path / "data" / "train")},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("CQ_CONFIG_JSON", str(p))
    return p


# ---------------------------------------------------------------- config() ---


def test_config_reads_env_var(cfg_file: Path):
    cfg = pcq.config()
    assert cfg["epochs"] == 3
    assert cfg["batch_size"] == 32
    assert cfg["lr"] == 0.001


def test_config_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """CQ_CONFIG_JSON 도 cq.yaml 도 없으면 RuntimeError.

    v2.12: cq.yaml fallback 추가됨. 둘 다 없을 때 메시지가
    'cq.yaml' / 'CQ_CONFIG_JSON' 중 하나를 언급해야 한다.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    with pytest.raises(RuntimeError) as ei:
        pcq.config()
    msg = str(ei.value)
    assert "cq.yaml" in msg or "CQ_CONFIG_JSON" in msg


# ------------------------------------------------------------------- log() ---


def test_log_emits_finite_numerics(capsys: pytest.CaptureFixture[str]):
    pcq.log(epoch=1, loss=0.42)
    out = capsys.readouterr().out
    assert "@epoch=1" in out
    assert "@loss=0.42" in out


def test_log_skips_bool_string_nan_inf(capsys: pytest.CaptureFixture[str]):
    pcq.log(
        flag=True,
        name="hi",
        x=float("nan"),
        y=float("inf"),
        z=float("-inf"),
    )
    out = capsys.readouterr().out
    assert out == ""  # 어떤 메트릭도 출력되지 않아야 한다


def test_log_emits_numpy_scalars(capsys: pytest.CaptureFixture[str]):
    pcq.log(loss=np.float32(0.5), step=np.int64(7))
    out = capsys.readouterr().out
    assert "@loss=0.5" in out
    assert "@step=7" in out


def test_log_warns_once_on_undeclared(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("CQ_DECLARED_METRICS", "epoch,loss")
    pcq.log(weird=1)
    pcq.log(weird=2)  # 두 번째 호출
    captured = capsys.readouterr()
    # 메트릭 출력은 모두 차단됨
    assert "@weird" not in captured.out
    # stderr에는 정확히 1번 경고만
    warning_lines = [
        line for line in captured.err.splitlines() if "undeclared metric key" in line
    ]
    assert len(warning_lines) == 1
    assert "'weird'" in warning_lines[0]


def test_log_strict_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CQ_DECLARED_METRICS", "epoch")
    with pytest.raises(RuntimeError, match="undeclared metric key"):
        pcq.log(strict=True, weird=1)


def test_log_allows_declared_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("CQ_DECLARED_METRICS", "epoch,loss")
    pcq.log(epoch=2, loss=0.1)
    out = capsys.readouterr().out
    assert "@epoch=2" in out
    assert "@loss=0.1" in out


def test_log_auto_reads_declared_from_cq_config_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """CQ_DECLARED_METRICS env 없을 때 CQ_CONFIG_JSON 의 _metrics_declared 자동 로드."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps({"_metrics_declared": ["epoch", "train_loss"]})
    )
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    monkeypatch.delenv("CQ_DECLARED_METRICS", raising=False)

    pcq.log(epoch=1)  # 선언됨 → 정상 출력
    pcq.log(weird_key=2)  # 미선언 → stderr 경고
    captured = capsys.readouterr()
    assert "@epoch=1" in captured.out
    assert "@weird_key" not in captured.out
    assert "weird_key" in captured.err
    assert "undeclared metric key" in captured.err


def test_log_env_overrides_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """env CQ_DECLARED_METRICS 가 CQ_CONFIG_JSON cfg 보다 우선."""
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"_metrics_declared": ["a", "b"]}))
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    monkeypatch.setenv("CQ_DECLARED_METRICS", "x,y")

    declared = core._read_declared_metrics()
    assert declared == {"x", "y"}  # env 우선


def test_log_no_declaration_warns_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """env 도 cfg 도 없으면 검증 skip — 모든 key 경고 없이 통과."""
    monkeypatch.delenv("CQ_DECLARED_METRICS", raising=False)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)

    pcq.log(weird=1, anything=2)
    captured = capsys.readouterr()
    assert "@weird=1" in captured.out
    assert "@anything=2" in captured.out
    assert "weird" not in captured.err
    assert "anything" not in captured.err


def test_log_handles_missing_cq_config_json_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """CQ_CONFIG_JSON 이 가리키는 파일이 없거나 _metrics_declared 키가 없으면 skip."""
    # 파일이 존재하지 않는 경로
    monkeypatch.setenv("CQ_CONFIG_JSON", str(tmp_path / "missing.json"))
    monkeypatch.delenv("CQ_DECLARED_METRICS", raising=False)
    pcq.log(any_key=1)
    captured = capsys.readouterr()
    assert "@any_key=1" in captured.out
    assert "any_key" not in captured.err


# ------------------------------------------------------------ output_dir() ---


def test_output_dir_creates_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    out = pcq.output_dir()
    assert out == Path("output")
    assert (tmp_path / "output").is_dir()


def test_output_dir_uses_config(cfg_file: Path):
    out = pcq.output_dir()
    cfg = json.loads(cfg_file.read_text())
    assert out == Path(cfg["output_dir"])
    assert out.is_dir()


# ------------------------------------------------------------- input_dir() ---


def test_input_dir_reads_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "from_env"
    monkeypatch.setenv("CQ_INPUT_DIR_TRAIN", str(target))
    assert pcq.input_dir("train") == target


def test_input_dir_falls_back_to_cfg(cfg_file: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CQ_INPUT_DIR_TRAIN", raising=False)
    cfg = json.loads(cfg_file.read_text())
    assert pcq.input_dir("train") == Path(cfg["inputs"]["train"])


def test_input_dir_raises_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CQ_CONFIG_JSON", raising=False)
    monkeypatch.delenv("CQ_INPUT_DIR_TRAIN", raising=False)
    with pytest.raises(FileNotFoundError, match="train"):
        pcq.input_dir("train")


def test_input_dir_env_takes_precedence(
    cfg_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "env_wins"
    monkeypatch.setenv("CQ_INPUT_DIR_TRAIN", str(target))
    assert pcq.input_dir("train") == target  # cfg가 아닌 env 값


# -------------------------------------------------------- seed_everything() --


def test_seed_everything_reproducibility():
    pcq.seed_everything(42)
    a = np.random.rand(5)
    pcq.seed_everything(42)
    b = np.random.rand(5)
    assert (a == b).all()


def test_seed_everything_python_random():
    import random as py_random

    pcq.seed_everything(123)
    a = [py_random.random() for _ in range(5)]
    pcq.seed_everything(123)
    b = [py_random.random() for _ in range(5)]
    assert a == b
