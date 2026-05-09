"""End-to-end 통합 테스트 — cq worker 의 train.py 호출을 시뮬레이션한다."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _write_config(tmp_path: Path, **overrides) -> Path:
    # 기본 config 위에 overrides 를 덮어쓴 임시 JSON 파일을 만든다.
    cfg = {
        "output_dir": str(tmp_path / "output"),
        "epochs": 1,
        "batch_size": 16,
        "lr": 0.001,
        "seed": 42,
        "_metrics_declared": [
            "epoch",
            "train_loss",
            "train_acc",
            "eval_loss",
            "eval_acc",
        ],
    }
    cfg.update(overrides)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def test_full_run_from_cq_config_json(tmp_path: Path) -> None:
    """cq worker 시뮬레이션: CQ_CONFIG_JSON env 설정 후 train.py subprocess 실행.

    검증:
    - exit code 0
    - output_dir 에 5개 아티팩트 존재
    - stdout 에 @epoch=, @train_loss= 라인 포함
    """
    cfg_path = _write_config(tmp_path)
    output_dir = tmp_path / "output"

    env = dict(os.environ)
    env["CQ_CONFIG_JSON"] = str(cfg_path)

    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "train.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"train.py failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # contract artifacts (v4.0 — save_all 이 작성하는 6개 표준 산출물)
    for name in [
        "config.json", "metrics.json", "manifest.json",
        "run_summary.json", "run_record.json", "validation_report.json",
    ]:
        assert (output_dir / name).exists(), f"missing artifact: {name}"
    # metrics.json 구조 확인
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "history" in metrics
    assert len(metrics["history"]) >= 1
    # stdout 에 @epoch= 메트릭 라인 확인
    assert "@epoch=" in result.stdout, f"no @epoch= in stdout:\n{result.stdout}"
    assert (
        "@train_loss=" in result.stdout
    ), f"no @train_loss= in stdout:\n{result.stdout}"


def test_undeclared_metric_warning(tmp_path: Path) -> None:
    """선언되지 않은 메트릭 key 로 pcq.log() 호출 시 stderr 경고 출력 확인."""
    cfg_path = tmp_path / "config.json"
    cfg = {
        "output_dir": str(tmp_path / "output"),
        "_metrics_declared": ["epoch"],  # epoch 만 선언
    }
    cfg_path.write_text(json.dumps(cfg))

    # weird_key 는 선언되지 않은 key — 경고 발생해야 함
    script = (
        "import pcq\n"
        "pcq.config()\n"
        "pcq.log(weird_key=1.0)\n"
        "pcq.log(weird_key=2.0)\n"
    )
    env = dict(os.environ)
    env["CQ_CONFIG_JSON"] = str(cfg_path)
    env["CQ_DECLARED_METRICS"] = "epoch"  # core._read_declared_metrics() 가 env 우선 사용

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # stderr 에 "undeclared" 경고가 최소 1번 등장해야 한다
    # (첫 호출 1회 + atexit 요약 1회 = 최대 2번까지 가능)
    warning_lines = [
        line for line in result.stderr.splitlines() if "undeclared" in line.lower()
    ]
    assert (
        len(warning_lines) >= 1
    ), f"expected 'undeclared' in stderr, got:\n{result.stderr}"


def test_missing_cq_config_json_raises() -> None:
    """CQ_CONFIG_JSON 미설정 시 train.py 는 0이 아닌 exit code 로 종료해야 한다."""
    env = {k: v for k, v in os.environ.items() if k != "CQ_CONFIG_JSON"}
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "train.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=10,
    )
    assert result.returncode != 0
    assert (
        "CQ_CONFIG_JSON" in result.stderr or "CQ_CONFIG_JSON" in result.stdout
    )
