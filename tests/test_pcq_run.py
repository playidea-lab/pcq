"""tests/test_pcq_run.py — Fix 2 (G0-2).

`pcq run` subcommand: cq.yaml.cmd 읽어 실행, configs 를 .pcq/runtime_cfg.json 에
dump 후 CQ_CONFIG_JSON 자동 wiring. fresh-user 의 first-class 진입점.

Surface:
  pcq run [--path PATH] [--config-only] [--json] [--jsonl]

기본 동작:
  - cq.yaml 읽음. cmd 없으면 reject.
  - configs 를 .pcq/runtime_cfg.json 에 dump (project_root 안).
  - subprocess.run(cmd, env={**os.environ, CQ_CONFIG_JSON=...})
  - exit code forward.
  - --config-only 면 cmd 실행 안 함, runtime_cfg_path 만 출력.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pcq.cli import main as cli_main


def _write_cq_yaml(
    project: Path, cmd: str, configs: dict | None = None, name: str = "demo"
) -> Path:
    """헬퍼 — project 에 cq.yaml 작성."""
    from pcq.agent.yaml_io import write_yaml

    data: dict = {"name": name, "cmd": cmd}
    if configs is not None:
        data["configs"] = configs
    p = project / "cq.yaml"
    write_yaml(data, p)
    return p


def test_pcq_run_json_is_pure_envelope_and_captures_stdout(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """--json stdout 은 순수 JSON 이고 child stdout 은 로그 파일로 분리된다."""
    project = tmp_path / "proj"
    project.mkdir()
    train_py = project / "train.py"
    train_py.write_text(
        "import sys\n"
        "print('hello-from-train')\n"
        "print('warn-from-train', file=sys.stderr)\n",
        encoding="utf-8",
    )
    cmd = f"{sys.executable} train.py"
    _write_cq_yaml(project, cmd, configs={"epochs": 3})

    rc = cli_main(["run", "--path", str(project), "--json"])
    captured = capfd.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["exit_code"] == 0
    assert payload["status"] == "completed"
    assert payload["stdout_tail"] == "hello-from-train\n"
    assert payload["stderr_tail"] == "warn-from-train\n"
    assert Path(payload["stdout_path"]).read_text(encoding="utf-8") == (
        "hello-from-train\n"
    )
    assert Path(payload["stderr_path"]).read_text(encoding="utf-8") == (
        "warn-from-train\n"
    )


def test_pcq_run_non_json_streams_child_stdout(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """human mode 는 child stdout 을 기존처럼 터미널로 흘린다."""
    project = tmp_path / "proj"
    project.mkdir()
    cmd = f'{sys.executable} -c "print(\'hello-human-run\')"'
    _write_cq_yaml(project, cmd, configs={"epochs": 3})

    rc = cli_main(["run", "--path", str(project)])
    captured = capfd.readouterr()
    assert rc == 0
    assert "hello-human-run" in captured.out
    assert "== Run ==" in captured.out


def test_pcq_run_sets_env_from_configs(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """runtime_cfg.json 작성 + CQ_CONFIG_JSON env 자동 주입."""
    project = tmp_path / "proj"
    project.mkdir()
    # train.py: CQ_CONFIG_JSON 읽고 epochs 값을 echo
    train_py = project / "train.py"
    train_py.write_text(
        "import os, json\n"
        "p = os.environ['CQ_CONFIG_JSON']\n"
        "cfg = json.loads(open(p).read())\n"
        "print('epochs=', cfg['epochs'])\n",
        encoding="utf-8",
    )
    cmd = f"{sys.executable} train.py"
    _write_cq_yaml(project, cmd, configs={"epochs": 7, "lr": 0.01})

    rc = cli_main(["run", "--path", str(project), "--json"])
    captured = capfd.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["stdout_tail"] == "epochs= 7\n"

    # runtime_cfg.json 이 .pcq/ 아래 작성되었는지 확인
    rcfg = project / ".pcq" / "runtime_cfg.json"
    assert rcfg.exists()
    written = json.loads(rcfg.read_text(encoding="utf-8"))
    assert written["epochs"] == 7
    assert written["lr"] == 0.01


def test_pcq_run_config_only_no_exec(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """--config-only: cmd 실행 안 함, runtime_cfg_path 만 출력."""
    project = tmp_path / "proj"
    project.mkdir()
    # 일부러 실패하는 cmd — config-only 면 실행되지 않으므로 통과해야 한다.
    _write_cq_yaml(project, "false", configs={"epochs": 4})

    rc = cli_main(
        ["run", "--path", str(project), "--config-only", "--json"]
    )
    captured = capfd.readouterr()
    assert rc == 0
    rcfg = project / ".pcq" / "runtime_cfg.json"
    assert rcfg.exists()
    # JSON 출력에 path 노출.
    assert "runtime_cfg.json" in captured.out


def test_pcq_run_forwards_exit_code(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """train.py 가 exit 1 로 끝나면 pcq run 도 exit 1."""
    project = tmp_path / "proj"
    project.mkdir()
    cmd = f'{sys.executable} -c "import sys; sys.exit(7)"'
    _write_cq_yaml(project, cmd, configs={"epochs": 1})

    rc = cli_main(["run", "--path", str(project), "--json"])
    captured = capfd.readouterr()
    # 7 그대로 forward. (process exit code domain — 0..255).
    assert rc == 7
    payload = json.loads(captured.out)
    assert payload["exit_code"] == 7
    assert payload["status"] == "failed"


def test_pcq_run_jsonl_emits_live_events_and_events_file(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """--jsonl stdout is parseable line-delimited events."""
    project = tmp_path / "proj"
    project.mkdir()
    train_py = project / "train.py"
    train_py.write_text(
        "import pcq, sys\n"
        "print('plain-line')\n"
        "pcq.log(epoch=1, eval_acc=0.75)\n"
        "print('warn-line', file=sys.stderr)\n",
        encoding="utf-8",
    )
    _write_cq_yaml(
        project,
        f"{sys.executable} train.py",
        configs={"metrics": []},
    )

    rc = cli_main(["run", "--path", str(project), "--jsonl"])
    captured = capfd.readouterr()
    assert rc == 0
    events = [json.loads(line) for line in captured.out.splitlines()]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert events[0]["event"] == "run.started"
    assert any(event["event"] == "stdout" for event in events)
    metric_events = [event for event in events if event["event"] == "metric"]
    assert metric_events
    assert metric_events[0]["metrics"] == {"epoch": 1, "eval_acc": 0.75}
    assert any(event["event"] == "stderr" for event in events)
    assert events[-1]["event"] == "run.completed"
    events_path = Path(events[-1]["events_path"])
    assert events_path.exists()
    file_events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert file_events == events


def test_pcq_run_events_file_with_final_json_envelope(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """--events + --json keeps stdout as final JSON while writing event JSONL."""
    project = tmp_path / "proj"
    project.mkdir()
    train_py = project / "train.py"
    train_py.write_text(
        "import pcq\n"
        "pcq.log(epoch=2, loss=0.125)\n",
        encoding="utf-8",
    )
    _write_cq_yaml(project, f"{sys.executable} train.py")

    rc = cli_main(
        [
            "run",
            "--path",
            str(project),
            "--events",
            "output/events.jsonl",
            "--json",
        ]
    )
    captured = capfd.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "completed"
    events_path = Path(payload["events_path"])
    assert events_path == project / "output" / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "run.started"
    assert any(event["event"] == "metric" for event in events)
    assert events[-1]["event"] == "run.completed"


def test_pcq_run_json_and_jsonl_are_mutually_exclusive(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    project = tmp_path / "proj"
    project.mkdir()
    _write_cq_yaml(project, f'{sys.executable} -c "print(1)"')

    rc = cli_main(["run", "--path", str(project), "--json", "--jsonl"])
    captured = capfd.readouterr()
    assert rc == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "mutually exclusive" in payload["error"]


def test_pcq_run_rejects_missing_cmd(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """cq.yaml 에 cmd 없으면 명확한 reject + nonzero exit."""
    from pcq.agent.yaml_io import write_yaml

    project = tmp_path / "proj"
    project.mkdir()
    write_yaml(
        {"name": "demo", "configs": {"epochs": 1}}, project / "cq.yaml"
    )

    rc = cli_main(["run", "--path", str(project), "--json"])
    captured = capfd.readouterr()
    assert rc != 0
    # 에러 메시지에 cmd 또는 cq.yaml 언급.
    combined = captured.out + captured.err
    assert "cmd" in combined.lower() or "cq.yaml" in combined.lower()


def test_pcq_run_rejects_missing_cq_yaml(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
):
    """cq.yaml 자체가 없으면 reject."""
    project = tmp_path / "no_cq_yaml_project"
    project.mkdir()

    rc = cli_main(["run", "--path", str(project), "--json"])
    captured = capfd.readouterr()
    assert rc != 0
    combined = captured.out + captured.err
    assert "cq.yaml" in combined.lower()
