"""Golden E2E workflows for agent-authored pcq experiments."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from pcq.agent.yaml_io import read_yaml, write_yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CQ_CONFIG_JSON", None)
    env.pop("CQ_DECLARED_METRICS", None)
    return env


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    expected: int | None = 0,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        env=env or _clean_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if expected is not None:
        assert proc.returncode == expected, (
            f"command failed: {args}\n"
            f"cwd={cwd}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _cli_json(*args: str, expected: int = 0, timeout: int = 120) -> dict[str, Any]:
    proc = _run(
        [sys.executable, "-m", "pcq.cli", *args, "--json"],
        cwd=REPO_ROOT,
        timeout=timeout,
        expected=expected,
    )
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"CLI output was not JSON:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        ) from e
    assert isinstance(out, dict)
    return out


def _git(project: Path, *args: str) -> None:
    _run(["git", *args], cwd=project)


def _init_git(project: Path) -> None:
    _git(project, "init", "-q")
    _git(project, "config", "user.email", "cq@example.test")
    _git(project, "config", "user.name", "cq test")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "init", "-q")


def _write_lockfiles(project: Path) -> None:
    (project / "pyproject.toml").write_text(
        "[project]\nname='golden-e2e'\n", encoding="utf-8"
    )
    (project / "uv.lock").write_text("# fake lock for golden e2e\n", encoding="utf-8")


def _write_worker_cfg(
    project: Path,
    *,
    output_dir: str,
    metrics: list[str],
    monitor: str = "eval_acc",
    mode: str = "max",
    **extra: Any,
) -> Path:
    cfg: dict[str, Any] = {
        "output_dir": output_dir,
        "seed": 42,
        "strictness": 3,
        "monitor": monitor,
        "mode": mode,
        "_metrics_declared": metrics,
        "_cmd": "uv run python train.py",
    }
    cfg.update(extra)
    path = project / "worker_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _run_contract_pipeline(
    project: Path,
    *,
    output_dir: Path,
    cfg_path: Path,
    inspect_load_atoms: bool = False,
    run_expected: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    inspect_args = ["inspect", str(project)]
    if inspect_load_atoms:
        inspect_args.append("--load-project-atoms")
    inspected = _cli_json(*inspect_args)
    assert inspected["has_cq_yaml"] is True

    pre = _cli_json("validate", str(project), "--strictness", "3")
    assert pre["status"] in ("pass", "warn")

    env = _clean_env()
    env["CQ_CONFIG_JSON"] = str(cfg_path)
    _run(
        [sys.executable, "train.py"],
        cwd=project,
        env=env,
        timeout=120,
        expected=run_expected,
    )

    post = _cli_json(
        "validate-run", str(output_dir), "--strictness", "3",
    )
    assert post["status"] == "pass"

    desc = _cli_json("describe-run", str(output_dir))
    assert desc["status"] in ("completed", "failed")

    for name in (
        "config.json",
        "metrics.json",
        "manifest.json",
        "run_summary.json",
        "run_record.json",
        "validation_report.json",
    ):
        assert (output_dir / name).exists(), f"missing {name}"

    return post, desc


def _write_script_project(
    project: Path,
    *,
    name: str,
    output_dir: str,
    metrics: list[str],
    script: str,
    inputs: dict[str, Any] | None = None,
) -> None:
    project.mkdir(parents=True, exist_ok=True)
    _write_lockfiles(project)
    (project / "cq.yaml").write_text(
        "\n".join([
            f"name: {name}",
            "cmd: uv run python train.py",
            "configs:",
            f"  output_dir: {output_dir}",
            "  seed: 42",
            "  strictness: 3",
            "  monitor: eval_acc",
            "  mode: max",
            "metrics:",
            *[f"  - {m}" for m in metrics],
            f"artifacts: [{output_dir}/]",
            "inputs: {}" if inputs is None else "inputs:",
        ])
        + "\n",
        encoding="utf-8",
    )
    if inputs is not None:
        data = read_yaml(project / "cq.yaml")
        data["inputs"] = inputs
        write_yaml(data, project / "cq.yaml")
    (project / "train.py").write_text(textwrap.dedent(script).lstrip(), encoding="utf-8")


def test_golden_synthetic_mnist_mlp_script_e2e(tmp_path):
    """Agent-authored torch script runs the full pcq contract loop."""
    project = tmp_path / "mnist_mlp"
    metrics = ["epoch", "train_loss", "eval_loss", "eval_acc"]
    _write_script_project(
        project,
        name="golden-mnist-mlp",
        output_dir="runs/mnist_mlp",
        metrics=metrics,
        script="""
            import pcq
            import torch
            import torch.nn as nn

            cfg = pcq.config()
            out = pcq.output_dir()
            pcq.seed_everything(cfg.get("seed", 42))

            g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
            x_train = torch.randn(32, 1, 28, 28, generator=g)
            y_train = torch.randint(0, 10, (32,), generator=g)
            x_eval = torch.randn(16, 1, 28, 28, generator=g)
            y_eval = torch.randint(0, 10, (16,), generator=g)

            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 16),
                nn.ReLU(),
                nn.Linear(16, 10),
            )
            loss_fn = nn.CrossEntropyLoss()
            opt = torch.optim.SGD(model.parameters(), lr=0.05)

            logits = model(x_train)
            loss = loss_fn(logits, y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                eval_logits = model(x_eval)
                eval_loss = loss_fn(eval_logits, y_eval)
                eval_acc = (eval_logits.argmax(-1) == y_eval).float().mean().item()

            torch.save(model.state_dict(), out / "model.pt")
            history = [{
                "epoch": 0,
                "train_loss": float(loss.detach()),
                "eval_loss": float(eval_loss.detach()),
                "eval_acc": float(eval_acc),
            }]
            pcq.log(**history[0])
            pcq.save_all(history=history, artifacts={"model": "model.pt"})
        """,
    )
    cfg_path = _write_worker_cfg(
        project, output_dir="runs/mnist_mlp", metrics=metrics,
    )
    _init_git(project)

    output_dir = project / "runs" / "mnist_mlp"
    _, desc = _run_contract_pipeline(
        project, output_dir=output_dir, cfg_path=cfg_path,
    )
    assert desc["target_metric"] == "eval_acc"
    assert (output_dir / "model.pt").exists()


def test_golden_trainer_fake_smoke_e2e(tmp_path):
    """Generated trainer project passes inspect/validate/run/post-run checks."""
    project = tmp_path / "trainer_fake"
    _cli_json(
        "init-experiment",
        "--preset", "vision/fake_smoke",
        "--output", str(project),
        "--force",
    )
    _write_lockfiles(project)
    data = read_yaml(project / "cq.yaml")
    data["configs"]["output_dir"] = "runs/fake_smoke"
    data["configs"]["strictness"] = 3
    data["inputs"] = {}
    data["artifacts"] = ["runs/fake_smoke/"]
    write_yaml(data, project / "cq.yaml")
    metrics = list(data["metrics"])
    cfg_path = _write_worker_cfg(
        project,
        output_dir="runs/fake_smoke",
        metrics=metrics,
        preset="vision/fake_smoke",
        epochs=1,
        batch_size=8,
    )
    _init_git(project)

    output_dir = project / "runs" / "fake_smoke"
    post, desc = _run_contract_pipeline(
        project, output_dir=output_dir, cfg_path=cfg_path,
    )
    assert post["strictness_name"] == "reproducible"
    assert desc["epochs_completed"] >= 1
    assert (output_dir / "model.pt").exists()


def test_golden_project_atom_scaffold_smoke_and_run_e2e(tmp_path):
    """Project-local atom scaffold can be loaded, smoked, and used in a run."""
    project = tmp_path / "atom_project"
    metrics = ["epoch", "eval_acc"]
    _write_script_project(
        project,
        name="golden-project-atom",
        output_dir="output",
        metrics=metrics,
        inputs={"synthetic": {"opaque": True, "reason": "generated in test"}},
        script="""
            import pcq
            import torch
            import torch.nn.functional as F
            from pcq import registry
            from pcq.registry.loader import load_project_atoms

            cfg = pcq.config()
            out = pcq.output_dir()
            pcq.seed_everything(cfg.get("seed", 42))
            load_project_atoms(".")

            model = registry.models.get("e2e_model").factory(
                in_channels=3, num_classes=2,
            )
            x = torch.randn(4, 3, 8, 8)
            y = torch.randint(0, 2, (4,))
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            eval_acc = (logits.argmax(-1) == y).float().mean().item()

            torch.save(model.state_dict(), out / "model.pt")
            history = [{"epoch": 0, "eval_acc": float(eval_acc)}]
            pcq.log(**history[0])
            pcq.save_all(history=history, artifacts={"model": "model.pt"})
        """,
    )
    scaffold = _cli_json(
        "atoms", "scaffold", "model", "e2e_model",
        "--path", str(project),
    )
    assert scaffold["status"] == "created"
    atom_validation = _cli_json("atoms", "validate-local", str(project))
    assert atom_validation["status"] in ("pass", "warn")
    smoke = _cli_json(
        "atoms", "smoke", "model", "e2e_model",
        "--load-project", str(project),
    )
    assert smoke["passed"] is True

    cfg_path = _write_worker_cfg(project, output_dir="output", metrics=metrics)
    _init_git(project)

    post, desc = _run_contract_pipeline(
        project,
        output_dir=project / "output",
        cfg_path=cfg_path,
        inspect_load_atoms=True,
    )
    assert post["status"] == "pass"
    assert desc["status"] == "completed"


def test_golden_failed_run_and_lineage_e2e(tmp_path):
    """Structured failed runs and parent lineage remain machine-readable."""
    project = tmp_path / "failed_lineage"
    metrics = ["epoch", "eval_acc"]
    _write_script_project(
        project,
        name="golden-lineage",
        output_dir="parent",
        metrics=metrics,
        inputs={},
        script="""
            import pcq

            cfg = pcq.config()
            pcq.output_dir()
            pcq.seed_everything(cfg.get("seed", 42))
            value = float(cfg.get("eval_acc", 0.5))
            history = [{"epoch": 0, "eval_acc": value}]

            if cfg.get("force_failure"):
                failure = {
                    "category": "runtime_error",
                    "message": "intentional golden failure",
                    "suggested_fix": "fix the generated training step",
                }
                pcq.log(**history[0])
                pcq.save_all(history=history, status="failed", failure=failure)
                raise SystemExit(2)

            pcq.log(**history[0])
            pcq.save_all(history=history, status="completed")
        """,
    )
    _init_git(project)

    parent_cfg = _write_worker_cfg(
        project, output_dir="parent", metrics=metrics, eval_acc=0.6,
    )
    _run_contract_pipeline(
        project, output_dir=project / "parent", cfg_path=parent_cfg,
    )
    parent_rr = json.loads((project / "parent" / "run_record.json").read_text())

    child_cfg = _write_worker_cfg(
        project,
        output_dir="child",
        metrics=metrics,
        eval_acc=0.0,
        force_failure=True,
        _parent_run_id=parent_rr["run"]["id"],
        _parent_run_path=str(project / "parent"),
    )
    _run_contract_pipeline(
        project,
        output_dir=project / "child",
        cfg_path=child_cfg,
        run_expected=2,
    )

    child_desc = _cli_json("describe-run", str(project / "child"))
    assert child_desc["status"] == "failed"
    assert child_desc["failure"]["category"] == "runtime_error"

    lineage = _cli_json("lineage", str(project / "child"))
    assert len(lineage["chain"]) == 2
    assert lineage["chain"][0]["status"] == "failed"
    assert lineage["chain"][1]["run_id"] == parent_rr["run"]["id"]
