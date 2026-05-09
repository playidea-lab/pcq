# Agent Operating Guide

## Purpose

This guide describes how a coding agent should create, modify, validate, and
interpret CQ ML experiments with `pcq`.

The agent should treat `cq.yaml` as the execution contract and `run_record.json`
as the completion record.

## First Principles

- `pcq` is a contract runtime, not a model catalog.
- Built-in atoms are reference examples for onboarding, smoke tests, and
  contract verification.
- Production experiment logic belongs in project-local code.
- Any ML framework is allowed if it honors the CQ contract.
- Prefer structured operations and validation before free-form edits.
- Never infer artifact paths from cwd when `cq.yaml` defines `output_dir`.

## Non-Negotiable Contract

Every agent-authored experiment must make these five things explicit:

1. how to run: top-level `cq.yaml.cmd`
2. what changed: project-local code or `cq.yaml.configs`
3. what to measure: `cq.yaml.metrics` plus monitor/mode in config
4. where to write: `pcq.output_dir()`, never a hard-coded path
5. how to finish: `pcq.save_all(...)` or an equivalent `pcq.finalize_run(...)`

Minimum `cq.yaml`:

```yaml
name: my-experiment
cmd: uv run python train.py
configs:
  output_dir: output
  seed: 42
  strictness: 2
  monitor: eval_acc
  mode: max
metrics:
  - epoch
  - eval_acc
artifacts:
  - output/
inputs: {}
```

Minimum `train.py`:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

# Train or evaluate with any framework.
score = 0.0
history = [{"epoch": 0, "eval_acc": score}]

pcq.log(**history[-1])
pcq.save_all(history=history, status="completed")
```

If a script does not meet this contract, fix the contract before improving the
model.

## Install The Library

Add the runtime to the project. The PyPI distribution is named `pcq`; the
short `cq` name is already occupied and is reserved conceptually for the
managed CQ service boundary. Use Python `import pcq` and the `pcq` CLI command:

```bash
uv add pcq               # core
uv add 'pcq[vision]'     # + torchvision / timm
uv add 'pcq[dist]'       # + accelerate
uv add 'pcq[nlp]'        # + transformers
```

`pcq init-experiment --with-pyproject` generates a fresh project
`pyproject.toml` with `pcq` already wired.

## Install Into Agent Runtimes

`pcq` can install its canonical agent instructions and skill into the current
project. This is explicit; package installation never modifies project files.

```bash
pcq agent install --target codex --path .
pcq agent install --target claude --path .
pcq agent install --target both --path . --dry-run --json
pcq agent status --target both --path . --json
```

Runtime paths:

| Target | Instructions | Skill |
|---|---|---|
| Codex | `AGENTS.md` | `.agents/skills/pcq/SKILL.md` |
| Claude Code | `CLAUDE.md` | `.claude/skills/pcq/SKILL.md` |

Use `pcq init-experiment --agent codex|claude|both` when scaffolding a fresh
experiment and installing agent guidance in the same step.

Use `pcq agent status --json` as the read-only health check before assuming an
agent runtime can see pcq instructions. The status output reports each expected
asset as `installed`, `missing`, `partial`, `stale`, `unmanaged`, or
`divergent`, with a repair command when pcq can suggest one.

## Initial Triage

When given a project:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --json
```

## Running A Project

`pcq run` (v2.12) is the fresh-user entry point — it reads `cq.yaml.cmd`,
dumps `configs` into `<project>/.pcq/runtime_cfg.json`, sets
`CQ_CONFIG_JSON`, and execs the command. The exit code is forwarded.

```bash
# Standard: read cq.yaml.cmd and run it.
pcq run --path .

# Write runtime cfg only, do not exec (CI/debug).
pcq run --path . --config-only --json
```

This replaces the manual `CQ_CONFIG_JSON=cfg.json uv run python train.py`
pattern. PlanSet-expanded directories work the same way:

```bash
pcq apply-planset planset.json --output-pattern "runs/exp{i}"
pcq run --path runs/exp0 --json
```

When `--json` is used, stdout is reserved for a parseable JSON envelope only.
The child process streams are captured to `.pcq/run_stdout.log` and
`.pcq/run_stderr.log`, with `stdout_tail` and `stderr_tail` included in the
JSON result. Use `pcq run --path .` without `--json` when a human wants live
terminal streaming.

When an agent needs live progress, use JSONL events:

```bash
pcq run --path . --jsonl
pcq run --path . --events output/events.jsonl --json
```

`--jsonl` emits `run.started`, `stdout`, `stderr`, `metric`, and final
`run.completed` / `run.failed` events. `--events PATH --json` keeps stdout as a
final JSON envelope and writes the same event stream to a file.

Inside `train.py`, `pcq.config()` falls back to `cq.yaml.configs` even when
`CQ_CONFIG_JSON` is unset, so direct `python train.py` works without
ceremony for ad-hoc local runs.

The agent should identify:

- selected `cq.yaml`
- project root
- command
- output directory
- declared metrics
- available inputs
- existing artifacts
- whether the project uses script, Trainer, Experiment, or custom atoms

## Choosing An Implementation Style

Use this decision tree:

```text
Does an external library own the training lifecycle?
  -> yes: contract script
  -> no:
       Does the user need named components that can be swapped or reused?
         -> yes: project-local atoms + RecipeSpec / Trainer
         -> no:
              Is the loop custom but still mostly Torch-like?
                -> yes: Experiment subclass
                -> no: small contract script or Trainer preset
```

Decision table:

| Situation | Use | Edit Surface |
|---|---|---|
| HF Trainer, TabPFN, PyCaret, sklearn, XGBoost, LightGBM | contract script | `train.py`, `cq.yaml.configs` |
| simple Torch script, one-off baseline | contract script | `train.py`, `cq.yaml.configs` |
| custom model/loss/dataset should be selected by name | project-local atom | `atoms/`, `cq_atoms.py`, recipe/config |
| standard recipe with small overrides | Trainer | `cq.yaml.configs`, atom refs |
| custom Torch loop with checkpoint/resume/device boilerplate | Experiment subclass | `train.py`, local modules |
| project-specific component missing from pcq | project-local atom or helper | project files only |

Default to the smallest surface that keeps the experiment contract-compliant.
Do not add a built-in atom, adapter, or pcq internal feature for a single
project experiment.

### Contract Script

Use a contract script when the framework owns the workflow:

- Hugging Face Trainer
- TabPFN
- PyCaret
- scikit-learn
- XGBoost
- LightGBM
- custom non-Torch code

Required pattern:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# framework code

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

Do not add a framework adapter unless there is a repeated, validated need. The
contract is the integration layer.

Agent rules:

- Put all framework-specific setup in project-local code.
- Keep hyperparameters in `cq.yaml.configs` or `CQ_CONFIG_JSON`, not hidden
  constants.
- Save framework artifacts under `pcq.output_dir()`.
- Convert framework metrics into declared `pcq.log(...)` keys.
- Call `pcq.save_all(...)` even for failed runs when enough evidence exists.

### Project-Local Atoms

Use project-local atoms when a component should be named, selected, swapped,
validated, or reused by an agent.

Examples:

- new model architecture
- custom loss
- custom metric
- dataset wrapper
- optimizer factory
- scheduler factory

Scaffold:

```bash
pcq atoms scaffold model dental_unet
pcq atoms validate-local
pcq atoms smoke model dental_unet --load-project .
```

The implementation can be arbitrary Python. The public surface must declare:

- name
- kind
- params
- tasks
- input contract
- output contract
- label or metric contract when applicable
- smoke safety when applicable

### Trainer

Use `Trainer` when a recipe or atom composition is the main work:

```python
import pcq

cfg = pcq.config()
pcq.seed_everything(cfg.get("seed", 42))
pcq.Trainer.from_cfg(cfg).fit()
```

Trainer is useful when the agent needs to swap atoms through config or
structured plans.

### Experiment

Use `Experiment` when the training loop is still PyTorch-like but needs custom
logic:

```python
import pcq

class MyExperiment(pcq.Experiment):
    ...

cfg = pcq.config()
MyExperiment(cfg=cfg).fit()
```

Keep Experiment subclasses small. If most logic is a reusable component, move it
to a project-local atom.

## Copyable Authoring Patterns

These examples are intentionally framework-light. The important part is not the
library being used; the important part is that the CQ contract is honored.
The always-runnable repository example is
`examples/contract_numpy.py`; `examples/contract_sklearn.py` is an optional
third-party dependency example.

### Torch Contract Script

```yaml
name: torch-mlp
cmd: uv run python train.py
configs:
  output_dir: runs/torch_mlp
  seed: 42
  strictness: 3
  monitor: eval_acc
  mode: max
  lr: 0.05
metrics:
  - epoch
  - train_loss
  - eval_acc
artifacts:
  - runs/torch_mlp/
inputs:
  synthetic:
    opaque: true
    reason: generated locally for smoke testing
```

```python
import pcq
import torch
import torch.nn as nn

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
x = torch.randn(32, 4, generator=g)
y = torch.randint(0, 2, (32,), generator=g)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
opt = torch.optim.SGD(model.parameters(), lr=float(cfg.get("lr", 0.05)))
loss_fn = nn.CrossEntropyLoss()

logits = model(x)
loss = loss_fn(logits, y)
opt.zero_grad()
loss.backward()
opt.step()

eval_acc = (logits.argmax(-1) == y).float().mean().item()
torch.save(model.state_dict(), out / "model.pt")

history = [{
    "epoch": 0,
    "train_loss": float(loss.detach()),
    "eval_acc": float(eval_acc),
}]
pcq.log(**history[-1])
pcq.save_all(history=history, artifacts={"model": "model.pt"})
```

### sklearn Contract Script

Use this shape when sklearn is a project dependency. `pcq` does not need a
sklearn adapter.

```yaml
name: sklearn-classifier
cmd: uv run python train.py
configs:
  output_dir: runs/sklearn
  seed: 42
  strictness: 3
  monitor: eval_acc
  mode: max
  n_estimators: 100
metrics:
  - epoch
  - eval_acc
artifacts:
  - runs/sklearn/
inputs:
  iris:
    source: sklearn.datasets.load_iris
```

```python
import pickle

import pcq
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

x, y = load_iris(return_X_y=True)
x_train, x_eval, y_train, y_eval = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=int(cfg.get("seed", 42)),
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=int(cfg.get("n_estimators", 100)),
    random_state=int(cfg.get("seed", 42)),
)
model.fit(x_train, y_train)
eval_acc = float(model.score(x_eval, y_eval))

with (out / "model.pkl").open("wb") as f:
    pickle.dump(model, f)

history = [{"epoch": 0, "eval_acc": eval_acc}]
pcq.log(**history[-1])
pcq.save_all(history=history, artifacts={"model": "model.pkl"})
```

### Arbitrary Framework Script

Use this shape for HF Trainer, TabPFN, PyCaret, command-driven binaries, or any
other library with its own lifecycle.

```python
import json
import subprocess

import pcq

cfg = pcq.config()
out = pcq.output_dir()

result_path = out / "framework_result.json"
model_path = out / "model.bin"

subprocess.run(
    [
        "some-framework-train",
        "--seed",
        str(cfg.get("seed", 42)),
        "--out",
        str(out),
    ],
    check=True,
)

result = json.loads(result_path.read_text())
history = [{
    "epoch": int(result.get("epoch", 0)),
    "eval_score": float(result["eval_score"]),
}]

pcq.log(**history[-1])
pcq.save_all(history=history, artifacts={"model": model_path.name})
```

If the external framework fails after producing partial evidence, catch the
exception, write a structured `failure`, and call `pcq.save_all(status="failed",
failure=failure)` before exiting non-zero.

## Editing Rules

### Prefer Local Project Code

When a user asks for a new model, loss, metric, or data transform, add it under:

```text
cq_atoms.py
atoms/
recipes/
train.py
```

Do not edit `pcq` internals to support one project-specific component.

### Keep Built-Ins As Examples

Reference examples such as `pcq.examples.models.small_cnn`,
`pcq.examples.datasets.fake`, `pcq.examples.optim.adamw`,
`pcq.examples.sched.cosine`, and `pcq.loss.cross_entropy` may be used for smoke
and tutorial projects, but the agent should not treat them as the production
extension path.

### Preserve The Contract

After edits, confirm:

- `cq.yaml.cmd` still points to the entrypoint
- user code calls `pcq.config()` or otherwise reads the runtime config
- user code writes to `pcq.output_dir()`
- declared metrics match emitted metrics
- `pcq.save_all(...)` or equivalent artifacts exist
- output artifacts are under the resolved output directory

## Forbidden Patterns

These patterns make an experiment hard for an agent or service to operate.

| Pattern | Why It Is Bad | Fix |
|---|---|---|
| `Path("output")`, `"output/model.pt"` | ignores custom `configs.output_dir` | use `out = pcq.output_dir()` |
| metrics emitted but not declared | strict schema cannot validate the run | update `cq.yaml.metrics` |
| model/loss/dataset added inside `pcq` internals | project-specific code becomes library scope | create project-local atom/helper |
| hidden network download at import time | inspect/smoke can hang or fail offline | make data preparation explicit in `cmd` or inputs |
| project atom side effects during import | read-only inspect becomes unsafe | move heavy work into factory functions |
| framework adapter for a one-off library | duplicates the real framework API | use contract script |
| artifact writes outside output dir | worker collection misses evidence | write under `pcq.output_dir()` |
| failure exits before standard artifacts | service sees only process failure | save structured failed run when possible |
| unseeded random split | result cannot be reproduced | use `cfg.seed` and record inputs |

When a forbidden pattern is found, fix it before changing model quality.

## Pre-Run Checklist

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

For atom work:

```bash
pcq atoms validate-local
pcq atoms smoke <kind> <name> --load-project .
```

For Python syntax:

```bash
uv run python -m py_compile train.py
```

For reproducibility-sensitive work:

```bash
pcq validate . --strictness 3 --json
```

Use project-specific tests when present.

## Post-Run Checklist

```bash
pcq validate-run <output_dir> --strictness 3 --json
pcq describe-run <output_dir> --json
```

The agent should read:

- run status
- best metric
- monitor direction
- validation status
- artifact summary
- source snapshot
- input identity
- parent run lineage when present
- reproducibility evidence
- `decision_facts` booleans and counts

## Follow-Up Experiment Loop

```text
describe previous run
  -> identify bottleneck or metric target
  -> modify project-local code or config
  -> validate
  -> run
  -> validate-run
  -> compare-runs
  -> decide next action
```

Use:

```bash
pcq compare-runs <old_output_dir> <new_output_dir> --json
pcq lineage <new_output_dir> --json
```

`compare-runs` should be read as comparison evidence: best/last metric movement,
config/input changes, validation and failure differences, artifact/source
differences, and `decision_facts`. The agent decides whether that evidence is
enough to continue, branch, rollback, or stop.

## Common Failure Patterns

### Artifacts Written To The Wrong Directory

Symptom:

- `metrics.json` exists in `output/`
- `run_record.json` exists in `runs/exp001/`

Fix:

- use `pcq.output_dir()`
- ensure contract helpers share the resolved runtime context
- avoid `Path("output")` in user code

### Missing Metric Declaration

Symptom:

- stdout warning for undeclared metric
- strict validation failure

Fix:

- add metric to `cq.yaml.metrics`
- or ensure `_metrics_declared` is injected from the service

### Project Atom Not Found

Symptom:

- `unknown model`
- `unknown loss`
- `pcq atoms list` does not show project atom

Fix:

- import atom module from `cq_atoms.py`
- ensure `atoms/__init__.py` exists when using package imports
- run `pcq atoms list --load-project . --source project --json`

### Variable Output Shapes

Symptom:

- DataLoader collate error
- loss shape mismatch

Fix:

- add resize/pad/collate logic in the project dataset
- update atom contracts to reflect actual shapes
- smoke test the affected atom

## What The Agent Should Produce

A completed experiment change should leave:

- updated project files
- valid `cq.yaml`
- passing pre-run validation
- standard post-run artifacts, when a run was executed
- `run_record.json`
- a concise summary of changed intent and observed result
