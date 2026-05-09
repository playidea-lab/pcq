# Agent Operating Guide

## Purpose

This guide describes how a coding agent should create, modify, validate, and
interpret ML experiments with `pcq`.

The agent should treat `cq.yaml` as the execution contract and
`run_record.json` as the completion record.

## First Principles

- `pcq` is an experiment boundary, not a model catalog.
- `pcq` does not compete with PyTorch, HF Trainer, Lightning, sklearn, XGBoost,
  TabPFN, PyCaret, shell scripts, or custom project code.
- The contract is the adapter.
- Production experiment logic belongs in project-local files.
- Prefer JSON/JSONL commands over scraping terminal prose.
- Treat exit code as incomplete evidence until `validate-run` has been read.
- Never infer artifact paths from cwd when `cq.yaml` defines `output_dir`.

## Non-Negotiable Contract

Every agent-authored experiment must make these five things explicit:

1. how to run: top-level `cq.yaml.cmd`
2. what changed: project-local code or `cq.yaml.configs`
3. what to measure: `cq.yaml.metrics` plus monitor/mode in config
4. where to write: `pcq.output_dir()`, never a hard-coded path
5. how to finish: `pcq.save_all(...)` or equivalent `pcq.finalize_run(...)`

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

## Install

```bash
uv add pcq
```

The PyPI distribution, import, and CLI are all named `pcq`:

```python
import pcq
```

```bash
pcq --help
```

## Install Agent Runtime Assets

`pcq` can install canonical agent instructions and skill files into the current
project. This is explicit; package installation never modifies project files.

```bash
pcq agent install --target codex --path .
pcq agent install --target claude --path .
pcq agent install --target both --path . --dry-run --json
pcq agent status --target both --path . --json
```

Use `pcq agent status --json` as a read-only health check before assuming an
agent runtime can see pcq instructions.

## Initial Triage

When given a project:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

Identify:

- selected `cq.yaml`
- project root
- command
- output directory
- declared metrics
- available inputs
- existing artifacts
- prior run records
- whether evidence is already valid

Read-side commands should not train, download data, or import heavy project
code by default.

## Choosing An Implementation Style

Default to a contract script.

Use project-local helpers only when they reduce real project complexity. Do not
add a framework adapter or core `pcq` feature for one experiment.

| Situation | Use | Edit Surface |
|---|---|---|
| HF Trainer, Lightning, TabPFN, PyCaret, sklearn, XGBoost, LightGBM | contract script | `train.py`, local modules, `cq.yaml.configs` |
| simple Torch baseline | contract script | `train.py`, `cq.yaml.configs` |
| custom training lifecycle | contract script or local helper | project files only |
| command-driven external tool | contract script wrapper | `train.py`, shell command, output parser |
| repeated project pattern | local helper module | project files only |

The agent should not ask whether `pcq` has built-in support for a framework.
If the framework can be called from project-local code and its result can be
converted to metrics/artifacts, it is usable.

## Contract Script Pattern

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

Agent rules:

- put framework-specific setup in project-local code
- keep hyperparameters in `cq.yaml.configs` or runtime config
- save framework artifacts under `pcq.output_dir()`
- convert framework metrics into declared `pcq.log(...)` keys
- call `pcq.save_all(...)` for completed runs
- preserve structured failure evidence for failed runs when possible

## Running A Project

`pcq run` reads `cq.yaml.cmd`, writes runtime config into `.pcq/`, sets
`CQ_CONFIG_JSON`, and executes the command.

```bash
pcq run --path .
pcq run --path . --json
pcq run --path . --jsonl
pcq run --path . --events output/events.jsonl --json
```

Use:

- `--json` for one final parseable envelope
- `--jsonl` for live events
- `--events PATH --json` for final JSON stdout plus persisted live events

Do not parse human terminal output when JSON or JSONL is available.

## Post-Run Checklist

```bash
pcq validate-run output --strictness 3 --json
pcq describe-run output --json
```

Read:

- run status
- best metric
- monitor direction
- validation status
- artifact summary
- source snapshot
- input identity
- parent run lineage
- reproducibility evidence
- `decision_facts`
- structured failure evidence

## Comparison Loop

```bash
pcq compare-runs parent_output candidate_output --json
pcq lineage candidate_output --json
```

`compare-runs` is evidence, not policy. The agent decides whether to continue,
branch, rollback, rerun, or stop.

## Iteration

Use structured config plans when a bounded config change is enough:

```bash
pcq apply-plan experiment.plan.json --json
```

Use direct project-local code edits when the next experiment needs new research
logic. Keep those edits in the user project, not in `pcq` internals.

## Forbidden Patterns

| Pattern | Why it is bad | Fix |
|---|---|---|
| `Path("output")`, `"output/model.pt"` | ignores custom `configs.output_dir` | `pcq.output_dir()` |
| metrics emitted but not declared | strict schema cannot validate the run | update `cq.yaml.metrics` |
| process exit code treated as success | artifacts may be missing | run `validate-run` |
| parsing prose logs | fragile agent behavior | use JSON/JSONL |
| one-off framework adapter in pcq core | competes with the framework | contract script |
| artifact writes outside output dir | worker collection misses evidence | write under `pcq.output_dir()` |
| failure exits before evidence | service sees only process failure | save structured failed run when possible |
| unseeded random split | result cannot be reproduced | use `cfg.seed` and record inputs |

## Failure Handling

When a framework fails after producing partial evidence:

```python
try:
    ...
except Exception as exc:
    pcq.save_all(
        history=history,
        status="failed",
        failure={"type": type(exc).__name__, "message": str(exc)},
    )
    raise
```

The raised exception should still produce a non-zero process exit. The saved
evidence lets `describe-run` explain the failure.

## Final Agent Rule

Operate the experiment boundary.

Do not compete with the training method. Use whatever method fits, then make the
run observable, verifiable, comparable, and repeatable through `pcq`.
