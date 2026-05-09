# pcq Introduction

`pcq` is an open-source experiment evidence and control library.

It does not replace your training framework. It standardizes what surrounds a
training run: configuration, metric emission, artifact layout, validation,
lineage, comparison, and the final run record an agent or service can reason
about.

The central idea:

```text
pcq does not operate the model.
pcq operates the experiment boundary.
```

## What It Solves

ML experiments often begin as local scripts and later need to run in CI, on a
remote worker, or under an automated coding agent. The training code may work,
but the surrounding evidence is usually inconsistent:

- Where did the config come from?
- Which metrics were declared and emitted?
- Which files are the official artifacts?
- Which git commit, lockfile, and input identity produced the result?
- Did the run finish as a valid experiment or merely exit with code 0?
- Can an agent compare this run with a previous run and choose the next step?

`pcq` makes those answers explicit.

## The Core Contract

```text
cq.yaml
  -> command, configs, metrics, inputs, artifacts

training code
  -> PyTorch, HF Trainer, Lightning, sklearn, TabPFN, PyCaret,
     XGBoost, shell script, remote job, or custom code

pcq
  -> config loading, metric logging, artifact helpers, validation, RunRecord

output/
  -> config.json, metrics.json, manifest.json, run_summary.json,
     run_record.json, validation_report.json
```

`cq.yaml`, `CQ_CONFIG_JSON`, and `cq://` are runtime contract names and stay
stable. `pcq` is the Python evidence/control library that consumes that
contract.

## Who It Is For

- Researchers who want experiments that are easy to rerun and compare.
- ML engineers who need a small contract layer instead of another full trainer.
- Teams using coding agents to modify, run, validate, and compare experiments.
- CI or orchestration systems that need predictable JSON output and artifacts.
- CQ service users who want local projects to be CQ-runnable without service
  imports in their training code.

## What pcq Is Not

- Not a model zoo.
- Not a training framework.
- Not an experiment tracking SaaS.
- Not a replacement for PyTorch Lightning, HF Trainer, W&B, MLflow, DVC, or CQ.
- Not a framework adapter matrix.

The contract is the adapter. If a script can read config, emit metrics, and
write standard artifacts, it can be operated by `pcq`.

## Primary Workflow

### 1. Write A Contract Script

Use whatever training stack fits the problem.

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# Any ML framework can run here.
score = 0.74

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

### 2. Run With Machine Output

```bash
pcq run --path . --json
pcq run --path . --jsonl
```

### 3. Validate And Describe Evidence

```bash
pcq validate-run output --strictness 3 --json
pcq describe-run output --json
```

### 4. Compare And Iterate

```bash
pcq compare-runs old_output new_output --json
pcq lineage new_output --json
pcq apply-plan experiment.plan.json --json
```

## Agent-Operable By Design

Agents need structured surfaces, not prose-only instructions. `pcq` provides:

- `pcq resolve --json`
- `pcq inspect --json`
- `pcq validate --json`
- `pcq run --json`
- `pcq run --jsonl`
- `pcq validate-run --json`
- `pcq describe-run --json`
- `pcq compare-runs --json`
- `pcq lineage --json`
- `pcq apply-plan --json`
- `pcq agent install/status --json`

The goal is that an agent can understand the experiment contract, make a
bounded project-local change, run it, validate it, compare it with previous
evidence, and decide the next step.

## Standard Artifacts

A complete run should produce:

- `config.json`
- `metrics.json`
- `manifest.json`
- `run_summary.json`
- `run_record.json`
- `validation_report.json`

`run_record.json` is the canonical completion object. It combines execution,
source, environment, input identity, metric schema, artifact manifest, agent
provenance, validation, and summary evidence.

## Relationship With CQ

```text
pcq = open-source experiment evidence/control library
cq  = managed execution, queue, artifact collection, dashboard, agent loop
```

CQ service is one consumer of the contract. `pcq` remains useful without CQ:
locally, in CI, in notebooks, and in third-party orchestrators.

## Install

```bash
uv add pcq
```

## Start

```bash
pcq init-experiment --style script --output ./my-exp --with-pyproject
cd ./my-exp
uv sync
pcq run --jsonl
pcq validate-run output --json
pcq describe-run output --json
```

## Current Direction

The v4 direction is contract-first:

- no required trainer abstraction
- no production model/loss/dataset catalog in core
- no framework adapter matrix
- project-local training code is first-class
- JSON/JSONL evidence and control surfaces are the product

Read [pcq v4 Direction](V4_DIRECTION.md) next.
