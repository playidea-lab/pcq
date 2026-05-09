# pcq Introduction

`pcq` is an open-source ML experiment contract library.

It does not replace your training framework. It standardizes what surrounds a
training run: configuration, metric emission, artifact layout, validation,
lineage, and the final run record an agent or service can reason about.

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
  -> PyTorch, sklearn, HF Trainer, TabPFN, PyCaret, XGBoost, custom code

pcq
  -> config loading, metric logging, artifact helpers, validation, RunRecord

output/
  -> config.json, metrics.json, manifest.json, run_summary.json,
     run_record.json, validation_report.json
```

`cq.yaml`, `CQ_CONFIG_JSON`, and `cq://` are CQ runtime contract names and stay
stable. `pcq` is the Python authoring and evidence library that consumes that
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
- Not an experiment tracking SaaS.
- Not a replacement for PyTorch Lightning, HF Trainer, W&B, MLflow, or CQ.
- Not a framework adapter matrix.

The contract is the adapter. If a script can read config, emit metrics, and
write standard artifacts, it can be operated by `pcq`.

## Three Ways To Use It

### 1. Contract Script

Use this when another framework owns the training loop.

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

### 2. Experiment

Use this when you want a small PyTorch training loop with standard checkpointing
and artifacts.

```python
import pcq

class MyExperiment(pcq.Experiment):
    ...

MyExperiment().fit()
```

### 3. Trainer + Project Atoms

Use this when an agent or team should swap models, datasets, losses, optimizers,
or schedulers by name.

```python
import pcq

cfg = pcq.config()
pcq.Trainer.from_cfg(cfg).fit()
```

Project-local atoms live in your project, not in the `pcq` package. Built-in
atoms are reference examples for smoke tests and onboarding.

## Agent-Operable By Design

Agents need structured surfaces, not prose-only instructions. `pcq` provides:

- `pcq inspect --json`
- `pcq validate --json`
- `pcq run --json`
- `pcq validate-run --json`
- `pcq describe-run --json`
- `pcq compare-runs --json`
- `pcq lineage --json`
- `pcq agent install/status --json`

The goal is that an agent can look at a project, understand the experiment
contract, make a bounded change, run it, validate it, and compare it with
previous evidence.

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
pcq = open-source experiment contract library
cq  = managed execution, queue, artifact collection, dashboard, agent loop
```

CQ service is one consumer of the contract. `pcq` remains useful without CQ:
locally, in CI, in notebooks, and in third-party orchestrators.

## Install

```bash
uv add pcq
```

Optional extras:

```bash
uv add 'pcq[vision]'
uv add 'pcq[dist]'
uv add 'pcq[nlp]'
uv add 'pcq[yaml]'
```

## Start

```bash
pcq init-experiment --style script --output ./my-exp --with-pyproject
cd ./my-exp
uv sync
pcq run --json
pcq validate-run output --json
pcq describe-run output --json
```

## Current Release Line

`pcq` v3 is the single-name release line:

- PyPI package: `pcq`
- Python import: `import pcq`
- CLI: `pcq`
- GitHub repository: `https://github.com/playidea-lab/pcq`
- Runtime workspace: `.pcq/`
- JSON contract namespace: `pcq.*`

The CQ runtime contract names `cq.yaml`, `CQ_CONFIG_JSON`, and `cq://` remain
unchanged.
