# Agent-Operable pcq

## Purpose

This document defines what `agent-operable` means for `pcq` v4.

The goal is not for an agent to operate a `pcq` trainer, recipe catalog, or
model registry. The goal is for an agent to operate the experiment boundary:
create or modify project-local training code, execute it, observe it, validate
the evidence, compare it with previous runs, and decide the next step from
machine-readable facts.

See [pcq v4 Direction](V4_DIRECTION.md) for the product identity.

## Definition

`pcq` is agent-operable when an agent can perform this loop without scraping
human prose and without editing `pcq` internals:

```text
inspect contract
  -> validate pre-run state
  -> author or modify project-local experiment code
  -> run with JSON/JSONL evidence
  -> validate post-run artifacts
  -> describe decision facts
  -> compare with a parent run
  -> apply the next structured plan
  -> repeat or stop
```

## Agent-Readable Is The Base Layer

Agent-readable means an agent can parse facts.

Examples:

```bash
pcq run --json
pcq run --jsonl
pcq describe-run output --json
pcq compare-runs old_output new_output --json
```

This is necessary but insufficient. A JSON schema alone is not agent-operable;
it only makes results easier to read.

## Agent-Verifiable Is The Trust Layer

Agent-verifiable means an agent can check whether the result has enough
evidence to be trusted.

Examples:

```bash
pcq validate . --strictness 2 --json
pcq validate-run output --strictness 3 --json
```

The validation report should state:

- selected strictness
- required evidence
- present evidence
- missing evidence
- blocking failures
- warnings

## Agent-Operable Is The Control Layer

Agent-operable means an agent can act through stable commands.

Primary control surfaces:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --json
pcq run --json
pcq run --jsonl
pcq validate-run output --json
pcq describe-run output --json
pcq compare-runs old_output new_output --json
pcq lineage output --json
pcq apply-plan experiment.plan.json --json
```

The agent should not need a private mental model of `pcq` internals. The CLI
and artifacts are the operating surface.

## Agent-Authorable Is The Creation Layer

Agent-authorable means an agent can write a new experiment and keep it
contract-compliant.

The default authoring pattern is a contract script:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# Use any training or evaluation tool here.

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

This authoring path supports PyTorch, Hugging Face Trainer, Lightning, sklearn,
XGBoost, TabPFN, PyCaret, shell commands, remote job wrappers, and custom code
without framework adapters.

## Boundary Between pcq And The Agent

`pcq` reports facts:

- what command ran
- which config was used
- which metrics were emitted
- which artifacts exist
- which evidence is missing
- whether validation passed
- whether two runs are comparable
- whether a metric improved, regressed, tied, or is unavailable
- which parent run a candidate descends from

The agent or service decides policy:

- accept the run
- reject the run
- rerun
- branch
- rollback
- ask the user
- schedule a bigger job
- stop the loop

This separation matters. If `pcq` starts deciding research policy, it becomes a
workflow agent. If it only emits raw files, it stays merely agent-readable. The
right product boundary is structured facts plus safe control surfaces.

## Boundary Between pcq And CQ Service

CQ service owns:

- user goal intake
- LLM prompting and tool strategy
- workspace lifecycle
- permissions and sandboxing
- queueing and GPU scheduling
- credential management
- dashboard and collaboration UI
- cross-project policy

`pcq` owns:

- local project contract resolution
- local run execution wrapper
- local metric/artifact evidence
- local validation
- local run description
- local comparison and lineage facts
- local structured plan application
- agent-readable documentation assets

CQ service can be powerful because `pcq` gives it a standard local substrate.
But `pcq` must remain useful without CQ service.

## Required Properties

### Read-Side Safety

Read-side commands should be safe for an agent to call before trusting a
project.

Required:

- do not train
- do not download datasets
- do not import heavy optional dependencies by default
- do not create output directories from read-only inspection
- do not execute arbitrary project code unless explicitly requested
- return structured warnings instead of unhandled tracebacks

### Output Directory Consistency

All commands must resolve output paths through the same project resolver.

Required:

- `pcq.output_dir()`
- `save_all`
- `finalize`
- `validate-run`
- `inspect`
- `describe-run`
- `compare-runs`

must agree on `configs.output_dir` from `cq.yaml` / runtime config.

### Failure As Evidence

A failed experiment should still be useful evidence when possible.

Required:

- `pcq run --json` reports exit code and captured stdout/stderr paths
- `pcq run --jsonl` emits `run.failed` or `run.error`
- `pcq.save_all(status="failed", failure=...)` can preserve partial facts
- `describe-run` handles missing or partial artifacts without crashing
- `compare-runs` can explain incomparability

### No Prose Scraping

Every agent-critical command needs a machine-readable surface.

Required:

- JSON for snapshot commands
- JSONL for live streams
- stable schema versions
- explicit missing fields instead of implicit prose-only explanations

## Core Agent Journey

### 1. Discover

Agents should read:

- `README.md`
- `docs/V4_DIRECTION.md`
- `docs/AGENT_OPERATING_GUIDE.md`
- `site/llms.txt`
- `site/agent-manifest.json`

### 2. Inspect

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

The agent should identify:

- project root
- selected `cq.yaml`
- command
- config
- output directory
- metrics
- inputs
- artifacts
- previous evidence

### 3. Author Or Modify

The agent should modify project-local code only:

- `train.py`
- local modules
- `cq.yaml.configs`
- project-specific helper files

It should not add framework support to `pcq` internals for one experiment.

### 4. Run

```bash
pcq run --path . --jsonl
```

Use JSONL when progress matters. Use `--json` when only a final envelope is
needed.

### 5. Validate

```bash
pcq validate-run output --strictness 3 --json
```

The agent should treat a process exit code as incomplete evidence until
post-run validation is read.

### 6. Describe

```bash
pcq describe-run output --json
```

The agent should read `decision_facts`, best/last metrics, validation status,
artifact summary, source identity, input identity, and failure evidence.

### 7. Compare

```bash
pcq compare-runs parent_output candidate_output --json
pcq lineage candidate_output --json
```

The agent should use comparison facts to choose the next action. `pcq` should
not choose that action itself.

### 8. Iterate

```bash
pcq apply-plan experiment.plan.json --json
```

`apply-plan` should make bounded, reviewable changes. It should not become a
general code-writing agent.

## Good Authoring Pattern

```text
cq.yaml owns run intent
train.py owns framework-specific code
pcq owns evidence/control
agent owns policy
```

## Forbidden Patterns

| Pattern | Why it breaks operability | Better path |
|---|---|---|
| hard-coded `output/` paths | ignores `configs.output_dir` | `pcq.output_dir()` |
| undeclared metrics | strict validation cannot reason about the run | update `cq.yaml.metrics` |
| success judged by exit code only | artifacts may be missing or invalid | `validate-run` |
| parsing terminal prose | fragile agent behavior | JSON/JSONL |
| adding one-off adapters to `pcq` core | competes with the framework | contract script |
| hidden dataset downloads during inspect | read-side command becomes unsafe | explicit data prep/input declaration |
| framework-specific code inside pcq internals | scope leak | project-local code |

## Acceptance Criteria

`pcq` should claim agent-operable status only when:

- a fresh agent can discover the contract from docs/site files
- read-side commands are safe and machine-readable
- `pcq run --jsonl` gives live progress without prose scraping
- standard artifacts are produced or missing evidence is explicit
- strictness levels are documented and visible in reports
- `describe-run` exposes decision facts
- `compare-runs` exposes metric/config/source/artifact differences
- lineage is available for parent-child run chains
- failed runs can be represented as structured evidence
- project-local contract scripts can use arbitrary frameworks without adapters

## Final Identity

`agent-readable evidence contract` is the base.

`agent-operable experiment boundary` is the product.
