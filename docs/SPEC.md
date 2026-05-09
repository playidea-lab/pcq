# pcq Spec

## Summary

`pcq` is a framework-neutral experiment evidence and control library.

It is CQ-compatible, but not CQ-only. It standardizes the boundary around a run:
configuration, metric emission, artifact layout, validation, run records,
comparison, lineage, and iteration. It does not replace PyTorch, Hugging Face
Trainer, Lightning, sklearn, XGBoost, TabPFN, PyCaret, W&B, MLflow, DVC, or CQ.

The package is distributed on PyPI as `pcq`:

```bash
uv add pcq
```

```python
import pcq
```

See [pcq v4 Direction](V4_DIRECTION.md) for the identity decision that guides
this spec.

## Product Boundary

`pcq` owns:

- resolving the run contract from `cq.yaml` and runtime environment
- loading config for project-local code
- resolving output directories
- emitting and capturing metrics
- finalizing standard artifacts
- validating evidence
- describing runs as machine-readable facts
- comparing runs as machine-readable facts
- preserving lineage
- applying structured next-run plans
- exposing agent-readable docs and JSON/JSONL surfaces

`pcq` does not own:

- the training loop
- model architecture catalogs
- loss/optimizer/scheduler catalogs
- framework adapters as a required integration mechanism
- cloud orchestration credentials
- queueing or GPU scheduling
- dashboard policy
- deciding whether a run is good enough

The contract is the adapter.

## Runtime Contract

The runtime contract is intentionally small.

```text
Project file:
  cq.yaml

Config input:
  CQ_CONFIG_JSON -> path to normalized inline configs JSON
  fallback -> cq.yaml.configs for direct local runs when supported

Metric output:
  stdout line -> "@key=value @other=value"

Artifact output:
  files under resolved output_dir
  cq.yaml.artifacts globs are collected by the consumer after process exit

Completion:
  process exit code
  standard artifacts
  run_record.json
  validation_report.json
```

`pcq` must not introduce a second hidden runtime protocol.

## `cq.yaml`

Minimum:

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

Rules:

- top-level `cmd` is the command to execute
- `configs` is the runtime config visible to training code
- `metrics` declares numeric metric keys that may be emitted
- `inputs` records input identity; `cq://` URIs are opaque strings to `pcq`
- `artifacts` declares what should be collected
- `configs.output_dir` is the single source for output location

## Python API

### `pcq.config()`

Reads runtime config.

Required behavior:

- returns a JSON-safe dict
- reads `CQ_CONFIG_JSON` when present
- may fall back to `cq.yaml.configs` for direct local runs
- preserves declared metrics and top-level execution context when available
- fails with a clear message when no config source can be resolved

### `pcq.output_dir()`

Resolves the configured output directory.

Required behavior:

- honors `configs.output_dir`
- resolves relative paths against the project root, not arbitrary cwd
- creates the directory only on write-side paths
- matches the resolver used by `inspect`, `validate`, `finalize`, and
  `validate-run`

### `pcq.log(...)`

Emits numeric metrics in the standard stdout format.

Required behavior:

- prints finite numeric values as `@key=value`
- skips or rejects non-numeric values according to strictness
- warns or fails on undeclared metrics according to strictness
- works without importing training frameworks

### `pcq.save_all(...)`

Writes the standard artifact set from a script-style run.

Required behavior:

- writes `config.json`
- writes `metrics.json`
- writes `run_summary.json`
- writes `manifest.json`
- finalizes `run_record.json` by default
- writes or updates `validation_report.json` when post-run validation is invoked
- accepts structured failure evidence for failed or partial runs

### `pcq.finalize_run(...)`

Converges existing output artifacts into the canonical run record.

Required behavior:

- reads the same resolved config and project root as other commands
- records command, source, environment, inputs, metrics, artifacts, and summary
- records missing evidence explicitly instead of silently omitting it

## CLI Surface

Agent-facing commands must have stable JSON or JSONL output.

### Read Side

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

Rules:

- no training
- no dataset downloads
- no heavy optional framework imports by default
- no output directory creation from read-only inspection
- structured warnings instead of only prose

### Run Side

```bash
pcq run --path . --json
pcq run --path . --jsonl
pcq run --path . --events output/events.jsonl --json
```

Rules:

- `--json` emits one final envelope on stdout
- `--jsonl` emits newline-delimited live events
- `--events PATH --json` writes live JSONL evidence to a file while preserving
  final JSON stdout
- child stdout/stderr are captured and exposed through paths, tails, or events
- exit code is represented in the envelope and forwarded by the process

### Post-Run Side

```bash
pcq validate-run output --strictness 3 --json
pcq describe-run output --json
pcq compare-runs old_output new_output --json
pcq lineage output --json
```

Rules:

- commands operate from output artifacts and run records
- commands do not import training code
- `describe-run` reports facts, not policy
- `compare-runs` reports facts, not policy
- failure and partial runs are first-class records

### Write Side

```bash
pcq init-experiment --style script --output ./exp --with-pyproject
pcq apply-plan experiment.plan.json --json
pcq agent install --target codex --path .
pcq agent status --target both --path . --json
```

Rules:

- writes must be explicit
- generated projects should default to contract scripts
- package installation must not modify agent runtime files
- `apply-plan` should be bounded, reviewable, and idempotent where possible

## JSONL Run Events

`pcq run --jsonl` emits live event objects.

Required event types:

- `run.started`
- `stdout`
- `stderr`
- `metric`
- `run.completed`
- `run.failed`
- `run.error`
- `run.config_only`

Each event should include at least:

- `schema_version`
- `seq`
- `time`
- `event`

Metric events are derived from `pcq.log(...)` stdout lines.

## Standard Artifacts

A complete output directory should contain:

- `config.json`
- `metrics.json`
- `manifest.json`
- `run_summary.json`
- `run_record.json`
- `validation_report.json`

Optional artifacts include model files, checkpoints, framework logs, plots, and
external tracker identifiers.

## RunRecord

`run_record.json` is the canonical completion object.

It should include:

- run identity and status
- execution command and config path
- source identity, including git sha and dirty state when available
- environment identity, including Python/platform and lockfile hash when
  available
- input identity
- declared metrics and metric history path
- artifact manifest entries with hashes when available
- summary best/last metric facts
- validation status and report path
- agent provenance when present
- parent run identity when present
- structured failure evidence when present

See [RunRecord Standard](RUN_RECORD.md).

## Strictness

Strictness levels define how much evidence is required.

The default should be usable for local development. Higher levels should be used
for CI, service workers, and automatic agent loops.

Validation output must always report:

- selected strictness level
- required evidence for that level
- present evidence
- missing evidence
- blocking failures
- warnings

See [Strictness Evidence Matrix](STRICTNESS.md).

## Framework Neutrality

The default integration path is a contract script:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# Any ML framework or custom code.

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

Rules:

- no official adapter is required
- project-local code owns framework-specific details
- examples may demonstrate common patterns
- examples must not become a support matrix
- base import must remain lightweight

## Non-Goals

- reimplementing Lightning, HF Trainer, PyCaret, W&B, MLflow, DVC, or CQ
- automatic hyperparameter sweep management as a core concern
- a built-in production model/loss/dataset catalog
- hidden network activity during inspect/validate
- CQ Hub/Drive clients in the core package
- a policy engine for research decisions

## Completion Criteria

From a fresh agent session, `pcq` is complete enough when the agent can:

1. discover the contract from `llms.txt` or repository docs
2. create a project-local contract script around any ML framework
3. run it with `pcq run --jsonl`
4. validate its output with `validate-run`
5. read decision facts with `describe-run`
6. compare against a parent run
7. preserve lineage
8. apply a structured next-run plan
9. repeat without editing `pcq` internals or scraping prose

## Assumptions

- package distribution, import, CLI, repository, and JSON namespace are `pcq`
- `cq.yaml`, `CQ_CONFIG_JSON`, and `cq://` remain runtime contract names
- CQ service consumes the contract but is not required
- v3 Trainer/catalog concepts are not v4 product identity
