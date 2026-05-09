# RunRecord Standard

## Summary

`pcq` should treat a run as complete only when it can produce a
machine-readable experiment record, not merely when the training process exits.

The target object is `RunRecord`:

```text
RunRecord =
  execution contract
  + source snapshot
  + environment snapshot
  + input identity
  + metric schema
  + artifact manifest
  + agent plan provenance
  + validation report
  + result summary
```

This is the difference between ordinary logs and agent-operable experiment
history. Logs help a human look back. A `RunRecord` lets an agent compare runs,
explain what changed, and create the next experiment.

## Cold Review

The current v1.13 direction is correct but still incomplete.

What exists now:

- `cq.yaml` describes the command, configs, declared metrics, and artifact globs.
- `pcq.log()` emits metric lines in a stable stdout format.
- `Experiment.fit()` writes standard output artifacts.
- contract scripts can call `pcq.save_all(...)` to write `config.json`,
  `metrics.json`, `run_summary.json`, and `manifest.json`.
- project-local atoms and `ExperimentPlan` give agents bounded places to edit.

What is still missing for strong reproducibility:

- source snapshot or patch identity
- dependency lockfile identity
- dataset/input identity
- structured metric semantics
- artifact checksums and sizes
- post-run validation report
- one canonical `run_record.json`

Therefore v1.13 should be described as the first contract-artifact layer, not as
full reproducibility. Full reproducibility starts when `run_record.json` becomes
the primary completion artifact.

## Output Layout

Recommended project shape:

```text
project/
  cq.yaml
  train.py
  cq_atoms.py
  atoms/
    models.py
    losses.py
  output/
    config.json
    metrics.json
    manifest.json
    run_summary.json
    run_record.json        # target standard
    validation_report.json # target standard
    model.pt
    last.ckpt
    best.ckpt
```

`model.pt`, `last.ckpt`, and `best.ckpt` are framework-specific artifacts.
`config.json`, `metrics.json`, `manifest.json`, `run_summary.json`, and
`run_record.json` are contract artifacts.

## cq.yaml Contract

Minimal current form:

```yaml
name: dental-seg-unet-v004
cmd: uv run python train.py

configs:
  seed: 42
  output_dir: output
  epochs: 50
  batch_size: 8
  lr: 0.0003
  monitor: eval_iou
  mode: max

metrics:
  - epoch
  - train_loss
  - train_iou
  - eval_loss
  - eval_iou

artifacts:
  - output/
```

Target structured form:

```yaml
name: dental-seg-unet-v004
cmd: uv run python train.py

configs:
  seed: 42
  output_dir: output
  epochs: 50
  batch_size: 8
  lr: 0.0003

inputs:
  dataset:
    name: dental
    version: v12
    uri: cq://datasets/dental/v12
    split: train-val-2026-05-01
    sha256: "..."

metrics:
  eval_iou:
    mode: max
    split: val
    aggregation: macro
    sample_count: 1240
  eval_loss:
    mode: min
    split: val
    aggregation: mean

artifacts:
  output/:
    kind: run_output
```

Compatibility rule: v1.x should keep accepting the list-style `metrics` and
`artifacts` forms. Structured forms add semantics for agents and CQ service.

## Contract Script API

Any ML framework can produce CQ-compatible artifacts by following the contract.
No framework adapter is required.

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

# Use any ML library here:
# HF Trainer, TabPFN, PyCaret, sklearn, XGBoost, custom code.
score = 0.74

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

`pcq.save_all(...)` currently writes:

- `config.json`
- `metrics.json`
- `run_summary.json`
- `manifest.json`

These files are RunRecord components. They are not yet a full RunRecord.

## Artifact Manifest

Current v1.13 manifest shape:

```json
{
  "schema_version": 1,
  "files": [
    {"path": "metrics.json", "kind": "metrics"},
    {"path": "config.json", "kind": "config"},
    {"path": "model.pt", "kind": "model"}
  ]
}
```

Target evidence manifest:

```json
{
  "schema_version": 2,
  "files": [
    {
      "path": "model.pt",
      "kind": "model",
      "sha256": "...",
      "size_bytes": 18423920,
      "created_at": "2026-05-05T10:58:00Z"
    }
  ]
}
```

The target manifest is not just a file list. It is evidence that the artifact
collected later is the artifact produced by the run.

## Metric Schema

Metric values alone are not enough for agent decisions. `eval_iou=0.74` does not
say which split, aggregation, ignore index, or sample count was used.

Target metric declaration:

```json
{
  "name": "eval_iou",
  "mode": "max",
  "split": "val",
  "aggregation": "macro",
  "sample_count": 1240,
  "higher_is_better": true
}
```

Rules:

- every target metric must declare `mode`
- split and aggregation should be declared when known
- sample count should be recorded at run finalization when available
- `run_summary.json.best` must refer to a declared target metric

## RunRecord Schema

Target `output/run_record.json`:

```json
{
  "schema_version": 1,
  "run": {
    "id": "run_20260505_001",
    "name": "dental-seg-unet-v004",
    "status": "completed",
    "started_at": "2026-05-05T10:12:00Z",
    "finished_at": "2026-05-05T10:58:00Z"
  },
  "execution": {
    "cmd": "uv run python train.py",
    "cwd": ".",
    "config_path": "cq.yaml"
  },
  "source": {
    "git_sha": "8bb2ec07",
    "dirty": false,
    "patch_sha256": null,
    "changed_files": [],
    "cq_yaml_path": "cq.yaml",
    "cq_yaml_sha256": "..."
  },
  "environment": {
    "python": "3.11.8",
    "platform": "linux-x86_64",
    "pcq_version": "2.8.0",
    "torch_version": "2.9.0",
    "cuda_available": false,
    "device": "cpu",
    "lockfile": "uv.lock",
    "lockfile_sha256": "..."
  },
  "config": {
    "cq_yaml_path": "cq.yaml",
    "cq_yaml_sha256": "...",
    "config_json_path": "config.json",
    "config_json_sha256": "...",
    "seed": 42,
    "strictness": 3,
    "output_dir": "output"
  },
  "inputs": {
    "dataset": {
      "name": "dental",
      "version": "v12",
      "uri": "cq://datasets/dental/v12",
      "split": "train-val-2026-05-01",
      "sha256": "..."
    }
  },
  "input_summary": {
    "count": 1,
    "names": ["dataset"],
    "identity": {
      "dataset": {
        "has_uri": true,
        "has_path": false,
        "has_sha256": true,
        "has_manifest": false,
        "opaque": false
      }
    }
  },
  "metrics": {
    "declared": [
      {
        "name": "eval_iou",
        "mode": "max",
        "split": "val",
        "aggregation": "macro"
      }
    ],
    "history_path": "metrics.json"
  },
  "artifacts": [
    {
      "path": "model.pt",
      "kind": "model",
      "sha256": "...",
      "size_bytes": 18423920
    }
  ],
  "summary": {
    "target_metric": "eval_iou",
    "best": {"epoch": 37, "eval_iou": 0.742}
  },
  "agent": {
    "plan_id": "exp-004",
    "intent": "Increase Dice contribution for boundary quality",
    "approval_status": "approved"
  },
  "validation": {
    "status": "pass",
    "report_path": "validation_report.json"
  }
}
```

### Streaming Partial RunRecord (v2.11)

While training is in progress, callers may write a partial `run_record.json`
via `pcq.save_partial_run_record(history=...)`. The library writes the file via
tmp + `os.replace`, so concurrent readers always see fully-written JSON.

Two extra fields appear on `run` while partial:

```json
{
  "run": {
    "status": "running",
    "partial": true,
    "last_updated_at": "2026-05-08T11:42:13Z"
  }
}
```

After `pcq.finalize_run()` runs, the same file is rewritten with `partial`
removed (default false) and `status` set to `completed` / `failed` / `partial`
(the legacy "incomplete" status). `last_updated_at` is refreshed to the
finalize timestamp.

`pcq` is the *system* layer here: it captures time evidence. Reading a
partial record and deciding what to do (interpret trajectory shape, recommend
the next plan, kill a diverging run) is the *agent's* responsibility — kept
out of the library on purpose.

### Structured Failure Envelope (v2.11)

`run_summary.json.failure` and any external consumer of failure information
should treat the field as a `FailureInfo` envelope:

```json
{
  "failure": {
    "error_code": "ERR_OUT_OF_MEMORY",
    "category": "oom",
    "message": "CUDA out of memory at batch 17",
    "evidence": {"batch_size": 32, "free_gb": 0.2},
    "suggested_fix": "reduce batch size to 16"
  }
}
```

- `error_code` is machine-readable (one of `ERR_MISSING_DEPENDENCY`,
  `ERR_INVALID_CONFIG`, `ERR_DATASET_UNAVAILABLE`, `ERR_OUT_OF_MEMORY`,
  `ERR_TIMEOUT`, `ERR_RUNTIME`).
- `category` is the legacy free-form string and stays for backward
  compatibility — older RunRecords with only `category` continue to load and
  the matching `error_code` is derived.
- `evidence` is structured key/value (pcq fills it on automatic
  classification; agents may add fields).
- `suggested_fix` is **natural language**, intended for the agent to read and
  reason about. pcq does not turn it into commands. That mapping is policy
  and lives outside the library.

## Validation Gates

Pre-run validation should check:

- `cq.yaml` parses
- command exists
- output directory policy is clear
- declared metric schema is valid
- dataset/input identity is present or explicitly unknown
- source state can be recorded
- dependency lockfile can be identified or warning is emitted
- `ExperimentPlan` is valid when present

Post-run validation should check:

- declared metrics were emitted
- target metric exists in history
- standard contract artifacts exist
- manifest entries point to real files
- checksums and sizes are recorded when enabled
- `run_summary.json` agrees with `metrics.json`
- `run_record.json` is complete

Managed CQ service should treat a run as complete only after post-run
validation passes or records a structured failure.

## Multi-Run Schema (v2.11)

`ExperimentPlanSet` lets agents express a *set* of related plans (fork, grid,
sweep) without putting policy into pcq:

```json
{
  "schema_version": 1,
  "id": "sweep-001",
  "intent": "lr × wd grid",
  "base": {"preset": "vision/fake_smoke"},
  "parent_run_id": "run_baseline_abc",
  "plans": [
    {"id": "exp-000", "changes": [...]},
    {"id": "exp-001", "changes": [...]}
  ]
}
```

`pcq apply-planset path.json --output-pattern "runs/exp{i}"` expands the set
into N output directories with `parent_run_id` propagated. The policy that
*chose* the plans (random / grid / Bayesian / agent LLM) is outside pcq.

## Responsibility Split

`pcq` owns:

- schema definitions (RunRecord, ExperimentPlan, ExperimentPlanSet,
  FailureInfo)
- local config loading
- metric emission helpers
- contract artifact helpers (including streaming partial RunRecord)
- local validation
- project-local atom contract
- `ExperimentPlan` / `ExperimentPlanSet` format
- `run_record.json` generation

CQ service owns:

- remote execution
- GPU queue and scheduling
- source snapshot storage
- dataset and artifact storage
- secret management
- approval and permission flow
- dashboard, comparison, and reports
- cost control
- multi-run agent orchestration

## Release Path

Recommended sequence:

1. v1.13: ship contract artifact helpers and script-style scaffolding.
2. v1.14: add checksums and sizes to `manifest.json`.
3. v1.15: add structured `inputs` and metric schema support in `cq.yaml`.
4. v1.16: add `pcq finalize` and `pcq validate-run`.
5. v2: make `run_record.json` the primary CQ service run object.

The product line should stay: framework freedom, strict run records.
