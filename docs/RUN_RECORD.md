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

### Metadata Fields: `attribution`, `worker_spec`, `fingerprint` (v4.4–4.6)

Three optional sibling fields extend the RunRecord schema with provenance,
hardware, and dataset-shape evidence. All three are additive: existing records
without them are valid.

#### `attribution` — who authored and operated the run

```json
"attribution": {
  "schema_version": 1,
  "author":    { "kind": "human" | "agent", "id": "...", "persona_id": "..." | null },
  "committer": { "kind": "human" | "agent", "id": "...", "persona_id": "..." | null },
  "operator":  "...",
  "session_id": "..." | null
}
```

| Field | Comes from | Meaning |
|---|---|---|
| `author` | `CQ_ATTRIBUTION_AUTHOR_*` / `cq.yaml.attribution.*` / auto (git user) | Who originated the experiment intent |
| `committer` | `CQ_ATTRIBUTION_COMMITTER_*` / auto | Who built and submitted the job (may be an AI agent) |
| `operator` | `CQ_ATTRIBUTION_OPERATOR` / `cq.yaml.attribution.operator` | Human or organisation bearing legal/reputational responsibility |
| `session_id` | `CQ_ATTRIBUTION_SESSION_ID` / CLI flag | Conversation or session trace handle (optional) |

Resolution precedence: `CLI flags > CQ_ATTRIBUTION_* env vars > cq.yaml attribution.* > auto-infer > NULL`.

**PII**: `operator` and `id` are free strings and may contain real names or email addresses. Use a pseudonym or UUID in any environment where records may be shared externally.

#### `worker_spec` — where the run executed

```json
"worker_spec": {
  "schema_version": 1,
  "cpu":  { "model": "...", "cores_physical": null, "cores_logical": null, "max_freq_mhz": null },
  "memory": { "total_gb": null },
  "accelerator": { "kind": "cuda | mps | cpu", "gpus": [{ "model": "...", "vram_gb": null, "cuda_version": null, "bus_id": null, "torch_ordinal": null }] },
  "os": { "system": "...", "machine": "...", "release": null },
  "container": { "kind": "none | docker | k8s | other", "image": null, "detector_hint": null },
  "source": "detected | declared | merged",
  "visible_devices": null
}
```

| Field group | Comes from |
|---|---|
| `cpu.*`, `memory.*` | Auto-detected via psutil; overridable by `CQ_WORKER_CPU_*` / `CQ_WORKER_MEMORY_*` |
| `accelerator.*`, `gpus[*].*` | Auto-detected via PyTorch/NVML; overridable by `CQ_WORKER_ACCELERATOR_KIND`, `CQ_WORKER_GPU_*` |
| `os.*` | Auto-detected via `platform`; overridable by `CQ_WORKER_OS_*` |
| `container.*` | Heuristic detection (`.dockerenv`, env vars); overridable by `CQ_WORKER_CONTAINER_KIND` |
| `source` | Set to `detected`, `declared`, or `merged` based on which resolution paths contributed |

`pcq describe-run --json` also exposes four top-level flat fields: `worker_spec_cpu_model`, `worker_spec_memory_gb`, `worker_spec_accelerator_kind`, `worker_spec_gpu_model_0`.

**PII policy (two layers)**:
- **R10 — auto-detection prohibition**: hostname, IP, MAC address, and login name are never emitted by auto-detection code.
- **R14 — declared path warning**: when fields are supplied via `CQ_WORKER_*` or `cq.yaml.worker.*`, pcq inspects free strings for hostname-like patterns and adds `WORKER_DECLARED_PII_LIKE` (severity L3) to `validation_report.json`.

#### `fingerprint` — what kind of data was used

```json
"fingerprint": {
  "schema_version": 1,
  "modality":   "tabular | image | text | time_series | audio | graph | other",
  "task_kind":  "classification | regression | segmentation | detection | seq2seq | generation | forecasting | anomaly_detection | clustering | other",
  "n_samples":  50000,
  "size_class": "small | medium | large | huge",
  "domain":     "general | medical | financial | regulated | other",
  "source":     "detected | detected_sampled | declared | merged"
}
```

| Field | Comes from |
|---|---|
| `modality`, `task_kind` | `pcq.fingerprint(X, y, modality=..., task_kind=...)` API argument, or `cq.yaml.fingerprint.modality` |
| `n_samples`, `size_class` | Auto-counted from dataset shape; declared fallback via `cq.yaml.fingerprint.n_samples` |
| `domain` | `cq.yaml.fingerprint.domain` or `CQ_FINGERPRINT_*`; defaults to `"general"` |
| `tabular.*`, `image.*`, etc. | Auto-extracted statistics (type counts, shape, target balance) by `pcq.fingerprint()` |
| `source` | `detected` / `detected_sampled` / `declared` / `merged` |

`pcq describe-run --json` exposes four flat fields: `fingerprint_modality`, `fingerprint_task_kind`, `fingerprint_n_samples`, `fingerprint_size_class`.

**PII 4-layer policy**:

| Layer | Rule | Effect |
|---|---|---|
| R10 | Auto-detection format prohibition | Column names, raw values, top-N frequencies are never emitted |
| R5 | Domain gate | `medical`, `financial`, `regulated` domains disable auto-detection; only declared values accepted |
| R5b | Heuristic sniffer | Column names checked against medical/financial keywords even at `domain = "general"`; emits L2 warning (no matched names in output) |
| R14 | Declared path PII warning | Free-string declared fields inspected for hostname, email, SSN-shape patterns; emits `FINGERPRINT_DECLARED_PII_LIKE` (L3) |

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

## Result Semantics For Agents

`failure.category` is a regex-based heuristic on `failure.message`. Free
strings are allowed when no category matches.

| Category | Retry / Abort hint |
|---|---|
| `config_error` | Abort — fix `cq.yaml` first; retry will fail identically |
| `missing_dependency` | Abort — run `uv add <package>` then retry |
| `dataset_missing` | Abort — verify input URIs or paths, then retry |
| `dataset_shape` | Abort — check tensor dimensions; fix code before retry |
| `label_contract` | Abort — check label range / dtype; fix code before retry |
| `loss_contract` | Abort — check loss function signature; fix before retry |
| `metric_contract` | Abort — declare the metric in `cq.yaml.metrics`, then retry |
| `oom` | Retry with smaller `batch_size` (halve); abort if already at minimum |
| `nan_loss` | Retry with lower `lr` or add gradient clipping; abort after 2 retries |
| `timeout` | Retry with larger `time_budget`; abort if resource limits are firm |
| `distributed_write_race` | Retry with reduced concurrent writers; abort if architecture issue |
| `accuracy_below_threshold` | Retry with tuned hyperparameters (smaller `lr`, longer training); abort after budget exhausted |
| `user_interrupted` | Respect the interruption — do not auto-retry |
| `disk_full` | Abort — free disk space then retry; auto-retry is unsafe |
| `model_load_failed` | Retry once after re-downloading or verifying checkpoint integrity; abort if hash mismatch persists |
| `unknown_exception` | Manual investigation required before retry |

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
