# JSON Contracts

`pcq` is an agent-operable experiment contract library. Its CLI JSON output is
therefore part of the public API: agents, CQ service workers, CI jobs, and other
orchestrators should be able to parse it without reading Python source.

This document defines the JSON contract freeze policy for the core
agent-facing surfaces.

## Contract Policy

Every machine-facing JSON payload must include:

- `schema_version`
- stable top-level status or identity fields for that command
- deterministic field types for required fields

For `schema_version: 1`, changes are **additive-only**:

- new optional fields may be added
- required fields may not be removed
- required field types may not change
- enum values may be extended only when existing consumers remain valid
- policy decisions such as "winner", "rollback", or "continue" must not be
  baked into pcq outputs

The machine-readable registry lives in
`pcq.agent.json_contracts.JSON_CONTRACTS`. Tests validate real command outputs
with `pcq.agent.validate_json_contract(...)`.

## Frozen Surfaces

### `pcq run --json`

Contract name: `pcq.run.envelope`

Required fields:

- `schema_version: int`
- `status: string`
- `project_root: string`
- `runtime_cfg_path: string`
- `cmd: string`

Stable statuses:

- `completed`
- `failed`
- `config_only`
- `error`

When a command is executed, the envelope also includes child process evidence:

- `exit_code`
- `stdout_path`
- `stderr_path`
- `stdout_tail`
- `stderr_tail`
- `stdout_tail_truncated`
- `stderr_tail_truncated`

Important boundary: stdout is pure JSON in `--json` mode. Child stdout/stderr
are captured into files and summarized in the envelope.

### `pcq run --jsonl`

Contract name: `pcq.run.event`

`pcq run --jsonl` emits newline-delimited JSON events while the child process is
running. Each line is independently parseable.

Required fields:

- `schema_version: int`
- `seq: int`
- `time: string`
- `event: string`

Stable event names:

- `run.started`
- `stdout`
- `stderr`
- `metric`
- `run.completed`
- `run.failed`
- `run.error`
- `run.config_only`

Metric events are derived from `pcq.log(...)` stdout lines in `@key=value`
format:

```json
{"schema_version":1,"seq":2,"event":"metric","metrics":{"epoch":1,"eval_acc":0.9}}
```

Use `pcq run --events output/events.jsonl --json` when an agent or service wants
a final JSON envelope on stdout and live event evidence persisted in a file.

### `pcq describe-run --json`

Contract name: `pcq.describe_run.record`

This is a single-run fact object. It is not a recommendation.

Required fields for a readable `run_record.json`:

- `schema_version`
- `status`
- `output_dir`
- `epochs_completed`
- `partial`
- `dirty`
- `validation_status`
- `decision_facts`

`decision_facts` is policy-free. It exposes facts such as:

- whether the run completed, failed, or is partial
- whether validation passed or failed
- whether target/best/last metrics exist
- artifact, metric, and input counts
- whether lockfile and cq.yaml hash evidence exists

Agents decide what to do with those facts.

#### `attribution` object in `pcq.describe_run.record`

The `attribution` object is an **optional** field in `pcq.describe_run.record`
(and in the underlying `run_record.json`). When present it conforms to this
nested shape:

```json
"attribution": {
  "schema_version": 1,
  "author": {
    "kind": "human" | "agent",
    "id": "<free string>",
    "persona_id": "<string>" | null
  },
  "committer": {
    "kind": "human" | "agent",
    "id": "<free string>",
    "persona_id": "<string>" | null
  },
  "operator": "<free string>",
  "session_id": "<string>" | null
}
```

When absent, readers must treat `attribution` as `null` — the absence is not a
contract violation. This preserves backward compatibility with run records
produced before attribution was introduced.

The `attribution` object is exposed verbatim by `pcq describe-run --json`; no
field is redacted or normalised at the format layer.

**Flat surface (T-PCQ-ATTR-2)**: A future task will expose the same data as
top-level flat keys (e.g. `attribution_operator`, `attribution_author_kind`) for
consumers that cannot parse nested objects. Until then, the nested form is the
only guaranteed surface.

**`signature` reserved name**: Within `attribution`, the key `signature` is
reserved for a Phase 2 cryptographic endorsement field. It is not part of the
current contract; parsers should tolerate its presence for forward-compatibility
but must not depend on it being absent.

#### `worker_spec` object in `pcq.describe_run.record`

The `worker_spec` object is an **optional** sibling of `attribution` in
`pcq.describe_run.record` (and in the underlying `run_record.json`). When
present it conforms to this nested shape:

```json
"worker_spec": {
  "schema_version": 1,
  "cpu": {
    "model": "<string> | null",
    "cores_physical": "<int> | null",
    "cores_logical": "<int> | null",
    "max_freq_mhz": "<number> | null"
  },
  "memory": {
    "total_gb": "<number> | null"
  },
  "accelerator": {
    "kind": "cuda | mps | cpu",
    "gpus": [
      {
        "model": "<string> | null",
        "vram_gb": "<number> | null",
        "cuda_version": "<string> | null",
        "bus_id": "<string> | null",
        "torch_ordinal": "<int> | null"
      }
    ]
  },
  "os": {
    "system": "<string>",
    "machine": "<string>",
    "release": "<string> | null"
  },
  "container": {
    "kind": "none | docker | k8s | other",
    "image": "<string> | null",
    "detector_hint": "<string> | null"
  },
  "source": "detected | declared | merged",
  "visible_devices": "<string> | null"
}
```

When absent, readers must treat `worker_spec` as `null` — the absence is not a
contract violation. This preserves backward compatibility with run records
produced before worker_spec was introduced.

**Flat surface**: `pcq describe-run --json` also exposes four top-level flat
fields for consumers that cannot parse nested objects:

| Flat field | Type | Source path |
|---|---|---|
| `worker_spec_cpu_model` | `string \| null` | `worker_spec.cpu.model` |
| `worker_spec_memory_gb` | `number \| null` | `worker_spec.memory.total_gb` |
| `worker_spec_accelerator_kind` | `string \| null` | `worker_spec.accelerator.kind` |
| `worker_spec_gpu_model_0` | `string \| null` | `worker_spec.accelerator.gpus[0].model` (null when gpus is empty) |

These four flat fields are present whenever `worker_spec` is present (even when
the nested field is partially null). When `worker_spec` itself is null or absent,
all four flat fields are also null.

**`pcq.compare_runs.diff` passthrough**: When worker_spec is present on both
runs being compared, `pcq compare-runs A B --json` includes
`worker_spec_changed: boolean` in its top-level `decision_facts` object — true
if any field in worker_spec differs between A and B.

**`pcq.run.envelope` passthrough**: `pcq run --json` carries `worker_spec` in
the envelope at the same path as `run_record.json`, populated at run-start time
from auto-detection or declared values.

#### `fingerprint` object in `pcq.describe_run.record`

The `fingerprint` object is an **optional** sibling of `attribution` and
`worker_spec` in `pcq.describe_run.record` (and in the underlying
`run_record.json`). When present it conforms to this nested shape:

```json
"fingerprint": {
  "schema_version": 1,
  "modality": "tabular" | "image" | "text" | "time_series" | "audio" | "graph" | "other",
  "task_kind": "classification" | "regression" | "segmentation" | "detection"
             | "seq2seq" | "generation" | "forecasting" | "anomaly_detection"
             | "clustering" | "other",
  "n_samples": "<int> | null",
  "size_class": "small" | "medium" | "large" | "huge",
  "domain": "general" | "medical" | "financial" | "regulated" | "other",
  "source": "detected" | "detected_sampled" | "declared" | "merged",
  "tabular": {
    "n_columns": "<int> | null",
    "type_counts": { "numeric": "<int>", "categorical": "<int>", "datetime": "<int>", "text": "<int>" },
    "target_balance": "<float> | null",
    "n_classes": "<int> | null",
    "missing_ratio_max": "<float> | null"
  },
  "image": {
    "input_shape": "<[int, int, int]> | null",
    "n_classes": "<int> | null"
  },
  "text": {
    "avg_token_len": "<int> | null",
    "vocab_kind": "english" | "korean" | "multilingual" | "code" | "other"
  },
  "time_series": {
    "seq_len": "<int> | null",
    "freq": "daily" | "hourly" | "irregular" | "other"
  },
  "audio": {
    "sample_rate": "<int> | null",
    "avg_duration_sec": "<float> | null"
  },
  "graph": {
    "n_nodes": "<int> | null",
    "n_edges": "<int> | null",
    "n_node_features": "<int> | null"
  },
  "modality_other": {
    "hint": "<string>",
    "payload": "<object>"
  }
}
```

Only the modality-specific sub-object matching `fingerprint.modality` is
expected to be populated. Other modality sub-objects may be absent or null.

When `modality = "other"`, `modality_other` **must** be present with at least
the `hint` field; `payload` may be an empty object `{}`.

When absent, readers must treat `fingerprint` as `null` — the absence is not a
contract violation. This preserves backward compatibility with run records
produced before fingerprint was introduced (R7).

**Flat surface**: `pcq describe-run --json` exposes four top-level flat fields
for consumers that cannot parse nested objects:

| Flat field | Type | Source path |
|---|---|---|
| `fingerprint_modality` | `string \| null` | `fingerprint.modality` |
| `fingerprint_task_kind` | `string \| null` | `fingerprint.task_kind` |
| `fingerprint_n_samples` | `int \| null` | `fingerprint.n_samples` |
| `fingerprint_size_class` | `string \| null` | `fingerprint.size_class` |

These four flat fields are present whenever `fingerprint` is present (even when
nested fields are partially null). When `fingerprint` itself is null or absent,
all four flat fields are also null.

**`pcq.compare_runs.diff` passthrough**: When `fingerprint` is present on both
runs being compared, `pcq compare-runs A B --json` includes
`fingerprint_changed: boolean` in its top-level `decision_facts` object — true
if any top-level field in `fingerprint` differs between A and B.

**`pcq.run.envelope` passthrough**: `pcq run --json` carries `fingerprint` in
the envelope at the same path as `run_record.json`, populated at run-start time
from any detected or declared values available before training begins.

**Conformance pairs (R9)**: Two golden pairs are required:

- `tests/conformance/pcq.describe_run.record/with-fingerprint-tabular/` —
  demonstrates a complete tabular fingerprint with all four flat fields.
- `tests/conformance/pcq.describe_run.record/without-fingerprint/` —
  demonstrates a valid record with `fingerprint: null` and all four flat fields
  null.

### `pcq compare-runs A B --json`

Contract name: `pcq.compare_runs.diff`

This is a pairwise comparison fact object. It is not a winner selector.

Required fields for two readable RunRecords:

- `schema_version`
- `a_run_id`
- `b_run_id`
- `target_metric`
- `a_target_metric`
- `b_target_metric`
- `mode`
- `metric_direction`
- `best`
- `last`
- `validation`
- `artifacts`
- `decision_facts`
- `a_status`
- `b_status`
- `a_is_ancestor_of_b`
- `b_is_ancestor_of_a`

Stable metric directions:

- `improved`
- `regressed`
- `tied`
- `incomparable`

`decision_facts.comparable` is true only when A and B have the same target
metric and the metric values can be compared. Services that want a "winner"
label should derive it from this evidence plus their own policy.

### `pcq validate --json` and `pcq validate-run --json`

Contract name: `pcq.validation_report`

Required fields:

- `schema_version`
- `status`
- `checks`
- `blocking_count`
- `warning_count`

Each check must include:

- `id`
- `status`
- `severity`
- `detail`

When strictness is selected or resolved, reports also include:

- `strictness`
- `strictness_name`

Validation reports are evidence objects. They should explain missing evidence
with structured checks and `suggested_fix` whenever possible.

### `pcq agent install --json`

Contract name: `pcq.agent_install.result`

This is the write-side agent runtime installation result.

Required fields:

- `schema_version`
- `project_root`
- `target`
- `dry_run`
- `force`
- `files_created`
- `files_updated`
- `files_skipped`
- `warnings`
- `operations`

Each operation includes:

- `path`
- `action`
- `kind`
- `agent`
- `reason`

Stable targets:

- `codex`
- `claude`
- `both`

### `pcq agent status --json`

Contract name: `pcq.agent_status.result`

This is the read-only agent runtime health result.

Required fields:

- `schema_version`
- `project_root`
- `target`
- `status`
- `assets`
- `repair_command`

Each asset includes:

- `path`
- `agent`
- `kind`
- `status`
- `detail`
- `suggested_fix`

Stable top-level statuses:

- `missing`
- `installed`
- `partial`
- `stale`
- `unmanaged`
- `divergent`

## Regression Gate

The JSON contract gate is:

```bash
uv run pytest tests/test_json_contracts.py
```

Before tagging, run the full suite as well:

```bash
uv run pytest -q
```
