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

## Attribution

Every `run_record.json` may carry an `attribution` object that records *who
authored the intent* and *who executed the run*. This is the machine-readable
foundation of the TheCommons "Wikipedia + bots" accountability model: human
contributors and AI agents appear on the same surface with separate roles.

### Why `attribution`, not `agent`

The top-level `agent` key in `run_record.json` is already reserved for the
CQ orchestration context — `{plan_id, intent, recipe, overrides}`. Putting
provenance data there would conflate two concerns. `attribution` is a
sibling key with its own schema.

### Schema (schema_version: 1, additive)

```json
"attribution": {
  "schema_version": 1,
  "author":    { "kind": "human" | "agent", "id": "...", "persona_id": "..." | null },
  "committer": { "kind": "human" | "agent", "id": "...", "persona_id": "..." | null },
  "operator":  "...",
  "session_id": "..." | null
}
```

Field semantics follow the Git author/committer convention:

| Field | Meaning |
|---|---|
| `author` | Who originated the intent ("I want to run this experiment") |
| `committer` | Who built and submitted the job (may be an AI agent) |
| `operator` | The human or organisation that bears legal and reputational responsibility — always a person/org, never an agent ID |
| `session_id` | Conversation or session trace handle (optional, aids audit) |

When a single human runs `pcq` directly without an agent in the loop,
`author`, `committer`, and `operator` all identify the same person.
When an AI agent (e.g. Claude Code) executes the run on behalf of a human,
`committer.kind = "agent"` and `operator` holds the human/org identity.

The `attribution` field is **optional** on `run_record.json`. Existing records
without it are valid; readers must treat absence as `null` (backward-compatible).

### `signature` — Phase 2 reserved field name

`signature` is a reserved name within `attribution` for a future cryptographic
endorsement of the record (e.g. operator key sign-off). It is **not part of the
Phase 1 schema** — do not emit or validate it now. Parsers must tolerate its
presence for forward-compatibility. The algorithm and key-management design will
be specified in a separate Phase 2 decision record.

### Resolution priority

When pcq resolves the `attribution` fields at write time, it applies this
precedence (first match wins):

```
CLI flags  >  CQ_ATTRIBUTION_* env vars  >  cq.yaml attribution.*  >  auto-infer  >  NULL
```

Supported environment variables:

| Variable | Field populated |
|---|---|
| `CQ_ATTRIBUTION_OPERATOR` | `attribution.operator` |
| `CQ_ATTRIBUTION_AUTHOR_ID` | `attribution.author.id` |
| `CQ_ATTRIBUTION_AUTHOR_KIND` | `attribution.author.kind` |
| `CQ_ATTRIBUTION_COMMITTER_ID` | `attribution.committer.id` |
| `CQ_ATTRIBUTION_COMMITTER_KIND` | `attribution.committer.kind` |
| `CQ_ATTRIBUTION_SESSION_ID` | `attribution.session_id` |
| `CQ_ATTRIBUTION_PERSONA_AUTHOR` | `attribution.author.persona_id` |
| `CQ_ATTRIBUTION_PERSONA_COMMITTER` | `attribution.committer.persona_id` |

Auto-inference (lowest priority): when git user config is available and no
explicit value is set, pcq may populate `author.id` and `committer.id` from
`git config user.email` or equivalent. This is a best-effort hint only.

### PII guidance

`attribution.operator` and the `id` fields are free strings. They **may**
contain personally-identifiable information (PII) such as email addresses or
real names. pcq does not redact these fields at the format layer.

Recommended practice:
- Use a **pseudonym or UUID** for `operator`, `author.id`, and `committer.id`
  in any environment where run records may be shared externally or ingested
  into TheCommons.
- Do **not** use email addresses or real names as the primary identifier in
  `operator`. An opaque organisation handle (e.g. `"pilab"`) or a UUID is
  preferred.
- If `git config user.email` is auto-inferred, be aware that CI environments
  may expose real email addresses through this path.

PII handling policy for stored or published records is the responsibility of
the consuming system (CQ Hub, TheCommons, CI), not pcq.

## Worker Spec

Every `run_record.json` may carry a `worker_spec` object that records *on which
machine the run was executed*. This is the second pillar of TheCommons
matchmaking input — after `attribution` (who) comes `worker_spec` (where).

### Why `worker_spec`, not inline fields

The top-level `attribution` key records the human/agent accountable for the run.
`worker_spec` is a sibling key that records the hardware environment. Keeping
them separate prevents the attribution object from becoming a god object, and
mirrors the author/committer pattern of Git: different concerns get different
namespaces.

### Schema (schema_version: 1, additive)

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

The `worker_spec` field is **optional** on `run_record.json`. Existing records
without it are valid; readers must treat absence as `null` (backward-compatible,
R7).

### Flat surface (4 fields, attribution pattern)

`pcq describe-run --json` exposes worker_spec both as the nested object above
and as four top-level flat keys (R6):

| Flat field | Source path |
|---|---|
| `worker_spec_cpu_model` | `worker_spec.cpu.model` |
| `worker_spec_memory_gb` | `worker_spec.memory.total_gb` |
| `worker_spec_accelerator_kind` | `worker_spec.accelerator.kind` |
| `worker_spec_gpu_model_0` | `worker_spec.accelerator.gpus[0].model` (null when empty) |

### Environment variables (CQ_WORKER_*)

Resolution precedence (first match wins):

```
CLI flag  >  CQ_WORKER_* env var  >  cq.yaml worker.*  >  auto-detect  >  NULL
```

| Variable | Populated field |
|---|---|
| `CQ_WORKER_CPU_MODEL` | `worker_spec.cpu.model` |
| `CQ_WORKER_CPU_CORES_PHYSICAL` | `worker_spec.cpu.cores_physical` |
| `CQ_WORKER_CPU_CORES_LOGICAL` | `worker_spec.cpu.cores_logical` |
| `CQ_WORKER_CPU_MAX_FREQ_MHZ` | `worker_spec.cpu.max_freq_mhz` |
| `CQ_WORKER_MEMORY_TOTAL_GB` | `worker_spec.memory.total_gb` |
| `CQ_WORKER_ACCELERATOR_KIND` | `worker_spec.accelerator.kind` |
| `CQ_WORKER_GPU_MODEL_0` | `worker_spec.accelerator.gpus[0].model` |
| `CQ_WORKER_GPU_VRAM_GB_0` | `worker_spec.accelerator.gpus[0].vram_gb` |
| `CQ_WORKER_GPU_CUDA_VERSION` | `worker_spec.accelerator.gpus[0].cuda_version` |
| `CQ_WORKER_OS_SYSTEM` | `worker_spec.os.system` |
| `CQ_WORKER_OS_MACHINE` | `worker_spec.os.machine` |
| `CQ_WORKER_OS_RELEASE` | `worker_spec.os.release` |
| `CQ_WORKER_CONTAINER_KIND` | `worker_spec.container.kind` |

When a `CQ_WORKER_*` variable overrides an auto-detected field, `source`
becomes `"merged"`. When all fields come from env vars / cfg, `source` is
`"declared"`.

### `source` audit values

| Value | Meaning |
|---|---|
| `detected` | All populated fields came from auto-detection (psutil + torch) |
| `declared` | All populated fields came from cfg / env vars (user-supplied) |
| `merged` | Some fields auto-detected, some overridden by cfg / env vars |

### PII layered policy

Worker spec applies a two-layer PII barrier:

**R10 — Auto-detection emit prohibition (format layer)**:
Auto-detection code must NEVER emit hostname, IP address, MAC address, or
user/login name into any `worker_spec` field. These are forbidden at the
code path that builds the `detected` or `merged` record, not only at
validation time. The format is a hard gate.

**R14 — Declared path PII warning (validation layer)**:
The auto-detection gate does not apply to `declared` or `merged` fields
supplied by the user (via `CQ_WORKER_*` or `cq.yaml.worker.*`). However,
when pcq writes `validation_report.json`, it inspects every free-string
`worker_spec` field (cpu.model, os.release, container.image,
container.detector_hint) for patterns that resemble a hostname
(letters/digits/hyphens + dots without spaces, matching `\b[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+\b`).
If a match is found, pcq adds a severity-3 (L3) warning:

```
code: WORKER_DECLARED_PII_LIKE
detail: "worker_spec.<field> may contain a hostname or FQDN — review before publishing"
```

Redaction is the consumer's responsibility. pcq never strips or hashes
worker_spec fields.

### Container kind enum

`worker_spec.container.kind` is a closed enum:

| Value | When to use |
|---|---|
| `none` | Bare-metal or VM — no container layer detected |
| `docker` | `/.dockerenv` present or `DOCKER_CONTAINER=1` in env |
| `k8s` | `KUBERNETES_SERVICE_HOST` in env |
| `other` | Any other containerisation detected or suspected (Podman, LXC, WSL, systemd-nspawn, Singularity, etc.) |

When multiple signals conflict, pcq chooses `other` and sets
`container.detector_hint` to the primary signal (e.g. `"podman-rootless"`).
`WORKER_CONTAINER_AMBIGUOUS` is emitted in `validation_report.json` when
two or more detection signals disagree.

### gpus[] deterministic ordering (R13)

The `gpus` array is ordered by NVML PCI bus_id ascending (e.g.
`"00000000:01:00.0"` before `"00000000:02:00.0"`). When NVML is unavailable,
the torch device ordinal (`torch.cuda.get_device_properties(i)`) is used as
a fallback ordering key.

Each GPU entry carries:

- `bus_id` — NVML PCI bus_id string when available, else null
- `torch_ordinal` — integer index as seen by PyTorch (`CUDA_VISIBLE_DEVICES`-aware)
- `visible_devices` — string copy of `CUDA_VISIBLE_DEVICES` at detection time
  (null if the variable is unset)

When `CUDA_VISIBLE_DEVICES` remaps ordinals, pcq records both the logical
ordinal and the raw bus_id to enable audit. A consumer that needs the physical
device should use `bus_id`; one that needs the PyTorch ordinal should use
`torch_ordinal`.

### Warning codes (R11 / R12 / R14)

All warning codes appear in `validation_report.json` under the `checks` array:

| Code | Severity | Condition |
|---|---|---|
| `WORKER_PSUTIL_MISSING` | L2 | `import psutil` fails — cpu/memory fields will be null |
| `WORKER_PSUTIL_PARTIAL` | L1 | psutil available but one or more cpu/memory fields returned None |
| `WORKER_TORCH_MISSING` | L1 | `import torch` fails — CUDA/MPS not detectable; accelerator defaults to `"cpu"` |
| `WORKER_CGROUP_DENIED` | L1 | cgroup read failed (permission error) — memory limit may be under-reported |
| `WORKER_CONTAINER_AMBIGUOUS` | L2 | Two or more container detection signals disagree |
| `WORKER_DECLARED_PII_LIKE` | L3 | A declared free-string field matches a hostname-like pattern |

Warnings never block run execution. They appear in `validation_report.json` and
are surfaced by `pcq validate-run`.

### cgroups — host view limitation (DEC-011)

pcq 1.x reports host-view memory and CPU from psutil. It does **not** read
cgroups v2 memory limits (`/sys/fs/cgroup/memory.max`) to infer container
memory limits. This is an intentional out-of-scope decision (cgroups 1.0 also
out-of-scope):

- psutil reports the host's physical RAM, not the container limit
- this is marked `source: "detected"` with no qualification
- if the container limit differs materially, the operator should override via
  `CQ_WORKER_MEMORY_TOTAL_GB` or `cq.yaml.worker.memory.total_gb` (declared)
- a future phase may add cgroups limit detection; it will emit a separate
  `memory.limit_gb` field — additive, no schema bump required

The `WORKER_CGROUP_DENIED` warning fires only when pcq explicitly attempts a
cgroup read and receives a permission error; it does not fire for the host-view
path.

### psutil dependency and wheel matrix

psutil is added as a required dependency (DEC-002):

```
psutil>=5.9
```

psutil publishes pre-built wheels for all targets pcq supports:

| Wheel target | psutil wheel available |
|---|---|
| `manylinux2014_x86_64` | yes |
| `manylinux2014_aarch64` | yes |
| `musllinux_1_2_x86_64` | yes |
| `macosx_11_arm64` (Apple Silicon) | yes |
| `macosx_10_15_x86_64` (Intel Mac) | yes |
| `win_amd64` | yes |
| `win_arm64` | yes |

psutil is pure-Python with a thin C extension; the extension is pre-compiled in
all wheels above. No compiler is required at install time on any of these
targets. License: BSD-3-Clause (compatible with Apache-2.0).

When psutil is not available (edge case: source-only install on an exotic
target), pcq emits `WORKER_PSUTIL_MISSING` and continues with null cpu/memory
fields — it does not crash.

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
