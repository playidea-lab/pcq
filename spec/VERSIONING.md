# Schema Versioning Policy

> Rules for evolving pcq's `schema_version` field across the JSON
> contracts (`spec/JSON_CONTRACTS.md`, `src/pcq/agent/json_contracts.py`,
> and the generated `spec/schemas/*.schema.json`).

## TL;DR

- Every pcq agent-facing JSON object carries `schema_version` (integer).
- **Within a major schema_version (e.g. `1`), changes are additive-only.**
  Callers may depend on every required field; pcq may add more.
- A breaking change → bump `schema_version` (1 → 2). Two majors
  coexist in the runtime for at least one minor pcq release.
- The library version (`pcq.__version__`, semver) is independent of
  `schema_version`. Many pcq releases share `schema_version=1`.

## Why "additive-only within MAJOR"

The frozen registry in [`src/pcq/agent/json_contracts.py`](../src/pcq/agent/json_contracts.py)
opens with this guarantee, in code:

```python
"""
The contracts here intentionally describe the minimum stable surface,
not every field a command may emit. Within ``schema_version == 1``
these required fields are additive-only: callers may depend on them,
and pcq may add more fields.
"""
```

Lifting this from a docstring to `spec/` makes it the single rule for
external implementations and conformance suites.

## Definitions

- **schema_version** — integer field present on every agent-facing JSON
  object (e.g. `pcq run --json` envelope, `pcq describe-run --json`
  record, `pcq validate-run --json` report).
- **Required field** — field listed in a contract's `required` set in
  `JSON_CONTRACTS` (or in the generated JSON Schema). Callers may rely
  on it being present, with the documented type, in every response that
  carries the matching `schema_version`.
- **Optional field** — anything else. May appear, may be absent, may be
  added in any minor pcq release within the same `schema_version`.

## Rules

### Allowed within the same `schema_version`

- Adding a new optional field to any contract.
- Adding a new contract to the registry.
- Adding a new value to an open enum (where the spec explicitly says
  "open").
- Loosening a constraint that callers were already required to handle
  (e.g. broadening `string | null` to `string | int | null` if `null`
  was always possible — agents must still defensively type-check).
- Making a previously-optional field required, **only** if every
  released pcq version that produced this `schema_version` already
  emitted it.

### Requires a `schema_version` MAJOR bump

- Removing a required field.
- Renaming a required field.
- Changing the type of a required field.
- Changing the meaning of an existing field (semantic break).
- Tightening an enum (removing a previously-valid value).

## Deprecation timeline

When a field, contract, or behavior is slated for removal in the next
MAJOR:

1. **Pre-deprecation release** (any pcq version on the current
   `schema_version`): ship the replacement alongside the old field.
   Both appear in JSON output. Both are documented.
2. **Deprecation release** (still current `schema_version`): keep both
   fields, but the old field's spec entry carries a `deprecated` note
   pointing at the replacement and the planned MAJOR release.
3. **Removal release** (MAJOR bump): old field gone, `schema_version`
   incremented, old contract still served when an agent explicitly
   requests it (see "Coexistence", below).

Minimum deprecation period: **at least one pcq minor release** with
both fields present. There is no calendar guarantee — pcq's release
cadence is irregular — only an *ordering* guarantee.

## Coexistence of two MAJORs

When `schema_version` is bumped from `N` to `N+1`:

- The new pcq runtime serves `N+1` by default.
- A caller that explicitly opts into `N` (e.g. via `--schema 1` CLI
  flag, `schema=1` in MCP tool input, or `Accept-Schema: 1` header for
  future HTTP endpoints) receives the legacy shape.
- Coexistence lasts at least one pcq minor release. After that, `N`
  may be removed; doing so is not itself a `schema_version` bump (it's
  a release-notes entry).

This protects in-flight agents and CI pipelines from same-day breakage
when a major rolls.

## Discovery

Agents can discover the runtime's supported `schema_version` set in
two equivalent ways:

- **CLI**: `pcq resolve --json` envelope includes
  `"supported_schema_versions": [1]` (single-element list today).
- **MCP**: `mcp__pcq__resolve_project` returns the same list under the
  same key.

When an agent receives a response whose `schema_version` is *not* in
its known set, the recommended behavior is:

- Log a structured warning naming the unexpected version.
- Continue processing fields it recognizes (forward-compat: extra
  fields are allowed).
- Treat absence of a known required field as the contract violation,
  not the version itself.

## Library version vs schema version

`pcq.__version__` follows semver (MAJOR.MINOR.PATCH). It changes with
every release — feature, fix, packaging, deprecation, or breaking
change. `schema_version` only changes when the JSON contract itself
breaks. As of this writing:

| pcq library | schema_version |
|-------------|----------------|
| 2.7.x — 2.12.x | `1` |
| 2.13.x — 4.x.x | `1` (frozen) |

A pcq 5.x release that, say, restructured the `validation_report`
shape would bump `schema_version` to `2` while shipping under `pcq 5`.

## Conformance link

Every contract change that needs MAJOR enforcement should have a
golden pair under
[`tests/conformance/<contract>/`](../tests/conformance/) demonstrating
the new shape — see [`CONFORMANCE.md`](./CONFORMANCE.md).

## Attribution field — env precedence and PII guidance

This section is normative for the `attribution` object introduced in
`run_record.json` / `pcq.describe_run.record`. See [`SPEC.md §
Attribution`](./SPEC.md) for the schema and field semantics.

### Resolution priority

When pcq writes `run_record.json`, it resolves each attribution field using
the following precedence (first winning source is used, remaining sources are
ignored for that field):

```
1. CLI flag (highest)
2. CQ_ATTRIBUTION_* environment variable
3. cq.yaml  attribution.*  config block
4. auto-infer (e.g. git config user.email)
5. NULL (field absent or null)    ← lowest
```

No source is mandatory. If all sources are absent, `attribution` itself is
`null` in the output, which is valid under `schema_version: 1`.

### Environment variables (CQ_ATTRIBUTION_*)

| Variable | Populated field |
|---|---|
| `CQ_ATTRIBUTION_OPERATOR` | `attribution.operator` |
| `CQ_ATTRIBUTION_AUTHOR_ID` | `attribution.author.id` |
| `CQ_ATTRIBUTION_AUTHOR_KIND` | `attribution.author.kind` |
| `CQ_ATTRIBUTION_COMMITTER_ID` | `attribution.committer.id` |
| `CQ_ATTRIBUTION_COMMITTER_KIND` | `attribution.committer.kind` |
| `CQ_ATTRIBUTION_SESSION_ID` | `attribution.session_id` |
| `CQ_ATTRIBUTION_PERSONA_AUTHOR` | `attribution.author.persona_id` |
| `CQ_ATTRIBUTION_PERSONA_COMMITTER` | `attribution.committer.persona_id` |

`CQ_ATTRIBUTION_AUTHOR_KIND` and `CQ_ATTRIBUTION_COMMITTER_KIND` accept the
values `"human"` or `"agent"`. Any other value should be rejected with a clear
error at write time.

### `cq.yaml` attribution block (example)

```yaml
attribution:
  operator: pilab
  author:
    id: changmin
    kind: human
  committer:
    id: claude-opus-4-7
    kind: agent
```

All sub-keys are optional. Omitted keys fall through to auto-infer or NULL.

### Additive-only schema bump policy for `attribution`

The `attribution` object itself is an optional additive field at
`schema_version: 1`. Its internal `schema_version` sub-field tracks the
shape of the object independently:

- Adding new optional keys inside `attribution` (e.g. `session_id`,
  `persona_id`) does **not** require a bump of the outer `schema_version`.
- The `attribution.schema_version` counter increments only when a field
  inside `attribution` is *removed or renamed* — which is itself a MAJOR
  bump of the outer `schema_version` (because it changes the contract of a
  field that callers may depend on).
- The `signature` key is reserved and must not be treated as absent-required;
  its absence is normal in Phase 1.

In practice: attribution additions ship as minor pcq releases with no
`schema_version` change.

### PII (Personally Identifiable Information) guidance

`operator`, `author.id`, and `committer.id` are free strings. They have no
built-in privacy enforcement at the pcq format layer.

**Recommended practice**:

1. **Use a pseudonym or UUID** for `operator`. Examples: `"pilab"`,
   `"550e8400-e29b-41d4-a716-446655440000"`. Avoid email addresses and real
   names as the primary identifier.
2. **Do not use email addresses directly** in `author.id` or `committer.id`
   in any environment where run records may be shared externally or ingested
   into TheCommons / a public evidence store.
3. **CI environments**: `git config user.email` auto-infer may expose real
   email addresses. Set `CQ_ATTRIBUTION_AUTHOR_ID` explicitly in CI if the
   output will be published.
4. **Redaction is the consumer's responsibility**. CQ Hub, TheCommons, and
   any downstream store must apply their own PII policy before persisting or
   publishing attribution data.

pcq will never strip or hash attribution fields on behalf of a caller. The
format is a neutral carrier; policy is upstream.

## Worker Spec — env precedence + cgroups limit + PII guidance

This section is normative for the `worker_spec` object introduced in
`run_record.json` / `pcq.describe_run.record`. See [`SPEC.md §
Worker Spec`](./SPEC.md) for the full schema and field semantics.

### Resolution priority

When pcq writes `run_record.json`, it resolves each worker_spec field using
the following precedence (first winning source is used):

```
1. CLI flag (highest — e.g. --worker-cpu-model)
2. CQ_WORKER_* environment variable
3. cq.yaml  worker.*  config block
4. auto-detect (psutil + torch + optional GPUtil)
5. NULL (field absent or null)    ← lowest
```

No source is mandatory. If all sources are absent or detection fails,
`worker_spec` itself is `null` in the output, which is valid under
`schema_version: 1`.

The `source` audit field reflects which sources contributed:

- `"detected"` — every populated field came from auto-detection
- `"declared"` — every populated field came from step 2 or 3 (user-supplied)
- `"merged"` — mix of auto-detected and user-supplied fields

### Environment variables (CQ_WORKER_*)

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
| `CQ_WORKER_CONTAINER_KIND` | `worker_spec.container.kind` (must be one of: `none`, `docker`, `k8s`, `other`) |

### `cq.yaml` worker block (example)

```yaml
worker:
  cpu:
    model: "Apple M2 Pro"
    cores_physical: 10
  memory:
    total_gb: 32
  accelerator:
    kind: mps
  container:
    kind: none
```

All sub-keys are optional. Omitted keys fall through to auto-detect or NULL.
Partial declarations produce `source: "merged"`.

### cgroups host-view limitation

pcq reports **host-view** memory from psutil (`psutil.virtual_memory().total`).
It does not read cgroups v1 or v2 memory limits at detection time. This is an
explicit out-of-scope decision (DEC-011):

- In a container, `psutil.virtual_memory()` typically returns the host's physical
  RAM, not the container's memory limit.
- pcq records this as `source: "detected"` with no caveats in the field itself.
- If the container limit differs materially from the host RAM, set
  `CQ_WORKER_MEMORY_TOTAL_GB` or `cq.yaml.worker.memory.total_gb` to the
  actual container limit (this produces `source: "merged"`).
- Future phase: a `memory.limit_gb` field sourced from cgroup may be added
  additively (no schema_version bump required).

### PII (Personally Identifiable Information) guidance

**R10 — Auto-detection layer (code-level prohibition)**:
Auto-detection code must NEVER write hostname, IP address, MAC address, or
user/login name into any `worker_spec` field. This applies unconditionally to
the `source: "detected"` and `source: "merged"` (auto-detected portion) paths.
No opt-in overrides this prohibition.

**R14 — Declared path warning (validation layer)**:
User-supplied values (via `CQ_WORKER_*` or `cq.yaml.worker.*`) are not subject
to R10 filtering — pcq does not redact them. However, at validation time, pcq
inspects every free-string `worker_spec` field for patterns that resemble a
hostname (regex: `\b[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+\b`). When a match is found
a severity-3 (L3) warning is added to `validation_report.json`:

```
code: WORKER_DECLARED_PII_LIKE
detail: "worker_spec.<field> may contain a hostname or FQDN — review before publishing"
```

This warning is advisory. It does not block run execution, validation, or
publication. Consumers (CQ Hub, TheCommons) must apply their own PII policy.

**Recommended practice**:

1. Do not put hostnames, FQDNs, or IP addresses in `cpu.model`,
   `os.release`, `container.image`, or `container.detector_hint`.
2. Use an opaque identifier (e.g. `"gpu-node-42"`) when a machine label is
   needed and the record will be published.
3. For CI environments, set `CQ_WORKER_CPU_MODEL` and other model fields
   explicitly so auto-detection is bypassed and the label is predictable.

### Additive-only schema bump policy for `worker_spec`

The `worker_spec` object is an optional additive field at `schema_version: 1`.
Its internal `schema_version` sub-field tracks the shape of the object:

- Adding new optional keys inside `worker_spec` (e.g. `disk`, `network`) does
  **not** require a bump of the outer `schema_version`.
- The `worker_spec.schema_version` counter increments only when a field inside
  `worker_spec` is *removed or renamed*.
- The four flat surface fields (`worker_spec_cpu_model`, `worker_spec_memory_gb`,
  `worker_spec_accelerator_kind`, `worker_spec_gpu_model_0`) are frozen once
  shipped; new summary fields follow the same flat naming convention and may be
  added in any minor pcq release.

In practice: worker_spec additions ship as minor pcq releases with no outer
`schema_version` change.
