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
