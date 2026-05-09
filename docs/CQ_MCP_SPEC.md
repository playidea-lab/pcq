# CQ MCP Tool Spec

## Purpose

This document defines the service-facing MCP surface that lets agents operate
CQ ML projects through structured tools instead of ad hoc shell commands.

The MCP layer belongs to the managed CQ service. `pcq` remains the open-source
contract library. MCP tools may call `pcq` CLI/API internally, but the contract
boundary stays `cq.yaml`, environment, stdout metrics, output artifacts, and
RunRecord.

For the remaining strictness, evidence, E2E, and release-hardening work that
must be completed before CQ service treats this surface as fully stable, see
[pcq Completion Roadmap](PCQ_COMPLETION_ROADMAP.md).

Core pcq CLI payloads are frozen in [JSON Contracts](JSON_CONTRACTS.md) and
`pcq.agent.json_contracts.JSON_CONTRACTS`. MCP wrappers should call
`pcq ... --json` or direct Python APIs and validate those payloads against the
contract registry where possible.
Validation strictness semantics are defined in
[Strictness Evidence Matrix](STRICTNESS.md) and
`pcq.agent.strictness.STRICTNESS_EVIDENCE_MATRIX`.

## Design Principles

- Tools expose structured JSON inputs and outputs.
- Tools do not hide the CQ runtime contract.
- Tools prefer project-local code for new research components.
- Tools are safe for agents: inspect and validate are read-only.
- Run tools return artifact locations and RunRecord summaries.
- Service-only concerns such as credentials, queues, GPUs, and upload storage
  stay outside the open-source `pcq` core.

## Tool Groups

### Project Introspection

#### `cq_resolve_project`

Resolve `cq.yaml` plus environment into a read-only view.

Input:

```json
{
  "project_root": "/work/project",
  "cq_yaml_path": null
}
```

Output:

```json
{
  "schema_version": 1,
  "project_root": "/work/project",
  "cq_yaml_path": "/work/project/cq.yaml",
  "name": "dental-seg-v4",
  "cmd": "uv run python train.py",
  "cfg": {"output_dir": "output", "epochs": 50},
  "declared_metrics": ["epoch", "eval_iou", "eval_loss"],
  "metrics_schema": {
    "eval_iou": {"mode": "max", "split": "val"}
  },
  "inputs": {
    "dataset": {"uri": "cq://datasets/dental/v12"}
  },
  "output_dir": "/work/project/output"
}
```

Side effects: none.

Current `pcq` CLI mapping:

```bash
pcq resolve PROJECT_ROOT --cq-yaml CQ_YAML_PATH --json
```

#### `cq_inspect_project`

Return project structure, entrypoint, recipes, atoms, and output evidence.

Input:

```json
{
  "project_root": "/work/project",
  "load_project_atoms": false
}
```

Output:

```json
{
  "project_type": "pcq",
  "entrypoint": {"path": "train.py", "kind": "script"},
  "outputs": {
    "output_dir": "output",
    "has_manifest": true,
    "has_metrics": true,
    "has_run_record": true
  },
  "warnings": [],
  "errors": []
}
```

Side effects: none when `load_project_atoms=false`. Dynamic atom loading should
be explicit because it imports project code.

Current `pcq` CLI mapping:

```bash
pcq inspect PROJECT_ROOT --json
pcq inspect PROJECT_ROOT --load-project-atoms --json
```

#### `cq_validate_project`

Run static and contract validation before execution.

Input:

```json
{
  "project_root": "/work/project",
  "strictness": 2
}
```

Output:

```json
{
  "status": "pass",
  "strictness": 2,
  "strictness_name": "standard",
  "checks": [
    {
      "id": "has_cq_yaml",
      "status": "pass",
      "severity": "info"
    }
  ]
}
```

Side effects: none.

Current `pcq` CLI mapping:

```bash
pcq validate PROJECT_ROOT --strictness 2 --json
```

### Scaffolding

#### `cq_scaffold_experiment`

Create a CQ-runnable experiment skeleton.

Input:

```json
{
  "project_root": "/work/project",
  "style": "script",
  "name": "tabular-baseline",
  "preset": null,
  "with_pyproject": true,
  "force": false
}
```

Output:

```json
{
  "files_created": ["cq.yaml", "train.py", "pyproject.toml"],
  "files_skipped": [],
  "warnings": []
}
```

Side effects: writes project files.

Current `pcq` CLI mapping:

```bash
pcq init-experiment --style script --output PROJECT_ROOT --name NAME --json
```

#### `cq_scaffold_atom`

Create a project-local atom skeleton.

Input:

```json
{
  "project_root": "/work/project",
  "kind": "model",
  "name": "dental_unet"
}
```

Output:

```json
{
  "status": "pass",
  "path": "atoms/models.py",
  "checks": []
}
```

Side effects: writes project files.

Current `pcq` CLI mapping:

```bash
pcq atoms scaffold KIND NAME --path PROJECT_ROOT --json
```

### Execution

#### `cq_run_experiment`

Submit or execute a CQ run.

Input:

```json
{
  "project_root": "/work/project",
  "cq_yaml_path": "/work/project/cq.yaml",
  "overrides": {
    "epochs": 50,
    "lr": 0.0003
  },
  "inputs": {
    "dataset": "cq://datasets/dental/v12"
  },
  "resources": {
    "gpu": "1xA100",
    "timeout_minutes": 120
  }
}
```

Output:

```json
{
  "run_id": "run_20260506_120000_ab12cd",
  "status": "queued",
  "output_dir": "cq://runs/run_20260506_120000_ab12cd/output"
}
```

Side effects: queues or starts a managed run.

#### `cq_finalize_run`

Create or refresh `run_record.json` and `validation_report.json`.

Input:

```json
{
  "project_root": "/work/project",
  "output_dir": "/work/project/output",
  "status": "completed"
}
```

Output:

```json
{
  "run_record_path": "/work/project/output/run_record.json",
  "validation_report_path": "/work/project/output/validation_report.json"
}
```

Side effects: writes output artifacts.

Current `pcq` CLI mapping:

```bash
pcq finalize OUTPUT_DIR --project-root PROJECT_ROOT --status completed --json
```

### Post-Run Analysis

#### `cq_validate_run`

Validate a completed output directory.

Input:

```json
{
  "output_dir": "/work/project/output",
  "strictness": 3
}
```

Output:

```json
{
  "status": "pass",
  "strictness": 3,
  "strictness_name": "reproducible",
  "checks": []
}
```

Side effects: none unless the service chooses to persist the validation result.

Current `pcq` CLI mapping:

```bash
pcq validate-run OUTPUT_DIR --strictness 3 --json
```

#### `cq_describe_run`

Return a compact run summary for agent decisions.

Input:

```json
{
  "output_dir": "/work/project/output"
}
```

Output:

```json
{
  "schema_version": 1,
  "status": "completed",
  "target_metric": "eval_iou",
  "mode": "max",
  "best": {"epoch": 41, "value": 0.713},
  "best_value": 0.713,
  "best_epoch": 41,
  "last": {"epoch": 50, "value": 0.709},
  "validation_status": "pass",
  "artifacts_summary": {"model": 1, "checkpoint": 2},
  "reproducibility_evidence": {
    "source": {"git_sha": "...", "dirty": false},
    "config": {"seed": 42, "strictness": 3}
  },
  "decision_facts": {
    "run_completed": true,
    "validation_passed": true,
    "has_best": true,
    "artifact_count": 3
  }
}
```

Side effects: none.

The tool returns facts, not recommendations. Policy such as whether to continue,
fork, stop, or rollback belongs to the calling agent or service.

Current `pcq` CLI mapping:

```bash
pcq describe-run OUTPUT_DIR --json
```

#### `cq_compare_runs`

Compare two completed runs.

Input:

```json
{
  "base_output_dir": "/work/runs/base/output",
  "candidate_output_dir": "/work/runs/candidate/output"
}
```

Output:

```json
{
  "schema_version": 1,
  "a_run_id": "baseline",
  "b_run_id": "candidate",
  "target_metric": "eval_iou",
  "a_target_metric": "eval_iou",
  "b_target_metric": "eval_iou",
  "mode": "max",
  "best": {"a": 0.701, "b": 0.713, "delta": 0.012, "direction": "improved"},
  "last": {"a": 0.699, "b": 0.709, "delta": 0.01, "direction": "improved"},
  "metric_delta": 0.012,
  "metric_direction": "improved",
  "config_changes": [{"key": "lr", "a": 0.001, "b": 0.0003}],
  "validation": {"a": "pass", "b": "pass", "same": true},
  "failure": {"same": true},
  "artifacts": {"a_count": 4, "b_count": 5},
  "source": {"same_git_sha": false, "same_cq_yaml_sha256": false},
  "decision_facts": {
    "comparable": true,
    "same_target_metric": true,
    "best_improved": true,
    "candidate_validated": true,
    "config_changed": true,
    "source_changed": true
  }
}
```

Side effects: none.

The tool returns comparison facts, not a winner recommendation. If a service
wants a winner label, it should derive it from `mode`, `metric_direction`,
validation, failure, and its own policy.

#### `cq_lineage`

Return parent chain for a run.

Input:

```json
{
  "output_dir": "/work/runs/candidate/output",
  "max_depth": 10
}
```

Output:

```json
{
  "nodes": [
    {"run_id": "candidate", "parent_run_id": "base"},
    {"run_id": "base", "parent_run_id": null}
  ],
  "warnings": []
}
```

Side effects: none.

### Data And Artifact Service Tools

These are CQ service tools, not `pcq` library functions.

#### `cq_fetch_dataset`

Input:

```json
{
  "uri": "cq://datasets/dental/v12",
  "mount_name": "dataset",
  "target_dir": "/work/inputs/dataset"
}
```

Output:

```json
{
  "local_path": "/work/inputs/dataset",
  "identity": {
    "uri": "cq://datasets/dental/v12",
    "version": "v12"
  }
}
```

#### `cq_upload_artifacts`

Input:

```json
{
  "run_id": "run_20260506_120000_ab12cd",
  "output_dir": "/work/project/output",
  "manifest_path": "/work/project/output/manifest.json"
}
```

Output:

```json
{
  "artifact_root": "cq://runs/run_20260506_120000_ab12cd/output",
  "files_uploaded": 8
}
```

## Error Shape

All tools should return structured errors:

```json
{
  "status": "error",
  "error": {
    "code": "missing_cq_yaml",
    "message": "cq.yaml not found under project_root",
    "recoverable": true,
    "suggested_fix": "run cq_scaffold_experiment or provide cq_yaml_path"
  }
}
```

## Minimal MCP Milestone

The first useful CQ MCP should implement:

1. `cq_resolve_project`
2. `cq_inspect_project`
3. `cq_validate_project`
4. `cq_scaffold_experiment`
5. `cq_scaffold_atom`
6. `cq_finalize_run`
7. `cq_validate_run`
8. `cq_describe_run`

`cq_run_experiment`, dataset fetch, artifact upload, compare, and lineage can
follow once the service queue and storage contracts are stable.
