# Strictness Evidence Matrix

`strictness` defines how much evidence pcq requires before a project or run can
be treated as reliable. It is not a training policy and does not decide whether
a model is good. It only controls evidence gates.

The machine-readable source of truth is
`pcq.agent.strictness.STRICTNESS_EVIDENCE_MATRIX`. Validation reports include the
selected level and cumulative `required_evidence` in the `strictness_level`
check.

## Levels

| Level | Name | Purpose |
|---|---|---|
| 0 | `parse` | editor feedback and very early scaffolds |
| 1 | `static` | pre-run agent authoring |
| 2 | `standard` | default local and development validation |
| 3 | `reproducible` | CI and serious experiment records |
| 4 | `service_grade` | managed CQ runs and publishable comparisons |

## Required Evidence

The matrix is cumulative. Level 3 includes levels 0, 1, and 2; level 4 includes
everything below it.

### Level 0: Parse

Pre-run:

- `cq_yaml_exists`
- `cq_yaml_parseable`
- `cmd_defined`

Post-run:

- `manifest_present`
- `manifest_parseable`
- `metrics_present`
- `metrics_well_formed`
- `run_summary_present`
- `run_summary_parseable`

### Level 1: Static

Pre-run:

- `metrics_declared`
- `artifacts_declared`
- `cq_config_called`
- `cq_log_called`
- `standard_artifacts_helper`

Post-run: no additional required evidence.

### Level 2: Standard

Pre-run:

- `recipe_importable`
- `recipe_metrics_in_yaml`
- `loss_label_ignore_index`
- `model_dataset_channels`
- `optional_extras_available`
- `monitor_candidates_declared`
- `manifest_evidence`

Post-run:

- `manifest_evidence`
- `summary_metrics_consistent`
- `run_record_complete`

### Level 3: Reproducible

Pre-run:

- `seed_evidence`
- `lockfile_evidence`
- `inputs_evidence`

Post-run:

- `run_record_present`
- `run_finalized`
- `run_record_execution_identity`
- `source_reproducibility`
- `environment_reproducibility`
- `lockfile_evidence`
- `seed_evidence`
- `metrics_schema_evidence`
- `run_record_inputs_evidence`

### Level 4: Service Grade

Pre-run:

- `service_input_identity`
- `service_metric_schema`
- `service_lineage_evidence`

Post-run:

- `service_input_identity`
- `service_metric_schema`
- `service_hardware_evidence`
- `service_lineage_evidence`

## Report Shape

Every validation report includes:

```json
{
  "schema_version": 1,
  "status": "pass",
  "strictness": 3,
  "strictness_name": "reproducible",
  "checks": [
    {
      "id": "strictness_level",
      "status": "pass",
      "severity": "info",
      "detail": "strictness=3 (reproducible)",
      "evidence": {
        "level": 3,
        "name": "reproducible",
        "required_evidence": {
          "pre_run": ["cq_yaml_exists", "cmd_defined", "seed_evidence"],
          "post_run": ["metrics_present", "run_record_present"]
        }
      }
    }
  ],
  "blocking_count": 0,
  "warning_count": 0
}
```

The example shortens the lists for readability. The actual report carries the
full cumulative matrix.

## Agent Rule

Agents should read `strictness_level.evidence.required_evidence` before deciding
that a run is complete enough for the current context. Missing required evidence
must be handled as a validation problem, not as a model-quality result.
