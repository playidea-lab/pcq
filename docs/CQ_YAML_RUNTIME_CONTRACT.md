# CQ YAML Runtime Contract

## Purpose

`cq.yaml` is the runtime contract between a CQ project, a CQ worker, `pcq`,
and an agent. It is not just project documentation. It is the source of truth
that describes what to run, which configuration values are active, which inputs
are bound, which metrics are expected, and where artifacts should be collected.

The target behavior is:

```text
cq.yaml + CQ_CONFIG_JSON + environment
  -> ResolvedConfig
  -> RunContext
  -> standard artifacts
  -> run_record.json
```

Every read-side and write-side command should use the same interpretation.

## Runtime Objects

### ResolvedConfig

`ResolvedConfig` is the read-only interpretation of the project contract.

It answers:

- where the project root is
- which `cq.yaml` was selected
- what the merged config is
- which metrics are declared
- which inputs and artifact declarations exist
- where `output_dir` points
- which parse warnings or errors were found

It must not:

- create directories
- change cwd
- write artifacts
- import project training code
- submit jobs

Read-side commands such as `pcq resolve`, `pcq inspect`, and pre-run
validation should be able to call the resolver without changing the workspace.

### RunContext

`RunContext` is the write-time context derived from `ResolvedConfig`.

It answers:

- the absolute output directory to write to
- the merged runtime config to pass to user code
- the resolved run name and command
- the metric declaration set used by `pcq.log`
- input mount paths visible to the running process

It may:

- create `output_dir`
- prepare temporary config JSON
- normalize environment variables

Only write-side paths should use `RunContext`: `pcq.output_dir()`,
`pcq.save_all()`, `pcq finalize`, worker execution, and post-run artifact
collection.

## Resolution Order

### Project Root

```text
explicit project_root argument
  -> _cq_project_root in CQ_CONFIG_JSON
  -> _CQ_PROJECT_ROOT env, if introduced by the worker
  -> parent of explicit cq_yaml_path
  -> cwd ancestor containing cq.yaml
  -> cwd
```

Ancestor search should stop at a nested project marker such as `.git` or
`pyproject.toml` when no `cq.yaml` exists in that directory. This prevents a
child project from accidentally inheriting a parent `cq.yaml`.

### CQ YAML Path

```text
explicit cq_yaml_path argument
  -> _cq_yaml_path in CQ_CONFIG_JSON
  -> _CQ_YAML_PATH env, if introduced by the worker
  -> project_root/cq.yaml
  -> project_root/pcq.yml
  -> cwd ancestor cq.yaml
```

### Runtime Config

```text
cq.yaml.configs
  merged with CQ_CONFIG_JSON
  where CQ_CONFIG_JSON wins on key conflict
```

The service may inject reserved keys into `CQ_CONFIG_JSON`, but local execution
must still work from `cq.yaml` alone.

Reserved keys should use a leading underscore:

- `_cmd`
- `_run_name`
- `_cq_project_root`
- `_cq_yaml_path`
- `_metrics_declared`
- `_parent_run_id`
- `_parent_run_path`

User-facing config keys should not need the underscore.

### Output Directory

```text
explicit output_dir argument
  -> CQ_CONFIG_JSON.output_dir
  -> cq.yaml.configs.output_dir
  -> project_root/output
```

Relative paths are always resolved against `project_root`, not the current
working directory.

Examples:

```yaml
configs:
  output_dir: output
```

```text
project/output
```

```yaml
configs:
  output_dir: runs/exp001
```

```text
project/runs/exp001
```

```yaml
configs:
  output_dir: /mnt/runs/exp001
```

```text
/mnt/runs/exp001
```

### Run Identity

```text
run.name:
  cfg._run_name
  -> cfg.name
  -> cq.yaml.name
  -> ""

execution.cmd:
  cfg._cmd
  -> cq.yaml.cmd
  -> ""
```

`cq.yaml.name` and `cq.yaml.cmd` are top-level contract fields. They should be
preserved in `run_record.json` even when the service does not inject `_run_name`
or `_cmd`.

### Metrics

```text
CQ_DECLARED_METRICS env
  -> cq.yaml.metrics
  -> CQ_CONFIG_JSON._metrics_declared
  -> no declared metric gate
```

`cq.yaml.metrics` may be list-style:

```yaml
metrics:
  - epoch
  - eval_acc
```

or schema-style:

```yaml
metrics:
  eval_iou:
    mode: max
    split: val
  eval_loss:
    mode: min
    split: val
```

`ResolvedConfig.declared_metrics` should always expose a list of metric names.
`ResolvedConfig.metrics_schema` should preserve schema-style metadata for
RunRecord and agent decision making.

### Inputs

`cq.yaml.inputs` is provenance and intent:

```yaml
inputs:
  dataset:
    uri: cq://datasets/dental/v12
    mount: data/dental
```

Worker-resolved input paths are operational bindings:

```text
CQ_INPUT_DIR_DATASET=/work/inputs/dataset
```

`pcq.input_dir("dataset")` should prefer the worker binding and may fall back to
config values when appropriate. `run_record.json` should preserve the original
`cq.yaml.inputs` identity so a future run can understand what data was meant.

## Standard Artifacts

All standard artifacts belong under the resolved `output_dir`.

Minimum contract artifacts:

- `config.json`
- `metrics.json`
- `run_summary.json`
- `manifest.json`
- `run_record.json`
- `validation_report.json`

Model or framework-specific outputs are also stored under `output_dir`, for
example:

- `model.pt`
- `model.pkl`
- `best.ckpt`
- `last.ckpt`
- `predictions.parquet`
- `plots/`

`manifest.json` indexes produced artifacts. `run_record.json` is the completion
boundary that combines execution, source, environment, inputs, metrics,
artifacts, validation, and result summary.

## Required Invariants

These invariants define correct behavior:

- `resolve_project()` is read-only.
- `pcq.output_dir()` returns the same path as the resolved runtime output dir.
- `pcq.save_all(finalize=True)` writes every standard artifact to the same
  output directory.
- `pcq inspect` reports artifacts from the resolved output directory.
- `pcq validate` checks post-run evidence in the resolved output directory.
- `pcq finalize <output_dir>` can find the project `cq.yaml` without relying
  on the output directory name.
- `run_record.json` preserves `cq.yaml.name`, `cq.yaml.cmd`, `cq.yaml.inputs`,
  and metric declarations when service-injected fields are absent.
- CQ service injection may override config values, but local `cq.yaml`-only
  execution remains valid.

## Example

```yaml
name: dental-seg-v4
cmd: uv run python train.py

inputs:
  dataset:
    uri: cq://datasets/dental/v12
    mount: data/dental

configs:
  output_dir: runs/exp001
  epochs: 50
  batch_size: 8
  lr: 0.0003
  seed: 42

metrics:
  eval_iou:
    mode: max
    split: val
  eval_loss:
    mode: min
    split: val

artifacts:
  - runs/exp001/
```

Expected local behavior:

```text
project/
  cq.yaml
  train.py
  runs/exp001/
    config.json
    metrics.json
    run_summary.json
    manifest.json
    run_record.json
    validation_report.json
```
