# Worker Execution Flow

## Purpose

A CQ worker is a contract executor. It is not a PyTorch runner, a Hugging Face
runner, a PyCaret runner, or a model-specific orchestrator.

The worker receives a project folder with `cq.yaml`, prepares the runtime
context, executes the declared command, then collects standard artifacts from
the resolved output directory.

```text
project folder + data bindings
  -> resolve cq.yaml contract
  -> prepare environment
  -> execute cmd
  -> collect metrics and artifacts
  -> validate run
  -> upload/index RunRecord
```

## Worker Responsibilities

The worker owns:

- checking out or receiving the project directory
- selecting the project `cq.yaml`
- resolving input data and mount paths
- creating the runtime config JSON
- setting environment variables
- executing `cq.yaml.cmd`
- collecting stdout metric events
- enforcing runtime limits
- collecting artifacts from `output_dir`
- running post-run validation
- uploading or indexing run artifacts

The worker does not own:

- model architecture decisions
- loss function implementation
- training loop semantics
- framework-specific adapters
- experiment strategy
- agent prompting

## Execution Steps

### 1. Receive Project

The worker receives a project layout similar to:

```text
project/
  cq.yaml
  train.py
  cq_atoms.py
  atoms/
  pyproject.toml
  uv.lock
```

`cq.yaml` is mandatory for CQ-managed execution.

### 2. Resolve Contract

The worker resolves:

- `project_root`
- `cq_yaml_path`
- `name`
- `cmd`
- `configs`
- `metrics`
- `inputs`
- `artifacts`
- `output_dir`

The same interpretation should be visible through:

```bash
pcq resolve --json
```

### 3. Prepare Inputs

For each `cq.yaml.inputs` entry, the worker resolves the input identity into a
local mount path.

Example:

```yaml
inputs:
  dataset:
    uri: cq://datasets/dental/v12
    mount: data/dental
```

Possible environment:

```text
CQ_INPUT_DIR_DATASET=/work/inputs/dataset
```

The worker should preserve the original input identity in the final
`run_record.json`. A mounted path alone is not enough to reproduce the run.

### 4. Prepare Runtime Config

The worker writes a normalized JSON config and points `CQ_CONFIG_JSON` at it.

Example:

```json
{
  "output_dir": "/work/runs/exp001/output",
  "epochs": 50,
  "batch_size": 8,
  "lr": 0.0003,
  "seed": 42,
  "_cq_project_root": "/work/project",
  "_cq_yaml_path": "/work/project/cq.yaml",
  "_cmd": "uv run python train.py",
  "_run_name": "dental-seg-v4",
  "_metrics_declared": ["epoch", "eval_iou", "eval_loss"]
}
```

The worker may also set:

```text
CQ_DECLARED_METRICS=epoch,eval_iou,eval_loss
```

This is an optimization and compatibility aid. The project should still be
valid when `cq.yaml` is the only source of declarations.

### 5. Execute Command

The worker executes:

```bash
cd /work/project
uv run python train.py
```

The command comes from `cq.yaml.cmd`. The worker may wrap it with sandbox,
timeout, GPU, or logging controls, but it should not change the experiment
contract.

### 6. User Code Uses pcq

Inside the process:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

# any ML framework code

pcq.log(epoch=0, eval_iou=0.71, eval_loss=0.42)
pcq.save_all(
    history=history,
    status="completed",
    artifacts={"model": "model.pt"},
)
```

The same contract can be honored without `Trainer` or `Experiment`.

### 7. Capture Metric Stream

`pcq.log(...)` emits stdout tokens:

```text
@epoch=0 @eval_iou=0.71 @eval_loss=0.42
```

The worker captures these for live progress. The durable history is still
`metrics.json` in the output directory.

### 8. Collect Artifacts

The worker collects files under the resolved `output_dir`, not a hard-coded
`output/` path.

Required post-run files:

- `config.json`
- `metrics.json`
- `run_summary.json`
- `manifest.json`
- `run_record.json`
- `validation_report.json`

If the user code did not call `pcq.save_all(finalize=True)`, the worker may run:

```bash
pcq finalize <output_dir>
```

This should preserve `cq.yaml` top-level metadata by resolving the project root
from `_cq_project_root`, `_cq_yaml_path`, or ancestor search.

### 9. Validate Run

The worker runs:

```bash
pcq validate-run <output_dir> --json
```

Validation should check:

- manifest exists
- manifest entries point to real files
- checksums match when present
- metrics and summary are valid JSON
- RunRecord exists
- validation result is recorded

### 10. Upload And Index

The service indexes `run_record.json` as the canonical completion object.

The service may separately upload all manifest files, but `run_record.json`
should be enough for an agent to decide:

- what was run
- with which config
- from which source state
- on which inputs
- what artifacts were produced
- what metric was best
- whether the run passed validation
- whether a follow-up experiment is justified

## Failure Handling

### Command Failure

If `cmd` exits non-zero, the worker should still attempt to collect partial
artifacts and write a structured failure record when possible.

Minimum failure evidence:

- exit code
- stderr tail
- stdout metric tail
- resolved config
- source snapshot
- output directory listing

### Missing Artifacts

Missing standard artifacts are post-run validation failures. The worker should
not treat a zero exit code as complete when `run_record.json` or
`validation_report.json` is missing.

### Undeclared Metrics

Undeclared metrics should be warnings in permissive mode and failures in strict
service mode. The service can choose policy, but the declared metric set must be
visible in the runtime config and RunRecord.

## Worker Contract Summary

```text
Input:
  project_root with cq.yaml
  optional mounted inputs
  optional service overrides

Execution:
  resolve cq.yaml
  prepare CQ_CONFIG_JSON
  run cq.yaml.cmd
  capture @metric=value stdout

Output:
  resolved output_dir
  standard artifacts
  run_record.json
  validation_report.json
```

The worker is successful when the contract is complete, not merely when the
process exits with code 0.
