# Agent-Operable pcq

## Purpose

This document describes how to evolve `pcq` from an agent-readable library into
an agent-operable library.

Current v1.4 direction is good: agents can inspect recipes, diff recipes, dry-run
trainer plans, and read provenance from outputs. The next step is stronger:
agents should be able to safely create, modify, validate, run, and evaluate CQ ML
experiments through stable structured contracts instead of editing arbitrary
Python by guesswork.

The target is:

```text
user goal
  -> CQ service coding agent
  -> pcq inspect / plan / validate / apply / summarize
  -> CQ run
  -> metrics + artifacts + provenance
  -> next experiment decision
```

`pcq-agent` is not an LLM runtime. It is an affordance layer for agents. The CQ
service owns prompting, tool execution, permissions, orchestration, and job
scheduling. `pcq` owns structured experiment contracts, metadata extraction,
scaffolding, validation, and result summarization.

## Definitions

### Agent-Readable

An agent-readable library exposes enough metadata for an agent to understand an
experiment without manually reading every file.

Examples:

```python
pcq.recipe_meta("vision/seg/fake_seg_smoke")
pcq.diff_recipes("vision/fake_smoke", "vision/seg/fake_seg_smoke")
pcq.Trainer(preset="vision/fake_smoke").dry_run()
```

This helps the agent understand what exists.

### Agent-Operable

An agent-operable library exposes stable operations that let an agent change the
experiment safely.

Examples:

```bash
pcq inspect . --json
pcq propose --goal "improve eval_iou" --json
pcq apply-plan experiment.plan.json
pcq validate . --json
pcq summarize-run output/ --json
```

This lets the agent perform work through contracts.

## Design Goal

Agents should rarely need to perform free-form edits to training code. The
normal path should be:

```text
inspect structured state
  -> create structured plan
  -> validate plan
  -> apply bounded patch
  -> run CQ job
  -> summarize results
  -> repeat
```

Free-form code editing remains possible, but it should be the fallback for new
research logic, not the default path for common experiment iteration.

When new research logic is needed, the preferred fallback is not to patch random
training code. The preferred fallback is to create a project-local atom with an
explicit contract. The implementation may be free-form PyTorch, but the public
surface should still be `AtomSpec`, `AtomRef`, and validation output.

## Boundary Between CQ Service And pcq

### CQ Service Responsibilities

The CQ service owns:

- user goal intake
- LLM prompting and tool selection
- project checkout and workspace lifecycle
- permission review and sandboxing
- CQ job submission and scheduling
- run comparison across jobs
- higher-level experiment strategy
- deciding whether to continue, stop, or ask a user

### pcq Responsibilities

`pcq` owns:

- project inspection
- recipe and atom metadata
- experiment plan schema
- safe scaffolding
- safe plan application
- local contract validation
- recipe acceptance validation
- artifact and run summary schema
- provenance normalization

### Explicit Non-Responsibility

`pcq` should not:

- call an LLM
- decide business goals
- submit CQ jobs directly from core APIs
- hide CQ worker behavior
- manage cloud credentials
- mutate files without a structured plan
- become a general workflow engine

## Target User Story

A user asks the CQ service:

```text
Improve dental segmentation mIoU.
```

The agent should be able to:

1. Inspect the project.
2. Detect that it is a CQ ML project.
3. Read available recipes, metrics, artifacts, and previous results.
4. Choose a base recipe or create a project-local recipe.
5. Create or select project-local atoms when the needed model, loss, transform,
   dataset, or metric does not already exist.
6. Create an experiment plan that changes monitor, loss, dataset transform,
   model, optimizer, scheduler, or config values.
7. Validate the plan without training.
8. Apply the plan to `cq.yaml`, `train.py`, recipe files, or local atom files.
9. Run the job through CQ.
10. Summarize output metrics and artifacts.
11. Decide the next experiment from structured evidence.

## Core Principle

The agent should manipulate experiment intent, not incidental Python syntax.

Bad default path:

```text
read train.py
guess where to edit
patch arbitrary Python
run
debug incidental errors
```

Preferred path:

```text
inspect recipe/config contract
emit ExperimentPlan JSON
validate ExperimentPlan
apply through pcq
run
summarize
```

For frameworks or workflows that do not fit the recipe/atom shape yet, a
contract script is still agent-operable if it follows the CQ contract:

```text
plain project-local Python
  -> pcq.config()
  -> arbitrary ML framework code
  -> pcq.log(...)
  -> standard artifacts
  -> run_summary.json
```

The default assumption is that project-local code can connect any library
directly. The agent should not wait for or invent framework-specific adapters;
it should normalize the run through the CQ contract.

## Architecture

Conceptual layers:

```text
pcq-core
  config(), input_dir(), output_dir(), log(), artifact helpers

pcq-train
  Experiment, Trainer, checkpoint, resume, monitor, metrics

pcq-recipes
  RecipeSpec, AtomRef, recipe catalog, acceptance tests

pcq-agent
  inspect, plan schema, apply-plan, validate, summarize-run

CQ service
  LLM loop, workspace control, CQ run submission, cross-run decisions
```

These can remain in one distribution. The boundaries matter because each layer
has a different stability contract.

## Operability Levels

### Level 0: Script-Compatible

The project can run a Python script under CQ.

Minimum:

- `cq.yaml`
- `cmd`
- `CQ_CONFIG_JSON`
- stdout metrics
- artifacts

### Level 1: pcq-Compatible

The project uses `pcq` low-level or `Experiment` APIs.

Minimum:

- `pcq.config()`
- `pcq.log()`
- `pcq.output_dir()`
- standard artifacts
- metrics history

### Level 2: Agent-Readable

The project exposes metadata for agents.

Minimum:

- recipe metadata
- dry run
- provenance in `config.json`
- artifact `manifest.json`
- acceptance status

v1.4 is mostly here.

### Level 3: Agent-Operable

The project can be modified through structured plans.

Minimum:

- `pcq inspect --json`
- typed recipe and atom metadata
- project-local atom discovery
- `ExperimentPlan` JSON schema
- `pcq validate --json`
- `pcq apply-plan`
- `run_summary.json`

### Level 4: Agent-Optimizable

The CQ service can run iterative improvement loops.

Minimum:

- previous run comparison
- target metric policy
- failure classification
- known issue extraction
- experiment lineage
- safe automatic next-plan generation

## Required CLI Surface

The Python API is useful for users. Agents need a JSON CLI because CLI output is
stable across processes and language runtimes.

### `pcq inspect`

Purpose: inspect a project folder without running training.

Command:

```bash
pcq inspect . --json
```

Required output:

```json
{
  "schema_version": 1,
  "project_root": "/workspace/project",
  "project_type": "pcq",
  "has_cq_yaml": true,
  "cq_yaml": {
    "path": "cq.yaml",
    "name": "dental-seg",
    "cmd": "uv run python train.py",
    "declared_metrics": ["epoch", "train_loss", "eval_iou"],
    "artifacts": ["output/"]
  },
  "entrypoint": {
    "path": "train.py",
    "kind": "trainer",
    "preset": "vision/seg/voc_unet"
  },
  "recipes": [
    {
      "name": "vision/seg/voc_unet",
      "task": "segmentation",
      "smoke_status": "pass",
      "contract_status": "pass",
      "requires_extras": ["vision"]
    }
  ],
  "outputs": {
    "output_dir": "output",
    "has_manifest": true,
    "has_metrics": true,
    "has_summary": true
  },
  "warnings": []
}
```

Rules:

- Must not import heavy optional dependencies unless needed for metadata.
- Must not instantiate large models.
- Must not download datasets.
- Must not create output artifacts.
- Must return machine-readable warnings instead of only printing text.
- Should load project-local atom metadata from predictable files such as
  `cq_atoms.py` or `atoms/*.py`.
- Project atom import failures should be structured warnings or validation
  errors, not unhandled tracebacks.

### `pcq recipe-meta`

Purpose: inspect one recipe.

Command:

```bash
pcq recipe-meta vision/seg/voc_unet --json
```

Required output:

```json
{
  "schema_version": 1,
  "name": "vision/seg/voc_unet",
  "task": "segmentation",
  "description": "Pascal VOC 2012 segmentation with UNet baseline",
  "declared_metrics": ["epoch", "train_loss", "train_iou", "eval_loss", "eval_iou"],
  "monitor_candidates": [
    {"name": "eval_iou", "mode": "max"},
    {"name": "eval_loss", "mode": "min"}
  ],
  "requires_extras": ["vision"],
  "atoms": {
    "dataset_train": {
      "kind": "dataset",
      "name": "voc_seg",
      "params": {"image_size": 256, "split": "train"},
      "shape_contract": {"x": ["B", 3, 256, 256], "y": ["B", 256, 256]}
    },
    "model": {
      "kind": "model",
      "name": "unet",
      "params": {"in_channels": 3, "num_classes": 21, "base_ch": 32}
    },
    "loss": {
      "kind": "loss",
      "name": "cross_entropy",
      "params": {"ignore_index": -1}
    }
  },
  "acceptance": {
    "smoke": {"status": "pass"},
    "contract": {"status": "pass"}
  },
  "known_issues": []
}
```

Rules:

- Metadata must be available without training.
- Optional dependency failures should appear as structured status.
- Metadata should explain what the agent can override.

### `pcq dry-run`

Purpose: show the assembled execution plan.

Command:

```bash
pcq dry-run . --json
```

Required output:

```json
{
  "schema_version": 1,
  "preset": "vision/seg/voc_unet",
  "cfg": {
    "epochs": 50,
    "batch_size": 16,
    "monitor": "eval_iou",
    "mode": "max"
  },
  "atoms": {
    "model": "unet(in_channels=3,num_classes=21,base_ch=32)",
    "loss": "cross_entropy(ignore_index=-1)"
  },
  "expected_metrics": ["epoch", "train_loss", "train_iou", "eval_loss", "eval_iou"],
  "expected_artifacts": [
    "model.pt",
    "config.json",
    "metrics.json",
    "last.ckpt",
    "best.ckpt",
    "manifest.json",
    "run_summary.json"
  ],
  "warnings": []
}
```

Rules:

- Must not train.
- Must not write artifacts except optional cache-free inspection files when
  explicitly requested.
- Must identify missing monitor or artifact contradictions.

### `pcq validate`

Purpose: validate the project and plan before CQ run submission.

Command:

```bash
pcq validate . --json
```

Required output:

```json
{
  "schema_version": 1,
  "status": "fail",
  "checks": [
    {
      "id": "metrics_declared",
      "status": "pass",
      "detail": "all emitted metrics are declared"
    },
    {
      "id": "dataset_shape_contract",
      "status": "fail",
      "severity": "blocking",
      "detail": "dataset emits variable image sizes but default collate requires fixed tensors",
      "suggested_fix": "set dataset resize or provide collate_fn"
    }
  ]
}
```

Validation should include:

- `cq.yaml` parse
- command existence
- `CQ_CONFIG_JSON` compatibility
- declared metric coverage
- monitor key exists
- mode is valid for monitor
- output artifact contract
- recipe smoke acceptance
- real-shape contract acceptance when metadata exists
- loss label compatibility, for example `ignore_index`
- dataset shape compatibility with default DataLoader
- optional dependency availability
- accelerate multi-process artifact/log safety
- project-local atom importability
- generated atom metadata completeness

### `pcq atoms scaffold`

Purpose: create a minimal project-local atom file that an agent can fill in.

Command:

```bash
pcq atoms scaffold model dental_unet --output atoms/models.py --json
```

Required output:

```json
{
  "schema_version": 1,
  "status": "created",
  "files_changed": ["atoms/models.py"],
  "atom": {"kind": "model", "name": "dental_unet"},
  "next_checks": ["pcq atoms validate-local", "pcq validate"]
}
```

Rules:

- Must create ordinary Python files in the project.
- Must not edit `pcq` package internals.
- Must include a `pcq.register_*` call with placeholder metadata.
- Must preserve existing files unless `--force` is passed.
- Generated code should be small, explicit, and easy to review.

### `pcq atoms validate-local`

Purpose: import project-local atoms and validate their metadata/contracts without
running training.

Command:

```bash
pcq atoms validate-local . --json
```

Validation should include:

- module import success
- duplicate atom names
- required param schema
- basic type/range metadata validity
- task presence
- input/output or label contract presence when relevant
- optional smoke contract checks when fake inputs are available

### `pcq init-experiment`

Purpose: scaffold a CQ-runnable ML experiment.

Command:

```bash
pcq init-experiment \
  --style trainer \
  --preset vision/seg/fake_seg_smoke \
  --name dental-seg-baseline \
  --output .
```

Supported styles:

- `trainer`: `pcq.Trainer.from_cfg(cfg)`, preset required.
- `experiment`: generated `pcq.Experiment` subclass, preset not required.
- `script`: framework-agnostic contract script, preset not required.

Script-style experiments are first-class. They are the preferred scaffold when
the agent wants to use Hugging Face Trainer, TabPFN, PyCaret, sklearn, XGBoost,
or custom non-torch training code without introducing a framework adapter.

Expected files:

```text
cq.yaml
train.py
recipes/          # trainer/experiment only
  local.py
cq_atoms.py       # trainer/experiment only
atoms/            # trainer/experiment only
  __init__.py
```

Rules:

- Must not overwrite existing files unless `--force` is given.
- Must emit a manifest of files created.
- Must create the smallest runnable experiment.
- Must include declared metrics.
- Must include standard artifacts.
- Script style should call `pcq.save_all(...)` so the output contract matches
  `Experiment.fit()` as closely as possible.

### `pcq apply-plan`

Purpose: apply a structured experiment plan to a project.

Command:

```bash
pcq apply-plan experiment.plan.json --json
```

Required output:

```json
{
  "schema_version": 1,
  "status": "applied",
  "files_changed": [
    "cq.yaml",
    "train.py",
    "recipes/local.py",
    "atoms/losses.py"
  ],
  "operations": [
    {"op": "set_config", "key": "monitor", "value": "eval_iou"},
    {"op": "register_atom", "kind": "loss", "name": "boundary_dice"},
    {"op": "set_atom", "atom": "loss", "name": "boundary_dice"}
  ],
  "warnings": []
}
```

Rules:

- Must be idempotent when possible.
- Must preserve unrelated user code.
- Must avoid broad reformatting.
- Must reject unknown operations.
- Must write a provenance note into plan output or config metadata.
- Must keep generated atoms in project-local files, not in `pcq` internals.

### `pcq summarize-run`

Purpose: summarize a completed output directory for agent decisions.

Command:

```bash
pcq summarize-run output/ --json
```

Required output:

```json
{
  "schema_version": 1,
  "status": "completed",
  "recipe": "vision/seg/voc_unet",
  "monitor": "eval_iou",
  "mode": "max",
  "best": {
    "epoch": 12,
    "metrics": {
      "eval_iou": 0.714,
      "eval_loss": 0.441
    },
    "checkpoint": "best.ckpt"
  },
  "last": {
    "epoch": 49,
    "metrics": {
      "eval_iou": 0.692,
      "eval_loss": 0.472
    }
  },
  "artifacts": {
    "model": "model.pt",
    "best_checkpoint": "best.ckpt",
    "metrics": "metrics.json",
    "config": "config.json",
    "manifest": "manifest.json"
  },
  "warnings": []
}
```

Rules:

- Must not require training code import.
- Must be based on output artifacts only.
- Must tolerate partial failed runs.
- Must classify failure when possible.

## Python API Surface

The CLI should be backed by public Python APIs.

Target modules:

```text
pcq.agent.inspect
pcq.agent.schema
pcq.agent.plan
pcq.agent.apply
pcq.agent.validate
pcq.agent.summary
```

Possible API:

```python
from pcq.agent import (
    inspect_project,
    recipe_meta,
    dry_run_project,
    validate_project,
    apply_plan,
    summarize_run,
)

state = inspect_project(".")
plan = ExperimentPlan.from_goal(...)
validation = validate_project(".", plan=plan)
result = apply_plan(".", plan)
summary = summarize_run("output")
```

All public return values should be dataclasses or Pydantic-like plain structures
with `.to_dict()` and JSON-safe values. The implementation does not need Pydantic
as a dependency if typed dataclasses plus explicit validators are enough.

## Required Schemas

### `ProjectInspection`

Purpose: full project state for an agent.

Required fields:

```json
{
  "schema_version": 1,
  "project_root": "string",
  "project_type": "pcq|cq|unknown",
  "has_cq_yaml": true,
  "cq_yaml": {},
  "entrypoint": {},
  "recipes": [],
  "outputs": {},
  "warnings": [],
  "errors": []
}
```

### `RecipeSpec`

Purpose: metadata-only recipe contract.

Required fields:

```json
{
  "schema_version": 1,
  "name": "string",
  "task": "classification|segmentation|text_classification|regression|custom",
  "description": "string",
  "declared_metrics": [],
  "monitor_candidates": [],
  "requires_extras": [],
  "atoms": {},
  "defaults": {},
  "acceptance": {},
  "known_issues": []
}
```

Important distinction:

- `RecipeSpec` is metadata.
- recipe factory is executable code.
- agents should inspect `RecipeSpec` first and execute factory only during
  validation or training.

### `AtomRef`

Purpose: metadata representation of an atom.

Required fields:

```json
{
  "kind": "model|dataset|loss|optimizer|scheduler|metric|collate",
  "name": "string",
  "params": {},
  "overridable": true,
  "requires_extras": [],
  "shape_contract": {},
  "label_contract": {},
  "resource_estimate": {}
}
```

Example:

```json
{
  "kind": "loss",
  "name": "cross_entropy",
  "params": {"ignore_index": -1},
  "label_contract": {
    "target_dtype": "int64",
    "valid_range": [0, 20],
    "ignore_index": -1
  }
}
```

Detailed atom registration, custom atom, and metadata-aware registry design is
specified in [Atom Registry And Custom Atoms](ATOM_REGISTRY.md).

### `ExperimentPlan`

Purpose: structured edit intent.

Required fields:

```json
{
  "schema_version": 1,
  "id": "exp-001",
  "intent": "improve_eval_iou",
  "base": {
    "preset": "vision/seg/voc_unet"
  },
  "target": {
    "metric": "eval_iou",
    "mode": "max"
  },
  "changes": [],
  "validation_policy": {
    "run_smoke": true,
    "run_contract": true,
    "allow_network": false
  }
}
```

Supported change operations:

```json
[
  {"op": "set_config", "key": "epochs", "value": 80},
  {"op": "set_config", "key": "monitor", "value": "eval_iou"},
  {"op": "set_config", "key": "mode", "value": "max"},
  {
    "op": "set_atom",
    "atom": "loss",
    "name": "cross_entropy",
    "params": {"ignore_index": -1}
  },
  {
    "op": "set_atom",
    "atom": "scheduler",
    "name": "cosine",
    "params": {"T_max": 80, "warmup": 1000}
  },
  {
    "op": "set_dataset_transform",
    "split": "train",
    "transform": "resize",
    "params": {"size": [256, 256]}
  },
  {
    "op": "set_smoke_override",
    "atom": "dataset_train",
    "name": "fake_seg",
    "params": {"num_samples": 16, "image_size": 64}
  }
]
```

Rules:

- Unknown ops fail validation.
- Unknown atom names fail validation.
- Config keys must be known or explicitly marked as custom.
- Changes must be serializable.
- Generated patches must be reviewable.

### `ValidationReport`

Purpose: gate before running expensive CQ jobs.

Required fields:

```json
{
  "schema_version": 1,
  "status": "pass|warn|fail",
  "checks": [],
  "blocking_count": 0,
  "warning_count": 0
}
```

Check format:

```json
{
  "id": "loss_label_contract",
  "status": "pass|warn|fail|skip",
  "severity": "blocking|warning|info",
  "detail": "string",
  "evidence": {},
  "suggested_fix": "string"
}
```

### `RunSummary`

Purpose: standard result object for agent decisions.

Required fields:

```json
{
  "schema_version": 1,
  "status": "completed|failed|partial|unknown",
  "recipe": "string|null",
  "target": {"metric": "string|null", "mode": "min|max|null"},
  "best": {},
  "last": {},
  "artifacts": {},
  "provenance": {},
  "warnings": [],
  "failure": null
}
```

Failure example:

```json
{
  "status": "failed",
  "failure": {
    "category": "dataset_shape",
    "message": "default collate cannot stack variable image sizes",
    "suggested_fix": "resize dataset outputs or provide collate_fn"
  }
}
```

## Recipe Metadata Refactor

Current recipes return executable dictionaries. This is simple but not ideal for
agents because metadata and execution are mixed.

Current:

```python
def voc_unet() -> dict:
    return {
        "model": pcq.examples.models.unet(...),
        "dataset_train": lambda _: datasets.voc_seg(...),
        "loss": loss.cross_entropy(),
        "metrics": [...]
    }
```

Target:

```python
from pcq.agent.schema import AtomRef, RecipeSpec


SPEC = RecipeSpec(
    name="vision/seg/voc_unet",
    task="segmentation",
    description="Pascal VOC segmentation with UNet",
    metrics=["epoch", "train_loss", "train_iou", "eval_loss", "eval_iou"],
    monitor_candidates=[
        {"name": "eval_iou", "mode": "max"},
        {"name": "eval_loss", "mode": "min"},
    ],
    atoms={
        "model": AtomRef("model", "unet", {"num_classes": 21}),
        "loss": AtomRef("loss", "cross_entropy", {"ignore_index": -1}),
        "dataset_train": AtomRef(
            "dataset",
            "voc_seg",
            {"split": "train", "image_size": 256},
        ),
    },
)


def voc_unet() -> dict:
    return SPEC.build()
```

Benefits:

- agents can inspect without instantiating heavy objects
- optional dependency requirements are explicit
- shape and label contracts can be validated
- patch plans can target atom refs
- factory stays available for normal users

Migration should support both formats:

1. If `SPEC` exists, use it.
2. Else call recipe function and infer best-effort metadata.
3. Warn when metadata is inferred and incomplete.

The atom side of this migration is covered separately in
[Atom Registry And Custom Atoms](ATOM_REGISTRY.md). In short, recipes should move
from executable objects to `AtomRef` entries wherever the agent needs to inspect
or patch the atom.

## Validation Gates

Validation must be concrete enough that the CQ service can trust it.

### Gate 1: Static Project Contract

Checks:

- `cq.yaml` exists
- `cmd` exists and points to a runnable entrypoint
- metrics list exists or explicit warning is emitted
- artifact glob includes output directory
- config can be converted to `CQ_CONFIG_JSON`
- output directory is not outside workspace unless explicitly allowed

### Gate 2: Recipe Contract

Checks:

- recipe import succeeds
- metadata exists or can be inferred
- required atoms exist
- task is known or custom
- declared metrics match task defaults
- monitor key is in declared metrics
- mode is compatible with monitor
- `smoke_safe` and `smoke_overrides` are coherent

### Gate 3: Atom Contract

Checks:

- dataset output shape is compatible with model input
- loss target contract is compatible with dataset labels
- metric shape contract is compatible with model outputs
- optimizer and scheduler factories accept expected inputs
- optional dependencies are present or marked missing
- resource estimate is within configured bounds when available

### Gate 4: Runtime Safety Contract

Checks:

- accelerate multi-process logging is main-process guarded
- artifact writes are main-process guarded
- checkpoint writes are atomic or main-process only
- resume checkpoint exists when explicit
- `resume: true` behavior is deterministic
- `best.ckpt` expectation matches monitor availability

### Gate 5: Acceptance Contract

Two acceptance levels:

```text
smoke_acceptance
  fast, fake data allowed, no network, CI friendly

contract_acceptance
  validates real-shape data path, label semantics, collate behavior, monitor,
  artifacts, and known optional dependency status
```

The agent should prefer recipes with both passing. It may use smoke-only recipes
for exploration if the plan explicitly marks the risk.

## CQ Service Integration Loop

The service loop should be deterministic and auditable.

```text
1. Inspect
   pcq inspect . --json

2. Summarize previous outputs
   pcq summarize-run output/ --json

3. Ask LLM for an ExperimentPlan
   Inputs: user goal, inspection JSON, recipe metadata, run summary
   Output: ExperimentPlan JSON

4. Validate plan
   pcq validate . --plan experiment.plan.json --json

5. Apply plan
   pcq apply-plan experiment.plan.json --json

6. Re-validate project
   pcq validate . --json

7. Submit CQ run
   CQ service runs cq.yaml command

8. Summarize result
   pcq summarize-run output/ --json

9. Decide next step
   stop, launch next plan, or ask user
```

The LLM should not be asked to reason from raw code alone when structured
inspection is available.

## Agent Prompt Contract

When CQ service asks an LLM to propose a plan, it should provide:

- user goal
- project inspection JSON
- selected recipe metadata
- previous run summary
- allowed operation list
- validation policy
- budget constraints
- explicit output schema

The LLM should return only `ExperimentPlan` JSON, not prose, for the apply path.

Example instruction:

```text
Return an ExperimentPlan JSON object only. Use only operations from
allowed_operations. Do not invent atom names that are absent from registry.
If a required atom is absent, propose a project-local atom file with explicit
metadata. Prefer config/atom changes over free-form training-loop edits. The
target metric is eval_iou and mode is max.
```

## Safety Model

### Bounded Mutations

`apply-plan` should only modify known locations:

- `cq.yaml`
- generated `train.py`
- generated project-local recipe file
- project-local atom files under `atoms/` or `cq_atoms.py`
- optional `recipes/local.py`

It should not edit arbitrary source files unless the plan includes an explicit
`custom_code_patch` operation, which should be disabled by default.

### Idempotence

Applying the same plan twice should either:

- produce no changes on the second run, or
- fail with a clear already-applied status

### Auditability

Every applied plan should leave provenance:

```json
{
  "_pcq_plan_id": "exp-001",
  "_pcq_plan_intent": "improve_eval_iou",
  "_pcq_plan_changes": [...]
}
```

This can be saved in `config.json`, `run_summary.json`, or a plan artifact.

### Permission Boundaries

`pcq` should not bypass CQ service permission review. It should only describe
files it intends to change and perform local deterministic edits.

## Result Semantics For Agents

Agents need more than raw metrics. They need interpretation-ready summaries.

The target completion object is `run_record.json`, described in
[RunRecord Standard](RUN_RECORD.md). Until that exists, `run_summary.json`,
`manifest.json`, `metrics.json`, and `config.json` are partial RunRecord
components.

`run_summary.json` should include:

- status
- recipe
- plan id
- target metric
- best epoch
- last epoch
- delta from previous run when provided
- artifact paths
- provenance sufficient to connect the run to its plan and source changes
- warnings
- failure category
- suggested next actions

`run_record.json` should include:

- execution contract
- source snapshot or patch identity
- dependency/runtime snapshot
- dataset/input identity
- structured metric declarations
- artifact manifest with checksums when available
- agent plan provenance
- validation report path and status

Suggested failure categories:

```text
config_error
missing_dependency
dataset_missing
dataset_shape
label_contract
loss_contract
metric_contract
oom
nan_loss
timeout
distributed_write_race
unknown_exception
```

The CQ service can map these categories to next actions.

## Implementation Phases

### Phase A: Fix Current Contract Gaps

Purpose: make current v1.4 reliable enough for agents to trust.

Required changes:

- add `ignore_index` to `pcq.loss.cross_entropy`
- update VOC recipe to use `ignore_index=-1`
- make VOC dataset shape fixed or add collate support
- guard accelerate logging and artifact writes to main process
- make `pcq.log()` read `_metrics_declared` from `CQ_CONFIG_JSON`
- make acceptance expectation for `best.ckpt` explicit and consistent

Exit criteria:

- all current review findings are resolved
- acceptance fails when monitor cannot create expected `best.ckpt`
- real-shape VOC contract does not fail on label or collate issues

### Phase B: JSON CLI MVP

Purpose: give CQ service stable machine interfaces.

Add commands:

```bash
pcq inspect . --json
pcq recipe-meta <preset> --json
pcq dry-run . --json
pcq validate . --json
pcq summarize-run output/ --json
```

Exit criteria:

- commands return valid JSON
- no command trains unless explicitly named as smoke run
- CI snapshots validate schema shape
- CQ service can call commands without importing Python modules directly

### Phase C: Typed Metadata Schema

Purpose: separate metadata from executable recipe factories.

Add:

- `RecipeSpec`
- `AtomRef`
- `ShapeContract`
- `LabelContract`
- `MetricContract`
- metadata fallback for old dict recipes

Exit criteria:

- all built-in recipes expose `RecipeSpec`
- recipe inspection does not instantiate heavy models
- optional dependency status is visible in metadata
- agent can list overridable atoms and valid params

### Phase D: Plan Schema And Apply

Purpose: make experiment modification structured.

Add:

- `ExperimentPlan`
- plan validation
- `apply-plan`
- safe `cq.yaml` update support
- safe generated `train.py` update support
- project-local recipe generation
- project-local atom generation

Exit criteria:

- plan can change monitor/mode/epochs/batch/lr
- plan can replace loss/optimizer/scheduler atoms
- plan can add a project-local atom with explicit metadata
- plan can add smoke overrides
- apply is idempotent for common changes
- apply reports changed files and operations

### Phase E: Run Summary And Comparison

Purpose: make outputs agent-decision-ready.

Add:

- `run_summary.json`
- summary CLI
- failure classifier
- previous run comparison helper

Exit criteria:

- agent can read best/last metrics without parsing raw history
- partial failed outputs produce structured summaries
- CQ service can compare two summaries by target metric

### Phase F: Agent Optimization Loop Support

Purpose: support iterative CQ service automation.

Add:

- experiment lineage fields
- plan id fields
- baseline id fields
- previous run references
- recommended next action hints

Exit criteria:

- CQ service can run multiple experiments and reconstruct lineage
- agent can explain which plan produced which result
- repeated loop does not require raw code inspection for normal changes

## Concrete v1.5 Scope Recommendation

The next release should not try to solve the full loop. It should ship the
minimum usable agent-operable substrate.

Recommended v1.5:

- fix current v1.4 contract gaps
- add `pcq inspect --json`
- add `pcq recipe-meta --json`
- add `pcq validate --json`
- add `pcq summarize-run --json`
- add `run_summary.json`
- define dataclass schemas

Do not include `apply-plan` yet unless the schemas are stable. Inspection and
validation should come before mutation.

## Concrete v1.6 Scope Recommendation

Recommended v1.6:

- add `ExperimentPlan`
- add `pcq apply-plan`
- add `init-experiment`
- migrate built-in recipes to `RecipeSpec`
- add project-local recipe scaffolding
- add project-local atom discovery and scaffolding

This is the release where agents become able to safely change experiments.

## Concrete v2 Scope Recommendation

Recommended v2:

- external recipe packages
- CQ service native integration
- run comparison across CQ jobs
- optional Drive input helpers
- contract-script examples for repeated third-party framework patterns
- `run_record.json` as the primary CQ service run object
- agent optimization loop helpers

This is where agents become able to run iterative improvement workflows at
service level.

## Test Strategy

### Unit Tests

Cover:

- schema serialization
- schema validation
- atom refs
- recipe specs
- plan operations
- validation report generation
- run summary parsing

### Golden JSON Tests

Every CLI command should have snapshot-style golden JSON tests.

Examples:

```text
tests/golden/inspect_fake_smoke.json
tests/golden/recipe_meta_voc_unet.json
tests/golden/validate_missing_monitor.json
tests/golden/summarize_failed_run.json
```

### Mutation Tests

For `apply-plan`:

- apply once changes expected files
- apply twice is idempotent
- unknown op fails
- unrelated code is preserved
- invalid atom name fails
- invalid monitor fails before writing

### Service Simulation Tests

Simulate the CQ service loop locally:

```text
inspect -> plan fixture -> validate -> apply -> validate -> train smoke -> summarize
```

This should run without network.

## Documentation Strategy

Docs should be split by audience:

- `README.md`: quick user path
- `docs/SPEC.md`: runtime and API contract
- `docs/VISION.md`: product direction
- `docs/AGENT_OPERABILITY.md`: CQ service and agent-operable transition
- `docs/CQ_YAML_RUNTIME_CONTRACT.md`: cq.yaml, config, metrics, input, and
  output resolution rules
- `docs/WORKER_EXECUTION_FLOW.md`: CQ worker execution and artifact collection
- `docs/AGENT_OPERATING_GUIDE.md`: step-by-step coding agent workflow
- `docs/CQ_MCP_SPEC.md`: managed-service MCP tool surface
- `docs/AGENT_ACCEPTANCE_CHECKLIST.md`: release gates for agent-operable
  behavior
- `docs/ATOM_REGISTRY.md`: metadata-aware atom registry and custom atoms
- `docs/RUN_RECORD.md`: reproducible run record and completion semantics
- `templates/AGENTS.pcq.md`: short project-local agent rules
- `skills/pcq/SKILL.md`: reusable agent workflow instructions
- future `docs/CLI.md`: exact CLI contract
- future `docs/SCHEMAS.md`: JSON schemas

The agent docs should include JSON examples because CQ service integration will
depend on them.

## Risks

### Risk: Metadata Becomes Stale

Mitigation:

- recipe acceptance checks metadata against emitted metrics and artifacts
- contract acceptance validates shape and label assumptions
- schemas require version numbers

### Risk: apply-plan Becomes A Code Generator That Breaks User Code

Mitigation:

- restrict generated file regions
- avoid arbitrary edits by default
- require explicit custom patch operation for free-form code
- report exact file changes

### Risk: CLI JSON Becomes Unstable

Mitigation:

- schema_version in every JSON payload
- golden tests
- additive changes by default
- deprecation window for removals

### Risk: CQ Service Duplicates pcq Logic

Mitigation:

- keep validation and summarization in `pcq`
- CQ service orchestrates and decides, but does not reimplement schemas

### Risk: The Library Becomes Too Heavy

Mitigation:

- lazy imports
- metadata-only inspection path
- optional extras
- no LLM runtime in package

## Success Criteria

`pcq` becomes agent-operable when:

- CQ service can inspect a new ML project through JSON without reading raw code
- CQ service can validate whether a recipe is safe to run
- CQ service can generate a plan and apply it without free-form editing
- every run produces `run_summary.json`
- failed runs produce categorized failure summaries
- agents can compare runs using structured metrics
- common changes require no manual Python patching
- core user-facing APIs remain simple

## Final Target

The final operating model should feel like this:

```text
cq.yaml + data + RecipeSpec
  -> pcq inspect
  -> ExperimentPlan
  -> pcq validate
  -> pcq apply-plan
  -> CQ run
  -> run_summary.json
  -> next ExperimentPlan
```

That is the agent-operable version of `pcq`: a standard production library for
creating, modifying, validating, running, and evaluating CQ ML experiments.
