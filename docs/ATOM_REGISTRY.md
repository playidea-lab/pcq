# Atom Registry And Custom Atoms

## Purpose

This document defines how `pcq` should evolve its atom system so agents can use
custom models, datasets, losses, optimizers, schedulers, and metrics safely.

The short answer is:

```text
All atom categories should become metadata-aware contract registries.
```

Metrics already point in the right direction: a metric has a name, callable,
task meaning, input contract, output direction, and monitor semantics. Models,
losses, schedulers, optimizers, datasets, transforms, and collate functions need
the same kind of registry treatment.

The registry is not meant to become a large built-in model/loss/scheduler zoo.
`pcq` cannot know every model, dataset, loss, and experimental idea a user or
agent may need. The registry exists so any executable atom, including
project-local and agent-generated atoms, can declare a contract and be validated
before it is used in a CQ run.

The current registry shape is mostly:

```text
name -> callable
```

The agent-operable target is:

```text
name -> AtomSpec(factory, metadata, param schema, contracts)
```

Humans can keep using normal Python functions. Agents should use structured
`AtomRef` objects and JSON-safe metadata.

## Contract Runtime, Not Catalog

The product direction is:

```text
pcq is a contract runtime for ML experiment atoms,
not a catalog that must implement every ML component upstream.
```

Built-in atoms should stay small and practical:

- smoke datasets and tiny models for acceptance tests
- common baseline components such as `cross_entropy`, `adamw`, and `cosine`
- tutorial recipes that demonstrate the contract
- reference implementations that show how to write custom atoms

The main extension path should be project-local atoms:

```text
project goal
  -> agent writes or edits project-local atom code
  -> atom declares metadata and contracts
  -> pcq validates atom refs and recipe compatibility
  -> CQ runs the experiment
  -> artifacts and metrics are summarized
```

This avoids the wrong maintenance model where `pcq` must keep adding every
possible segmentation head, contrastive loss, scheduler, dataset transform, or
research-specific metric. New research code belongs in the user project. `pcq`
standardizes how that code is named, described, validated, resolved, and
reported.

Not every integration needs to start as an atom. If a third-party framework is
best driven as a plain script, the project can use the low-level CQ helpers
directly and still be valid:

```text
pcq.config()
third-party library code
pcq.log(...)
metrics.json / manifest.json / run_summary.json
```

Atoms are useful when the component should be selected, swapped, validated, or
reused by an agent. Contract scripts are useful when the framework owns the
training flow and the main requirement is to normalize inputs and outputs.

## Atom Sources

An agent-operable project should treat atoms as coming from three sources.

### Built-In Atoms

Built-in atoms are **reference examples** for the contract, not a maintained
model zoo. They exist for smoke tests, onboarding examples, and contract
verification baselines. They must be stable and covered by acceptance tests,
but they must not expand into a broad model zoo.

In v2.4 every built-in atom is marked with `role="reference_example"` in its
`AtomSpec`, distinguishing it from project / generated / external atoms which
default to `role="user"`. In v2.7 model, dataset, optimizer, and scheduler
reference implementations live in `pcq.examples.{models,datasets,optim,sched}`;
`pcq.{models,datasets,optim,sched}` are compatibility facades that re-export the
same factory functions.

### Project Atoms

Project atoms live inside the current repository, for example under
`cq_atoms.py` or `atoms/*.py`. They are the primary extension mechanism for real
work. A CQ service agent should prefer adding a project atom over patching
`pcq` internals when a user asks for a new model, loss, transform, or metric.

### Generated Atoms

Generated atoms are project atoms authored by an agent during an experiment
loop. They are acceptable when they declare contracts, pass static validation,
and pass a small smoke check. Generated atoms should be ordinary Python files in
the project, not hidden runtime objects.

All three sources should end up in the same registry view:

```text
registry = builtin atoms + project atoms + generated project atoms
```

The source should be visible in metadata so agents can decide whether an atom is
safe to edit:

```json
{
  "kind": "loss",
  "name": "boundary_dice",
  "source": "project",
  "module": "atoms.losses",
  "metadata_status": "explicit"
}
```

## Why This Matters For Agents

An agent does not only need to know that an atom exists. It needs to answer:

- What task is this atom for?
- Which parameters can be set?
- Which parameters are required?
- What input shape does it expect?
- What output shape does it produce?
- What label dtype/range does the loss expect?
- Which metrics are natural monitor candidates?
- Does it need optional extras?
- Is it safe for smoke tests?
- Can it run without network?
- Can it be substituted by fake data?

Without this metadata, the agent must inspect code and guess. That works for
small scripts but does not scale to automatic CQ experiment generation.

## Human API vs Agent API

The design should keep two paths.

### Human API

Humans should be able to write direct Python:

```python
import pcq

model = pcq.examples.models.unet(in_channels=3, num_classes=21)
loss = pcq.loss.cross_entropy(ignore_index=-1)
sched = pcq.examples.sched.cosine(optimizer, T_max=50, warmup=500)
```

This path should stay simple and idiomatic.

### Agent API

Agents should use refs and specs:

```python
import pcq

model = pcq.model_ref("unet", {"in_channels": 3, "num_classes": 21})
loss = pcq.loss_ref("cross_entropy", {"ignore_index": -1})
sched = pcq.sched_ref("cosine", {"T_max": 50, "warmup": 500})
```

The refs are serializable and inspectable:

```json
{
  "kind": "model",
  "name": "unet",
  "params": {"in_channels": 3, "num_classes": 21}
}
```

At execution time, refs are resolved through the registry into Python objects.

## Core Types

### `AtomSpec`

`AtomSpec` is the registry entry.

Suggested dataclass:

```python
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AtomSpec:
    kind: str
    name: str
    factory: Callable[..., Any]
    params: dict[str, "ParamSpec"] = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    input_contract: dict[str, Any] = field(default_factory=dict)
    output_contract: dict[str, Any] = field(default_factory=dict)
    label_contract: dict[str, Any] = field(default_factory=dict)
    metric_contract: dict[str, Any] = field(default_factory=dict)
    requires_extras: list[str] = field(default_factory=list)
    smoke_safe: bool = True
    description: str = ""
    source: str = "builtin"       # builtin | project | generated | external
    module: str = ""
```

Required behavior:

- `factory` builds the actual Python object or callable.
- metadata is JSON-safe.
- `params` declares the accepted parameter surface.
- contracts help validation before training.
- `requires_extras` lets agents reason about dependency availability.
- `source` and `module` let agents distinguish maintained built-ins from
  project-local code that may be safe to edit.

### `ParamSpec`

`ParamSpec` describes one configurable parameter.

Suggested shape:

```python
@dataclass
class ParamSpec:
    type: str
    default: object | None = None
    required: bool = False
    choices: list[object] | None = None
    min: float | None = None
    max: float | None = None
    description: str = ""
```

Example:

```json
{
  "type": "int",
  "default": 21,
  "required": false,
  "min": 1,
  "description": "number of output classes"
}
```

### `AtomRef`

`AtomRef` is what recipes, plans, and agents use.

Suggested dataclass:

```python
@dataclass
class AtomRef:
    kind: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
```

JSON shape:

```json
{
  "kind": "loss",
  "name": "cross_entropy",
  "params": {"ignore_index": -1}
}
```

Resolution:

```python
obj = pcq.registry.losses.build_ref(AtomRef("loss", "cross_entropy", {"ignore_index": -1}))
```

Rules:

- unknown atom names fail validation
- unknown params fail validation unless explicitly allowed
- required params must be present
- param values must match declared type/choices/range
- resolved objects are created only when executing or validating runtime behavior

## Registry API

Current user-facing registration should be extended, not replaced.

### Target Registration Form

```python
pcq.register_model(
    "unet",
    factory=lambda in_channels=3, num_classes=21, base_ch=32: UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_ch=base_ch,
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "in_channels": {"type": "int", "default": 3, "min": 1},
            "num_classes": {"type": "int", "default": 21, "min": 1},
            "base_ch": {"type": "int", "default": 32, "min": 1},
        },
        "input_contract": {"x": ["B", "C", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes", "H", "W"]},
        "requires_extras": [],
        "smoke_safe": True,
    },
)
```

The old form should remain valid:

```python
pcq.register_model("unet", lambda: UNet())
```

When old form is used, `pcq` should create a best-effort `AtomSpec` with
minimal metadata and mark it as incomplete:

```json
{
  "metadata_status": "inferred",
  "warnings": ["missing param schema", "missing shape contract"]
}
```

### Registry Object Behavior

Each registry should support:

```python
pcq.registry.models.register(...)
pcq.registry.models.get("unet")          # AtomSpec
pcq.registry.models.build("unet", ...)   # object
pcq.registry.models.meta("unet")         # JSON-safe metadata
pcq.registry.models.list()               # names
pcq.registry.models.validate_ref(ref)    # ValidationReport checks
pcq.registry.models.build_ref(ref)       # object from AtomRef
```

Project atom discovery should be exposed as a separate operation, not hidden in
normal package import:

```python
pcq.registry.load_project_atoms(".")      # imports cq_atoms.py / atoms/*.py
pcq.registry.list_sources()               # builtin/project/generated summary
```

CLI commands should call the same discovery path before listing or validating
local atoms.

Existing shortcuts can remain:

```python
pcq.register_model(...)
pcq.register_loss(...)
pcq.register_sched(...)
pcq.register_metric(...)
```

## Built-In Atom Categories

This section documents the categories every registry should support. The
examples use built-in-looking names because they are easy to read, but the same
contract applies to project-local and generated atoms. The goal is category
consistency, not upstream ownership of every implementation.

### Models

Model atoms create `torch.nn.Module` instances.

Example registration:

```python
pcq.register_model(
    "small_cnn",
    factory=lambda in_channels=3, num_classes=10: SmallCNN(in_channels, num_classes),
    meta={
        "tasks": ["classification"],
        "params": {
            "in_channels": {"type": "int", "default": 3},
            "num_classes": {"type": "int", "default": 10},
        },
        "input_contract": {"x": ["B", "in_channels", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes"]},
    },
)
```

Custom model registration:

```python
import pcq
import torch.nn as nn


class DentalUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        ...

    def forward(self, x):
        ...


pcq.register_model(
    "dental_unet",
    factory=lambda in_channels=3, num_classes=2: DentalUNet(
        in_channels=in_channels,
        num_classes=num_classes,
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "in_channels": {"type": "int", "default": 3},
            "num_classes": {"type": "int", "default": 2},
        },
        "input_contract": {"x": ["B", "in_channels", "H", "W"]},
        "output_contract": {"logits": ["B", "num_classes", "H", "W"]},
    },
)
```

Recipe usage:

```python
"model": pcq.model_ref("dental_unet", {"in_channels": 3, "num_classes": 2})
```

### Losses

Loss atoms create `torch.nn.Module` instances or callables.

Example registration:

```python
pcq.register_loss(
    "cross_entropy",
    factory=lambda ignore_index=-100: torch.nn.CrossEntropyLoss(
        ignore_index=ignore_index
    ),
    meta={
        "tasks": ["classification", "segmentation"],
        "params": {
            "ignore_index": {"type": "int", "default": -100},
        },
        "input_contract": {
            "logits": ["B", "C", "..."],
            "target": ["B", "..."],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "C-1"],
            "ignore_index_param": "ignore_index",
        },
    },
)
```

Custom loss registration:

```python
import pcq
import torch
import torch.nn as nn


class BoundaryDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...


pcq.register_loss(
    "boundary_dice",
    factory=lambda smooth=1.0: BoundaryDiceLoss(smooth=smooth),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "smooth": {"type": "float", "default": 1.0, "min": 0.0},
        },
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, "C-1"],
        },
    },
)
```

Recipe usage:

```python
"loss": pcq.loss_ref("boundary_dice", {"smooth": 1.0})
```

### Optimizers

Optimizer atoms create `torch.optim.Optimizer` instances and require model
parameters at build time.

Example registration:

```python
pcq.register_optim(
    "adamw",
    factory=lambda params, lr=1e-3, weight_decay=0.0: torch.optim.AdamW(
        params, lr=lr, weight_decay=weight_decay
    ),
    meta={
        "params": {
            "lr": {"type": "float", "default": 1e-3, "min": 0.0},
            "weight_decay": {"type": "float", "default": 0.0, "min": 0.0},
        },
        "requires_model_params": True,
    },
)
```

Recipe usage:

```python
"optim": pcq.optim_ref("adamw", {"lr": 3e-4, "weight_decay": 1e-4})
```

At recipe build time this becomes:

```python
"optim_factory": lambda params: pcq.registry.optims.build("adamw", params, lr=3e-4)
```

### Schedulers

Scheduler atoms require an optimizer at build time.

Example:

```python
pcq.register_sched(
    "cosine",
    factory=lambda optimizer, T_max, warmup=0: cosine(
        optimizer, T_max=T_max, warmup=warmup
    ),
    meta={
        "params": {
            "T_max": {"type": "int", "required": True, "min": 1},
            "warmup": {"type": "int", "default": 0, "min": 0},
        },
        "requires_optimizer": True,
    },
)
```

Recipe usage:

```python
"sched": pcq.sched_ref("cosine", {"T_max": 50, "warmup": 500})
```

### Datasets

Dataset atoms create `torch.utils.data.Dataset` instances.

Example:

```python
pcq.register_dataset(
    "voc_seg",
    factory=lambda root, split="train", image_size=256, download=False: voc_seg(
        root=root,
        split=split,
        image_size=image_size,
        download=download,
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "root": {"type": "path", "required": True},
            "split": {"type": "str", "default": "train", "choices": ["train", "val"]},
            "image_size": {"type": "int", "default": 256},
            "download": {"type": "bool", "default": False},
        },
        "output_contract": {
            "x": ["C", "image_size", "image_size"],
            "y": ["image_size", "image_size"],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, 20],
            "ignore_index": -1,
        },
        "requires_extras": ["vision"],
        "network_required_if": {"download": True},
    },
)
```

Custom dataset registration:

```python
import pcq
from torch.utils.data import Dataset


class DentalSegDataset(Dataset):
    ...


pcq.register_dataset(
    "dental_seg",
    factory=lambda root, split="train", image_size=256: DentalSegDataset(
        root=root,
        split=split,
        image_size=image_size,
    ),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "root": {"type": "path", "required": True},
            "split": {"type": "str", "default": "train"},
            "image_size": {"type": "int", "default": 256},
        },
        "output_contract": {
            "x": ["C", "image_size", "image_size"],
            "y": ["image_size", "image_size"],
        },
        "label_contract": {
            "target_dtype": "int64",
            "valid_range": [0, 1],
            "ignore_index": -1,
        },
    },
)
```

Recipe usage:

```python
"dataset_train": pcq.dataset_ref(
    "dental_seg", {"root": "data", "split": "train", "image_size": 256}
)
```

### Metrics

Metric atoms are callables that produce scalar values.

Example:

```python
pcq.register_metric(
    "iou",
    factory=pcq.metric.iou,
    meta={
        "tasks": ["segmentation"],
        "mode": "max",
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
        "label_contract": {
            "ignore_index": -1,
        },
    },
)
```

Custom metric registration:

```python
def boundary_iou(logits, target):
    ...


pcq.register_metric(
    "boundary_iou",
    boundary_iou,
    meta={
        "tasks": ["segmentation"],
        "mode": "max",
        "input_contract": {
            "logits": ["B", "C", "H", "W"],
            "target": ["B", "H", "W"],
        },
    },
)
```

Recipe metadata should expose monitor candidates:

```python
"monitor_candidates": [
    {"name": "eval_boundary_iou", "mode": "max"},
    {"name": "eval_loss", "mode": "min"},
]
```

### Collate Functions

Collate atoms become important when datasets produce variable-sized samples.

Example:

```python
pcq.register_collate(
    "pad_to_max",
    factory=lambda ignore_index=-1: pad_to_max_collate(ignore_index=ignore_index),
    meta={
        "tasks": ["segmentation"],
        "params": {
            "ignore_index": {"type": "int", "default": -1},
        },
        "solves": ["variable_image_size"],
    },
)
```

Recipe usage:

```python
"collate": pcq.collate_ref("pad_to_max", {"ignore_index": -1})
```

This requires `Experiment` or `Trainer` to support passing a collate function
into `DataLoader`.

## RecipeSpec With AtomRefs

Agent-operable recipes should prefer refs over executable objects.

Target:

```python
from pcq.agent.schema import RecipeSpec


SPEC = RecipeSpec(
    name="vision/seg/dental_unet",
    task="segmentation",
    description="Dental binary segmentation with UNet",
    atoms={
        "dataset_train": pcq.dataset_ref(
            "dental_seg", {"root": "data", "split": "train", "image_size": 256}
        ),
        "dataset_eval": pcq.dataset_ref(
            "dental_seg", {"root": "data", "split": "val", "image_size": 256}
        ),
        "model": pcq.model_ref("dental_unet", {"num_classes": 2}),
        "loss": pcq.loss_ref("cross_entropy", {"ignore_index": -1}),
        "optim": pcq.optim_ref("adamw", {"lr": 3e-4, "weight_decay": 1e-4}),
        "sched": pcq.sched_ref("cosine", {"T_max": 80, "warmup": 1000}),
        "metrics": [
            pcq.metric_ref("iou", {"ignore_index": -1}),
            pcq.metric_ref("pixel_accuracy", {"ignore_index": -1}),
        ],
    },
    defaults={
        "epochs": 80,
        "batch_size": 8,
        "monitor": "eval_iou",
        "mode": "max",
    },
)
```

Factory compatibility:

```python
def dental_unet() -> dict:
    return SPEC.build()
```

`SPEC.build()` resolves refs into the current executable dictionary format:

```python
{
    "task": "segmentation",
    "dataset_train": lambda split: ...,
    "model": DentalUNet(...),
    "loss": CrossEntropyLoss(ignore_index=-1),
    "optim_factory": lambda params: AdamW(...),
    "sched_factory": lambda optimizer: ...,
    "metrics": [...],
    "epochs": 80,
    "batch_size": 8,
}
```

## Validation Implications

Metadata-aware atoms allow validation before expensive runs.

### Model-Dataset Compatibility

Check:

- dataset output channels match model input channels
- classification datasets produce `(x, y)` where y is class id
- segmentation datasets produce image/mask pairs
- variable shapes have resize or collate handling

### Loss-Label Compatibility

Check:

- target dtype matches loss requirements
- target range matches number of classes
- dataset ignore index matches loss ignore index
- loss supports output shape from model

Example failure:

```json
{
  "id": "loss_label_contract",
  "status": "fail",
  "severity": "blocking",
  "detail": "dataset uses ignore_index=-1 but loss cross_entropy has ignore_index=-100",
  "suggested_fix": "set loss cross_entropy ignore_index to -1"
}
```

### Metric-Monitor Compatibility

Check:

- monitor metric is declared
- monitor mode is known
- metric input shape matches model output and target
- metric ignore index matches dataset

### Scheduler-Optimizer Compatibility

Check:

- scheduler receives optimizer
- required scheduler params are set
- scheduler horizon is compatible with epochs when known

## Custom Atom Registration In User Projects

Project-local custom atoms should live in predictable files, for example:

```text
project/
  cq.yaml
  train.py
  recipes/
    local.py
  atoms/
    models.py
    losses.py
    datasets.py
    metrics.py
```

`train.py` should import the project-local atom modules before resolving recipes:

```python
import pcq
import atoms.models
import atoms.losses
import atoms.datasets
import atoms.metrics


cfg = pcq.config()
pcq.Trainer(preset=cfg["preset"], cfg=cfg).fit()
```

The agent should prefer adding or editing these local atom files instead of
patching library internals.

### Recommended Project Loader

Projects should have one predictable import point for local atoms:

```text
project/
  cq.yaml
  train.py
  cq_atoms.py
  atoms/
    models.py
    losses.py
    datasets.py
    metrics.py
```

`cq_atoms.py` can import all atom modules:

```python
# cq_atoms.py
import atoms.datasets  # noqa: F401
import atoms.losses    # noqa: F401
import atoms.metrics   # noqa: F401
import atoms.models    # noqa: F401
```

Then `train.py` has one stable hook:

```python
import pcq
import cq_atoms  # noqa: F401


cfg = pcq.config()
pcq.Trainer(preset=cfg["preset"], cfg=cfg).fit()
```

This gives the CQ service agent a safe place to add new atom files without
rewriting the training loop.

### Agent-Generated Atom Workflow

When the CQ service needs a new component, the agent should follow this loop:

1. Create or edit a project-local atom file.
2. Register the atom with `pcq.register_*`.
3. Declare `params`, `tasks`, and relevant contracts.
4. Add or update a `RecipeSpec` that references the atom with `AtomRef`.
5. Run static validation.
6. Run a smoke contract check with fake or tiny data.
7. Submit the CQ run only after validation passes.

The generated code can be arbitrary PyTorch or Python, but the registered atom
surface must be structured. This is the key distinction:

```text
implementation: free-form Python
public experiment surface: AtomSpec + AtomRef + validation report
```

### Minimum Contract By Kind

Every custom atom does not need exhaustive metadata on day one. It needs enough
metadata for an agent to avoid obvious invalid experiments.

Minimum model contract:

- accepted params
- task
- input contract
- output contract
- optional extras or hardware assumptions

Minimum dataset contract:

- accepted params
- split semantics
- sample output contract
- label dtype/range
- fixed shape or required collate/transform
- network/download behavior

Minimum loss contract:

- accepted params
- logits/target contract
- target dtype/range assumptions
- ignore index semantics when applicable

Minimum metric contract:

- accepted params
- input contract
- monitor mode, `min` or `max`
- ignore index semantics when applicable

Minimum optimizer/scheduler contract:

- accepted hyperparameters
- required runtime dependency (`params` or `optimizer`)
- required horizon params such as `T_max`

If metadata is incomplete, the registry can still load the atom, but validation
should mark it as `metadata_status="inferred"` or warn that agent automation is
limited.

## CLI Requirements

The registry should be inspectable through CLI.

```bash
pcq atoms list --kind model --json
pcq atoms show model unet --json
pcq atoms validate-ref model_ref.json --json
```

Example output:

```json
{
  "schema_version": 1,
  "kind": "model",
  "name": "unet",
  "params": {
    "in_channels": {"type": "int", "default": 3},
    "num_classes": {"type": "int", "default": 21}
  },
  "tasks": ["segmentation"],
  "input_contract": {"x": ["B", "C", "H", "W"]},
  "output_contract": {"logits": ["B", "num_classes", "H", "W"]}
}
```

## Migration Plan

### Step 1: Extend Registry Without Breaking Old API

- keep current `register_model(name, factory)` API
- add optional `meta=...`
- store `AtomSpec`
- infer minimal metadata for old registrations
- expose `.meta(name)` and `.build(name, **params)`

### Step 2: Add Ref Constructors

Add:

```python
pcq.model_ref(name, params=None)
pcq.dataset_ref(name, params=None)
pcq.loss_ref(name, params=None)
pcq.optim_ref(name, params=None)
pcq.sched_ref(name, params=None)
pcq.metric_ref(name, params=None)
pcq.collate_ref(name, params=None)
```

### Step 3: Teach Trainer To Resolve Refs

`Trainer` should accept both old executable atoms and new refs.

Supported:

```python
"model": pcq.examples.models.unet(...)
"model": pcq.model_ref("unet", {"num_classes": 21})
```

### Step 4: Add RecipeSpec

Built-in recipes should expose:

```python
SPEC = RecipeSpec(...)

def recipe_name() -> dict:
    return SPEC.build()
```

### Step 5: Keep Built-In Atoms Minimal And Contract-Rich

Add metadata for the built-ins that exist, but do not grow the built-in catalog
just to cover every possible experiment. Built-ins should be enough to test and
demonstrate the contract:

- `mlp`
- `small_cnn`
- `resnet18`
- `text_classifier`
- `unet`
- `deeplab_v3`
- `cross_entropy`
- `dice`
- `focal`
- `adamw`
- `sgd`
- `cosine`
- `fake`
- `fake_text`
- `fake_seg`
- `cifar10`
- `mnist`
- `voc_seg`
- all functional metrics

New domain-specific atoms should normally be project-local unless they are
general enough to serve as maintained examples.

### Step 6: Add Project Atom Discovery

Support predictable local atom loading:

- `cq_atoms.py`
- `atoms/*.py`
- optional explicit module list in `cq.yaml` or recipe metadata

Discovery must be deterministic and side-effect bounded:

- import only project atom modules
- do not instantiate large models during inspection
- do not download datasets during inspection
- report import failures as structured validation errors

### Step 7: Enforce Metadata In Acceptance

Acceptance should fail or warn when:

- required param schemas are missing
- shape contracts are missing for built-in atoms
- label contracts are missing for losses/datasets
- monitor candidates are absent
- smoke substitute atoms lack compatible contracts

## Backward Compatibility

The following must keep working:

```python
pcq.Trainer(task="classification", dataset="fake", model="mlp").fit()
pcq.Trainer(preset="vision/fake_smoke").fit()
pcq.register_model("custom", lambda: MyModel())
```

New metadata should be additive.

Compatibility rules:

- old factories can be wrapped in `AtomSpec`
- old recipe dicts can be inspected with best-effort metadata
- direct object atoms remain valid
- warnings should guide migration but not break users immediately

## Test Strategy

### Registry Tests

- register old-style factory
- register metadata-rich atom
- duplicate name behavior is deterministic
- unknown atom fails clearly
- `.build()` passes params correctly
- `.meta()` is JSON-safe
- param validation catches missing required params
- param validation catches invalid types and choices

### Ref Tests

- `model_ref` serializes to dict
- `loss_ref` resolves to module
- unknown ref fails validation
- unknown param fails validation
- ref works inside recipe dict
- ref works inside `RecipeSpec.build()`

### Contract Tests

- dataset output contract matches model input contract
- loss label contract catches ignore index mismatch
- metric contract catches incompatible output shape
- collate contract catches variable image size

### Agent CLI Tests

- `pcq atoms list --json`
- `pcq atoms show model unet --json`
- `pcq recipe-meta` includes atom specs
- `pcq validate` reports atom contract failures

## Recommended Next Implementation Scope

For the next version, do not try to fully convert every recipe or build a large
upstream atom catalog.

Recommended order:

1. Introduce `AtomSpec`, `ParamSpec`, and `AtomRef`.
2. Extend registry to store specs with optional metadata.
3. Add ref constructors.
4. Add metadata for `cross_entropy`, `unet`, `fake_seg`, `voc_seg`, and `iou`.
5. Convert segmentation recipes first because they expose shape and label
   contract issues most clearly.
6. Add project atom discovery through `cq_atoms.py` or `atoms/*.py`.
7. Add an atom scaffold command for agents.
8. Add validation for dataset-loss ignore index mismatch.
9. Add validation for variable image shape without resize/collate.
10. Add a smoke contract check for project-local/generated atoms.

This sequence directly supports the current agent-operability gaps.

## Success Criteria

The atom registry is agent-operable when:

- agents can list every atom with JSON metadata
- agents can see valid params before choosing changes
- agents can generate `AtomRef` without importing heavy code
- `Trainer` can resolve refs and direct objects
- custom atoms can be registered with metadata in user projects
- generated atoms can be loaded from project files and validated
- validation catches common shape, label, monitor, and dependency failures
- recipes can be represented as metadata-first `RecipeSpec`

## Final Target

The final model should be:

```text
Human:
  pcq.examples.models.unet(...)
  pcq.loss.cross_entropy(...)

Agent:
  AtomSpec in registry
  AtomRef in recipe/plan
  JSON metadata in CLI
  validation before run
  factory resolution only at execution time
```

This keeps the human API simple while making the same library safely operable by
CQ service agents.
