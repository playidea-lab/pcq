# pcq Vision

## One-Line Future

`pcq` is the standard authoring library for producing CQ-runnable ML
experiments from a project folder containing `cq.yaml`, data, and Python
recipes, then emitting metrics and artifacts in a consistent form.

## Problem

ML experiments often start as local scripts and later become remote jobs. During
that transition, each project reinvents the same pieces:

- config loading
- dataset/input path handling
- stdout metric formatting
- output directory conventions
- checkpoint and resume behavior
- artifact layout
- experiment comparison metadata

CQ already defines the runtime contract for remote execution. What is missing is
a Python authoring layer that lets users produce experiments that naturally fit
that contract without copying boilerplate training scripts.

## Product Position

`pcq` should occupy this position:

```text
PyTorch          -> tensor, model, optimizer runtime
Lightning/HF     -> broad training frameworks
W&B/MLflow       -> experiment tracking products
CQ               -> remote execution and orchestration
pcq             -> CQ-native ML experiment authoring SDK
```

The goal is not to become a general ML framework. The goal is to make CQ
experiments easy to author, inspect, run, resume, compare, and reproduce.

`pcq` should also not become a broad upstream catalog of every possible model,
loss, scheduler, transform, or dataset wrapper. Its durable value is the
contract that lets those components be created inside a project, registered,
validated, executed, and summarized by CQ service agents.

The long-term object is not merely `output/` or `run_summary.json`. It is a
`RunRecord`: a structured experiment record that can be replayed, audited,
compared, and used as the parent for the next agent-written experiment. See
[RunRecord Standard](RUN_RECORD.md).

## Core Mental Model

The standard experiment shape is:

```text
project/
  cq.yaml
  data/ or CQ inputs
  train.py or recipes
  output/

cq.yaml
  -> command to run
  -> configs passed through CQ_CONFIG_JSON
  -> declared metrics
  -> artifact globs

pcq
  -> reads config/input/output contract
  -> runs Experiment or Trainer
  -> logs @key=value metrics
  -> writes standard artifacts
  -> converges those artifacts into a RunRecord
```

`cq.yaml` is the execution contract. `pcq` is the Python authoring contract.
`output/` is the result contract. `run_record.json` should become the canonical
evidence object for CQ service.

## User Journey

### 1. Start From A Verified Recipe

```python
import pcq

pcq.Trainer(preset="vision/image_classification/cifar10_resnet18").fit()
```

The user gets a working CQ-compatible experiment with standard metrics,
checkpoints, and artifacts.

### 2. Override Or Add Atoms Instead Of Copying Scripts

```python
import pcq

pcq.Trainer(
    preset="vision/segmentation/unet_baseline",
    dataset=my_dataset,
    model=my_model,
    monitor="eval_miou",
    mode="max",
).fit()
```

The user or agent changes only the atoms that matter: dataset, model, loss,
optimizer, scheduler, metrics, or monitor. If the needed atom does not exist,
the preferred path is to add a project-local atom that declares its contract, not
to expand `pcq` internals.

### 3. Drop Down To Experiment When Needed

```python
import pcq


class MyExperiment(pcq.Experiment):
    def build_dataset(self, split):
        ...

    def training_step(self, batch):
        ...


MyExperiment().fit()
```

The user can leave the high-level API without leaving the CQ contract.

### 4. Use Any ML Library Through The Contract

`pcq` should not require every experiment to use the built-in torch Trainer.
If a project is better served by Hugging Face Trainer, TabPFN, PyCaret,
scikit-learn, XGBoost, or a custom script, the agent can write ordinary
project-local code as long as it honors the CQ output contract:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# Use any framework here.
...

pcq.log(epoch=0, eval_acc=eval_acc)
pcq.save_all(
    history=[{"epoch": 0, "eval_acc": eval_acc}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

There is no supported-framework matrix and no adapter requirement. The
foundation is that arbitrary ML code can be normalized into a CQ-compatible
experiment unit. In v1.13 this normalization produces contract artifacts; the
next step is to finalize those artifacts into a full RunRecord.

## Agent-Native Future

The long-term opportunity is not only a Python library. It is an experiment
authoring substrate that agents can safely manipulate.

A user should be able to ask:

```text
Improve dental segmentation mIoU.
```

An agent should then be able to:

- inspect `cq.yaml`, recipe metadata, and previous artifacts
- choose or generate a recipe
- override atoms and config values
- run the CQ job
- read metrics and artifacts
- propose the next experiment
- preserve provenance in a consistent shape

For this to work, `pcq` experiments must be structured, inspectable, and
predictable. Agents should not need to rewrite large training scripts for every
experiment. They should usually modify recipe names, atom factories, config
values, and monitor settings.

## Design Principles

1. CQ contract first
   - Env, stdout, and filesystem remain the only required runtime interface.
   - CQ worker/system does not import `pcq`.

2. Thin core, contract-rich project atoms
   - Core APIs stay small.
   - Built-in atoms stay minimal and smoke-testable.
   - Capability grows through verified recipes and project-local atoms.
   - New research components should usually live in the user project.

3. Inspectability over magic
   - Presets must show their atom composition.
   - Generated artifacts must be predictable.
   - Config should explain behavior rather than hide it.

4. Reproducibility by default
   - Save config, metrics, checkpoints, model weights, and provenance.
   - Resume behavior must be explicit and testable.
   - A run should not be considered complete until contract artifacts can be
     validated and summarized.

5. Contract scripts, not adapters
   - Any framework can be used from project-local code if it emits the standard
     CQ metrics and artifacts.
   - `pcq` should not define framework-specific adapters as a core concept.
   - Repeated patterns may become examples or scaffolds, but the contract stays
     the product boundary.
   - Core must remain lightweight.

## What pcq Must Not Become

- a Lightning clone
- a Hugging Face Trainer clone
- a callback/plugin ecosystem
- a model zoo with untested recipes
- an ever-growing upstream catalog of every possible atom
- an upstream catalog of atoms — built-ins exist as contract examples only
  (`role="reference_example"`), real research atoms are project-local
- a CQ Hub/Drive client hidden inside the core package
- a config magic layer that obscures what command actually runs

## Long-Term Architecture

```text
pcq-core
  config, input_dir, output_dir, log, contract artifact helpers

pcq-train
  Experiment, Trainer, checkpoint, resume, monitor, metrics

pcq-recipes
  verified recipe catalog for vision, segmentation, NLP, tabular, LoRA

pcq-agent
  metadata, RunRecord, and inspection helpers for agent-driven experiments
```

The package may remain one distribution while these boundaries are conceptual.
The boundaries matter because they keep the project from growing in the wrong
direction.

If framework helpers are ever added, they should be treated as reference
examples. They should not become required adapters or define which frameworks
are supported.

For the concrete transition from agent-readable helpers to an agent-operable
library, see [Agent-Operable pcq](AGENT_OPERABILITY.md). For the worker and
runtime contract details, see [CQ YAML Runtime Contract](CQ_YAML_RUNTIME_CONTRACT.md)
and [Worker Execution Flow](WORKER_EXECUTION_FLOW.md).

## Success Criteria

`pcq` is succeeding when:

- a new CQ ML experiment can start from a recipe in minutes
- every recipe has a smoke test and documented metrics/artifacts
- users can override atoms without copying training loops
- every run emits comparable metrics and standard artifacts
- every run can be represented as a RunRecord or a structured failure
- resume works predictably
- agents can inspect and modify experiments without rewriting whole scripts
- agents can create project-local atoms under an explicit contract
- the core API remains small enough to understand in one sitting

## North Star

`pcq` should make ML experiments feel like standard CQ production units:

```text
cq.yaml + data + recipe/config -> CQ run -> contract artifacts -> RunRecord
```

That is the durable value. Model support can grow, but the contract must stay
small, explicit, and reliable.
