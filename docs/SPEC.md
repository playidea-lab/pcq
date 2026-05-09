# pcq Spec

## Summary

`pcq` is a CQ-compatible, but not CQ-only, ML authoring library for writing
experiments that are short, reproducible, and easy to compare.

The package is distributed on PyPI as `pcq`. The short `cq` package name is
already occupied and is reserved conceptually for the managed CQ service
boundary, while this open-source contract library is imported as `pcq`. The
`pcq` CLI command is the authoring entry point:

```bash
uv add pcq
```

```python
import pcq
```

`pcq` is not a replacement for PyTorch Lightning, Hugging Face Trainer, PyCaret,
or CQ itself. It is a thin authoring layer over the `cq.yaml` runtime contract:
configuration comes from `cq.yaml` / `CQ_CONFIG_JSON`, metrics go to stdout or
standard artifact files, and artifacts are written to the configured output
directory. CQ service is one managed runtime that can consume this contract; a
local script, CI job, notebook workflow, third-party orchestrator, or coding
agent can consume it directly without CQ service.

The durable completion object should become a `RunRecord`: a machine-readable
experiment record that combines execution, source, environment, input identity,
metric schema, artifact manifest, agent provenance, validation, and result
summary. See [RunRecord Standard](RUN_RECORD.md).

For the remaining work required to treat `pcq` as complete from an
agent-operated service perspective, see
[pcq Completion Roadmap](PCQ_COMPLETION_ROADMAP.md).

## Direction

The library should grow as a small, strong standard layer for CQ-runnable and
standalone agent-operated experiments.

1. CQ contract first
   - CQ worker/system must continue to observe only env/stdout/filesystem.
   - The library must not require CQ Hub/Drive clients at runtime.
   - Experiments must still be runnable without `pcq` if they follow the same
     contract manually.

2. Recipe-first, atom-refine
   - Users should start from verified recipes.
   - Users should refine experiments by overriding dataset/model/loss/optimizer/
     scheduler atoms, not by copying large training scripts.
   - Recipes must remain inspectable: users can see which atoms a preset uses.
   - If a needed atom does not exist, users or agents should add a project-local
     registered atom with metadata instead of requiring `pcq` upstream changes.

3. Keep core small
   - `Experiment` may remove repetitive train/eval/checkpoint/resume boilerplate.
   - `Trainer` may compose common experiments from recipes and atoms.
   - The project must not drift into a full callback/plugin framework.

4. Reproducibility as default
   - Save config, metrics history, model weights, checkpoints, and git metadata.
   - Warn on undeclared metrics so `cq.yaml.metrics` and emitted logs stay aligned.
   - Resume should be explicit and predictable.

5. Contract scripts, not adapters
   - Any ML framework can be used from project-local code if the run follows the
     CQ contract.
   - `pcq` must not require or advertise framework-specific adapters as the
     normal integration path.
   - Repeated patterns may become examples or scaffolds, but not a supported
     framework matrix.
   - They must not make the base import heavy or couple `pcq` to CQ internals.

6. RunRecord as the completion boundary
   - A process exit code is not enough to mark an experiment complete.
   - `pcq` should assemble standard contract artifacts first, then converge on
     `run_record.json` as the canonical run evidence.
   - CQ service should treat a run as complete only after post-run contract
     validation passes or records a structured failure.

## Non-Goals

- Reimplementing Lightning, HF Trainer, PyCaret, or a full callback ecosystem.
- Direct Hub/Drive orchestration in the core package.
- Automatic hyperparameter sweep management.
- Hidden magic that changes the CQ worker contract.
- A broad model zoo without recipe acceptance tests.
- A built-in atom catalog that attempts to own every possible model, loss,
  scheduler, transform, or metric.
- A built-in atom catalog growing as a model zoo — built-in atoms are
  reference examples only (`role="reference_example"`), production atoms live
  in project-local `atoms/` via `pcq.register_*`.

## Core Contract

CQ system and `pcq` communicate through this runtime contract only:

```text
Config input:
  CQ_CONFIG_JSON -> path to normalized inline configs JSON

Declared metrics:
  CQ_DECLARED_METRICS -> optional comma-separated metric names
  cfg["_metrics_declared"] -> optional metric names injected from pcq.yaml.metrics

Metric output:
  stdout line -> "@key=value @other=value"

Artifact output:
  files under output_dir
  cq.yaml.artifacts globs are collected by the worker after process exit

Success/failure:
  cmd exit code decides job status
  missing metrics/artifacts are warnings unless the worker enforces otherwise
```

`pcq` wraps this contract for convenience. It must not introduce a second,
private runtime protocol.

The low-level contract is enough to connect arbitrary ML code. A project can use
Hugging Face Trainer, TabPFN, PyCaret, scikit-learn, XGBoost, or a custom script
without a built-in `pcq` integration, as long as it reads config from CQ, emits
declared metrics, and writes standard artifacts.

For the detailed resolver, worker, and artifact completion rules, see
[CQ YAML Runtime Contract](CQ_YAML_RUNTIME_CONTRACT.md) and
[Worker Execution Flow](WORKER_EXECUTION_FLOW.md).

## Public API

### Low-Level API

Low-level APIs expose the CQ contract directly.

```python
import pcq

cfg = pcq.config()
pcq.seed_everything(cfg.get("seed", 42))
out = pcq.output_dir()
train_dir = pcq.input_dir("train")
pcq.log(epoch=1, train_loss=0.42)
```

Required behavior:

- `pcq.config()` reads `CQ_CONFIG_JSON` and returns a `dict`.
- Missing `CQ_CONFIG_JSON` raises a clear `RuntimeError`.
- `pcq.output_dir()` creates and returns `cfg["output_dir"]` or `output`.
- `pcq.input_dir(name)` reads `CQ_INPUT_DIR_<NAME>` or `cfg["inputs"][name]`.
- `pcq.log(**values)` prints finite numeric values only.
- bool, string, NaN, and inf are skipped.
- undeclared metric keys emit one stderr warning and an exit summary.
- `pcq.log(strict=True, ...)` fails immediately on undeclared metric keys.

### Mid-Level API

`Experiment` is the main extensibility point. It is Lightning-style in shape but
must stay much smaller in scope.

```python
import pcq


class MyExperiment(pcq.Experiment):
    def build_dataset(self, split):
        ...

    def build_model(self):
        ...

    def build_loss(self):
        ...

    def build_optimizer(self, params):
        ...

    def build_scheduler(self, optimizer):
        ...

    def training_step(self, batch):
        ...

    def eval_step(self, batch):
        ...


MyExperiment().fit()
```

Required behavior:

- `Experiment` provides `cfg`, `device`, `output_dir`, and `history`.
- Default splits are `train` and `eval`.
- Default loop supports one model, one optimizer, train/eval epochs, and optional
  scheduler.
- `training_step` and `eval_step` return metric dictionaries.
- `loss` is the optimization key during training.
- `fit()` writes `model.pt`, `config.json`, `metrics.json`, `last.ckpt`, and
  `best.ckpt`.
- `resume_from` points to an explicit checkpoint and must fail if missing.
- `resume: true` may auto-resume from `output_dir/last.ckpt`; missing checkpoint
  means fresh start.
- Users may override `fit()` when the default loop is too small.

### Contract Script API

The lowest-friction integration path is a plain script that uses only low-level
CQ helpers. This is the default path for frameworks that do not fit the torch
`Experiment` shape.

```python
import pcq


cfg = pcq.config()
out = pcq.output_dir()

# Load data and use any ML library here.
# Examples: HF Trainer, TabPFN, PyCaret, sklearn, XGBoost, custom code.
...

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

Rules:

- This path is first-class.
- It must be inspectable through `cq.yaml`, artifacts, and summary files.
- It does not require any official framework integration.
- If the pattern becomes repeated, a project-local helper or atom can wrap it.
- Examples or scaffolds may exist later, but they must not define support.
- `pcq.save_all(...)` is the preferred way to write contract artifacts from a
  script-style run. It currently writes `config.json`, `metrics.json`,
  `run_summary.json`, and `manifest.json`.

Allowed focused extensions:

- gradient accumulation
- AMP / mixed precision
- best checkpoint monitor and `min`/`max` mode
- early stopping
- artifact manifest
- distributed path through optional `accelerate`

These should be config-driven and minimal. They should not become a general
callback system.

### High-Level API

`Trainer` is the one-liner API for common cases. It composes an internal
`Experiment` from a recipe plus optional atom overrides.

```python
import pcq

# Preset only.
pcq.Trainer(preset="vision/cifar10_smallcnn_baseline").fit()

# Preset plus atom override.
pcq.Trainer(
    preset="vision/cifar10_smallcnn_baseline",
    sched_factory=lambda optim: pcq.sched.cosine(optim, T_max=20, warmup=1000),
).fit()

# Atom names without preset.
pcq.Trainer(task="classification", dataset="fake", model="mlp").fit()
```

Required behavior:

- High-level API uses `Experiment` internally.
- Presets are inspectable through `Trainer.list_presets()` and
  `Trainer.print_recipe(name)`.
- Atom overrides replace the preset value directly.
- Unknown preset/model/dataset/task names fail with clear errors.
- Trainer must not hide enough behavior that users cannot drop down to
  `Experiment`.

## Atom And Registry System

Atoms are small factories or objects for:

- datasets
- models
- losses
- optimizers
- schedulers
- metrics

The extension direction is to make registration public and consistent. The
registry is a contract boundary, not a promise that `pcq` will implement every
atom itself:

```python
pcq.register_model("vit_b16", lambda: make_vit_b16(num_classes=10))


@pcq.register_dataset("my_dataset")
def my_dataset(split):
    return MyDataset(train=(split == "train"))
```

Rules:

- Direct Python atoms remain first-class. For built-in examples, prefer
  `pcq.examples.{models,datasets,optim,sched}`; `pcq.{models,datasets,optim,sched}`
  remain as v2 compatibility facades.
- String lookup exists for authoring convenience and recipes.
- Registries must be inspectable and deterministic.
- Internal hard-coded name maps should converge on the registry API.
- Built-in atoms should stay small and verified.
- Project-local atoms are the primary extension mechanism for real experiments.
- Agent-generated atoms are acceptable when they are ordinary project files with
  explicit `AtomSpec` metadata and smoke validation.

For the agent-operable target, registries should evolve from `name -> callable`
to `name -> AtomSpec(factory, metadata, param schema, contracts)`. See
[Atom Registry And Custom Atoms](ATOM_REGISTRY.md) for the detailed design,
including custom model/loss/scheduler registration and `AtomRef` recipe usage.

Project-local atom loading should be predictable. A project may expose local
atoms through a single import point:

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

`train.py` imports `cq_atoms` before resolving recipes, and `cq_atoms.py`
imports the concrete atom modules. This gives users and agents a bounded place
to add experimental code while preserving the CQ runtime contract.

## Recipe System

Recipes are transparent atom bundles. They are not hidden training programs.

Recommended recipe shape:

```python
def cifar10_resnet18():
    return {
        "task": "classification",
        "dataset": lambda split: pcq.examples.datasets.cifar10(
            "data", train=split == "train"
        ),
        "model": lambda: pcq.examples.models.resnet18(num_classes=10),
        "loss": pcq.loss.cross_entropy,
        "optim_factory": lambda params: pcq.examples.optim.adamw(params, lr=1e-3),
        "sched_factory": lambda optim: pcq.examples.sched.cosine(optim, T_max=50),
        "epochs": 50,
        "batch_size": 128,
        "metrics": ["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"],
    }
```

Recipe acceptance criteria:

- Import succeeds without running training.
- Recipe can be inspected without side effects.
- A fake/small-data smoke path exists for tests.
- `Trainer(preset=...).fit()` completes a one-epoch smoke run.
- Declared metrics match emitted metrics.
- Standard artifacts are produced.
- Resume smoke test passes when applicable.
- Network/download requirements are either optional, cached, or replaceable in
  tests.

Initial reference recipe family priorities:

1. `vision/image_classification`
2. `vision/segmentation`
3. `nlp/text_classification`
4. `tabular/classification`
5. `llm/finetune_lora`

## Config And Metrics

`cq.yaml` remains the source of runtime configuration.

```yaml
name: cifar10-baseline
cmd: uv run python train.py

configs:
  dataset: fake
  output_dir: output
  epochs: 5
  batch_size: 64
  lr: 0.001
  seed: 42
  resume: true

metrics:
  - epoch
  - train_loss
  - train_acc
  - eval_loss
  - eval_acc

artifacts:
  - output/
```

Metrics rules:

- There are no reserved metric keys.
- `epoch`, `global_step`, and `lr` are normal numeric metrics.
- Viewer chooses x/y axes.
- `pcq` should warn when emitted metric keys are not declared.
- Strict mode is for CI and recipe acceptance tests.

## Artifacts

Standard artifacts:

- `model.pt`: final model weights
- `config.json`: resolved config plus metadata
- `metrics.json`: `{"history": [...]}` metric history
- `run_summary.json`: best/last/target/provenance summary for agents
- `manifest.json`: output file index, currently `{"schema_version": 1, "files": [...]}`.
- `last.ckpt`: latest resumable checkpoint
- `best.ckpt`: checkpoint selected by monitor

Current artifact manifest:

```json
{
  "schema_version": 1,
  "files": [
    {"path": "model.pt", "kind": "model"},
    {"path": "metrics.json", "kind": "metrics"},
    {"path": "config.json", "kind": "config"}
  ]
}
```

Target manifest evolution:

- add `sha256`
- add `size_bytes`
- add `created_at` where available
- keep a compatibility path for the current `files` array

Target run completion artifact:

- `run_record.json`: canonical experiment record, defined in
  [RunRecord Standard](RUN_RECORD.md).

## Dependency Policy

- Base package: Python stdlib, `torch`, `numpy`.
- Optional distributed extra: `pcq[dist]` for `accelerate`.
- Optional vision extra: `pcq[vision]` for vision recipe dependencies.
- Optional NLP extra: `pcq[nlp]` for NLP recipe dependencies.
- Third-party framework support is just contract compliance from project-local
  code. It should not require official extras.
- Heavy imports must be lazy.
- Hub/Drive clients do not belong in the base package.

## Project Layout

```text
pcq/
  pyproject.toml
  src/pcq/
    __init__.py
    contract.py
    core.py
    experiment.py
    trainer.py
    _registry.py
    datasets.py
    models.py
    loss.py
    optim.py
    sched.py
    recipes/
    agent_assets/
      AGENTS.pcq.md
      skills/pcq/SKILL.md
  examples/
    cq.yaml
    train.py
  docs/
    RUN_RECORD.md
    CQ_YAML_RUNTIME_CONTRACT.md
    WORKER_EXECUTION_FLOW.md
    AGENT_OPERATING_GUIDE.md
    CQ_MCP_SPEC.md
    AGENT_ACCEPTANCE_CHECKLIST.md
    AGENT_OPERABILITY.md
    ATOM_REGISTRY.md
    VISION.md
  templates/
    AGENTS.pcq.md
  skills/
    pcq/
      SKILL.md
  project-template/
    cq_atoms.py
    atoms/
      models.py
      losses.py
      datasets.py
      metrics.py
  tests/
    test_core.py
    test_experiment.py
    test_trainer.py
    test_recipes.py
    test_integration.py
  README.md
  SPEC.md
```

## Test Plan

Core:

- `pcq.config()` reads `CQ_CONFIG_JSON`.
- missing `CQ_CONFIG_JSON` raises a clear error.
- `pcq.log()` prints only finite numeric values in `@key=value` format.
- `pcq.log()` skips bool/string/NaN/inf.
- undeclared metrics warn once and summarize on exit.
- strict undeclared metrics fail.
- `pcq.output_dir()` creates the configured directory.
- `pcq.input_dir(name)` resolves env and config inputs.

Experiment:

- fake dataset smoke run completes on CPU.
- standard artifacts are written.
- metrics history is saved.
- `resume_from` restores model/optimizer/scheduler state.
- `resume: true` auto-resumes from `last.ckpt` when present.
- missing explicit `resume_from` fails.

Trainer and recipes:

- atom-only trainer completes one epoch.
- preset trainer completes one epoch.
- preset atom override is applied.
- unknown names raise clear errors.
- recipe list and recipe inspection work.
- every recipe has an acceptance smoke test.

Integration:

- subprocess run with `CQ_CONFIG_JSON` simulates a CQ worker.
- stdout contains declared metrics.
- output directory contains expected artifacts.
- tests avoid network by substituting fake datasets where needed.

## Roadmap

### v1.2

- Align README, SPEC, pyproject, and examples around `pcq` install +
  `import pcq`.
- Promote atom registries to public API.
- Move Trainer string-name lookup to the registry API.
- Define recipe schema and acceptance tests.
- Add artifact manifest.
- Add monitor/mode for best checkpoint selection.

### v1.3

- Add segmentation recipe family.
- Add AMP and gradient accumulation.
- Add early stopping.
- Strengthen resume tests.
- Add richer metric aggregation.

### v2

- Optional CQ input/artifact helpers beyond `input_dir()` and `output_dir()`.
- External recipe packages.
- Contract-script examples for common third-party framework patterns if repeated
  demand is clear.
- Optional Drive integration without coupling core runtime to CQ internals.

## Assumptions

- Package distribution name is `pcq`; import name is `cq`.
- v1 supports Python/uv ML experiments.
- Core remains CQ-contract-first and small.
- Recipe catalog can grow only with smoke tests and inspectability.
- CQ worker continues to observe env/stdout/filesystem and never imports `cq`.
