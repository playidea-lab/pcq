---
name: pcq
description: >
  Use when creating, modifying, validating, running, or reviewing CQ ML
  experiments that use cq.yaml, pcq, project-local atoms, contract scripts,
  RunRecord artifacts, or CQ worker execution.
---

# pcq Skill

## Goal

Operate CQ ML experiments through the CQ runtime contract:

```text
cq.yaml -> resolved config -> execution -> standard artifacts -> RunRecord
```

Use this skill when a user asks to:

- create a CQ ML experiment
- modify an existing pcq experiment
- add a custom model, loss, dataset, metric, optimizer, or scheduler
- connect a third-party ML framework
- validate or summarize experiment artifacts
- debug missing metrics, artifacts, RunRecord, or output directory issues
- prepare a project for CQ worker execution

## Core Rules

- `cq.yaml` is the execution contract.
- `run_record.json` is the completion record.
- Use `pcq.output_dir()` for artifact paths.
- Use `pcq.save_all(...)` or equivalent standard artifact helpers.
- Built-in atoms are reference examples, not a production catalog.
- Real experiment components belong in project-local code.
- Any ML framework is valid if it honors the CQ contract.

## Start Here

From the project root, gather structured state:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

Read the output before editing. Identify:

- selected `cq.yaml`
- command
- output directory
- declared metrics
- existing output artifacts
- entrypoint style: script, Trainer, Experiment, or project-local atoms

## Choose The Style

```text
Third-party framework owns the training flow?
  -> contract script
Otherwise, component swapping matters?
  -> project-local atoms + Trainer/RecipeSpec
Otherwise, custom PyTorch train loop?
  -> Experiment
Otherwise
  -> Trainer preset or simple contract script
```

### Contract Script

Use for HF Trainer, TabPFN, PyCaret, sklearn, XGBoost, LightGBM, or custom
framework code.

Required shape:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()

# framework code

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

Do not create framework adapters unless the project has a repeated, validated
need. The contract is the adapter.

### Project-Local Atoms

Use when the component needs a name, metadata, validation, smoke testing, or
reuse.

Scaffold:

```bash
pcq atoms scaffold model my_model
pcq atoms scaffold loss my_loss
pcq atoms scaffold metric my_metric
```

Validate:

```bash
pcq atoms validate-local
pcq atoms smoke <kind> <name> --load-project .
```

Keep the registered atom metadata accurate:

- `tasks`
- `params`
- `input_contract`
- `output_contract`
- `label_contract` or `metric_contract` when applicable
- `requires_extras` when optional dependencies are needed
- `smoke_safe`

### Trainer

Use when a recipe/preset and atom composition are the main surface:

```python
import pcq

cfg = pcq.config()
pcq.seed_everything(cfg.get("seed", 42))
pcq.Trainer.from_cfg(cfg).fit()
```

### Experiment

Use when custom PyTorch train/eval logic is needed but the CQ contract should
still be handled by pcq:

```python
import pcq

class MyExperiment(pcq.Experiment):
    ...

cfg = pcq.config()
MyExperiment(cfg=cfg).fit()
```

## Validation Workflow

Before execution:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

After execution:

```bash
pcq validate-run <output_dir> --strictness 3 --json
pcq describe-run <output_dir> --json
```

For live agent observation during execution:

```bash
pcq run --path . --jsonl
pcq run --path . --events output/events.jsonl --json
```

For comparisons:

```bash
pcq compare-runs <base_output_dir> <candidate_output_dir> --json
pcq lineage <candidate_output_dir> --json
```

## Common Fixes

### Artifacts Are Split Across Directories

Cause:

- code used `Path("output")`
- helper ignored `cq.yaml.configs.output_dir`
- CLI guessed project root from output directory name

Fix:

- use `pcq.output_dir()`
- keep all standard artifacts under the resolved output directory
- rerun `pcq resolve --json` and `pcq inspect --json`

### Missing RunRecord

Fix:

```bash
pcq finalize <output_dir>
pcq validate-run <output_dir> --strictness 3 --json
```

Then inspect `run_record.json`.

### Unknown Project Atom

Fix:

- ensure `pcq_atoms.py` imports the atom module
- ensure `atoms/__init__.py` exists when using package imports
- run:

```bash
pcq atoms list --load-project . --source project --json
```

### Undeclared Metric

Fix:

- add the metric to `cq.yaml.metrics`
- or ensure the worker injects `_metrics_declared`
- keep emitted `pcq.log(...)` keys aligned with declarations

## Done Criteria

A completed agent change should leave:

- valid `cq.yaml`
- implementation code under project-local files unless changing pcq itself
- passing pre-run validation
- standard post-run artifacts when a run was executed
- `run_record.json`
- `validation_report.json`
- clear summary of what changed and what result evidence supports it

## References

- `docs/CQ_YAML_RUNTIME_CONTRACT.md`
- `docs/WORKER_EXECUTION_FLOW.md`
- `docs/AGENT_OPERATING_GUIDE.md`
- `docs/CQ_MCP_SPEC.md`
- `docs/AGENT_ACCEPTANCE_CHECKLIST.md`
