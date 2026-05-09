---
name: pcq
description: >
  Use when creating, modifying, validating, running, or reviewing CQ ML
  experiments that use cq.yaml, pcq contract scripts, RunRecord artifacts,
  or CQ worker execution.
---

# pcq Skill (v4.0)

## Goal

Operate CQ ML experiments through the CQ runtime contract:

```text
cq.yaml -> resolved config -> contract script -> standard artifacts -> RunRecord
```

Use this skill when a user asks to:

- create a CQ ML experiment
- modify an existing pcq experiment
- connect any ML framework (sklearn, XGBoost, HF Trainer, PyTorch, ...)
- validate or summarize experiment artifacts
- debug missing metrics, artifacts, RunRecord, or output directory issues
- prepare a project for CQ worker execution

## Core Rules

- `cq.yaml` is the execution contract.
- `run_record.json` is the completion record.
- Use `pcq.output_dir()` for artifact paths.
- Use `pcq.save_all(...)` or equivalent standard artifact helpers.
- pcq is a contract runtime + agent CLI — there is **no model catalog** inside
  pcq. All ML code (model, dataset, loss, optimizer, scheduler, metric, train
  loop) lives in your project's `train.py`.
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

## Contract Script Pattern

Every project has one `train.py` shaped roughly like this:

```python
import pcq

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

# === Your ML code here — any framework ===
# import sklearn / torch / xgboost / transformers / ...
# build model, train, evaluate
score = float(model.score(X_test, y_test))

pcq.log(epoch=0, eval_score=score)
pcq.save_all(
    history=[{"epoch": 0, "eval_score": score}],
    status="completed",
    artifacts={"model": "model.pkl"},
)
```

`pcq.save_all()` writes 6 standard artifacts in one call:

- `config.json`
- `metrics.json`
- `manifest.json`
- `run_summary.json`
- `run_record.json` (canonical completion SSOT)
- `validation_report.json` (post-run gates)

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

## Agent-Authored Plans

`ExperimentPlan` lets an LLM agent propose `set_config` mutations on cq.yaml:

```bash
pcq apply-plan plan.json --path . --json
pcq apply-planset planset.json --path . --output-pattern 'runs/exp{i}' --json
```

Plans only mutate `cq.yaml.configs.<key>`; `train.py` is your code and is not
touched.

## Common Fixes

### Artifacts Are Split Across Directories

Cause:

- code used `Path("output")` directly
- helper ignored `cq.yaml.configs.output_dir`

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

### Undeclared Metric

Fix:

- add the metric to `cq.yaml.metrics`
- keep emitted `pcq.log(...)` keys aligned with declarations

## Done Criteria

A completed agent change should leave:

- valid `cq.yaml`
- contract script `train.py`
- passing pre-run validation
- standard post-run artifacts when a run was executed
- `run_record.json`
- `validation_report.json`
- clear summary of what changed and what result evidence supports it

## References

- `docs/CQ_YAML_RUNTIME_CONTRACT.md`
- `docs/AGENT_OPERATING_GUIDE.md`
- `docs/JSON_CONTRACTS.md`
- `docs/STRICTNESS.md`
