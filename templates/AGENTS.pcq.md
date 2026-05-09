# PCQ Agent Rules

This project follows the CQ experiment contract.

## Runtime Contract

- Treat `cq.yaml` as the execution contract and source of truth.
- Resolve config through pcq / the CQ runtime contract; do not hard-code `output/`.
- Relative `output_dir` values are project-root relative.
- Write results under `pcq.output_dir()`.
- End runs with `pcq.save_all(...)` or equivalent standard artifacts.

## Standard Artifacts

Completed runs should produce:

- `config.json`
- `metrics.json`
- `run_summary.json`
- `manifest.json`
- `run_record.json`
- `validation_report.json`

`run_record.json` is the canonical completion record.

## Implementation Policy

- pcq is a contract runtime + agent CLI surface. There is no model catalog,
  recipe library, or framework adapter inside pcq.
- All ML code (datasets, models, losses, optimizers, schedulers, metrics,
  training loops) lives in your project's `train.py` or your own modules.
- Any ML framework is allowed when the script honors the CQ contract
  (read `cq.config()`, write `cq.save_all()`, emit `@key=value` metrics).

## Agent Workflow

Before edits or execution:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

After a serious run:

```bash
pcq run --path . --jsonl
pcq validate-run <output_dir> --strictness 3 --json
pcq describe-run <output_dir> --json
```

When comparing iterations:

```bash
pcq compare-runs <base_output_dir> <candidate_output_dir> --json
pcq lineage <candidate_output_dir> --json
```

When applying agent-authored experiment plans:

```bash
pcq apply-plan plan.json --path . --json
pcq apply-planset planset.json --path . --output-pattern 'runs/exp{i}' --json
```

## Do Not

- Do not write artifacts to a hard-coded `output/` path.
- Do not emit metrics that are missing from `cq.yaml.metrics`.
- Do not treat process exit code alone as run completion.
- Do not continue an experiment loop after blocking validation failure.
- Do not look for `pcq.Trainer`, `pcq.Experiment`, `pcq.recipes`, or
  `pcq.examples.*` — these were removed in v4.0. Write a contract script.
