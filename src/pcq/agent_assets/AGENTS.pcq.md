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

- Built-in atoms are reference examples for smoke, onboarding, and contract
  verification. They are not a production catalog.
- Production models, losses, metrics, datasets, optimizers, and schedulers
  belong in project-local code such as `pcq_atoms.py`, `atoms/`, `recipes/`, or
  `train.py`.
- Any ML framework is allowed when the script honors the CQ contract.
- Prefer contract scripts for framework-owned workflows.
- Prefer project-local atoms when a component must be selected, swapped,
  validated, or reused.
- Do not add framework adapters or project-specific components to `pcq`
  internals for a single experiment.

## Agent Workflow

Before edits or execution:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

For project-local atoms:

```bash
pcq atoms validate-local
pcq atoms smoke <kind> <name> --load-project .
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

## Do Not

- Do not write artifacts to a hard-coded `output/` path.
- Do not emit metrics that are missing from `cq.yaml.metrics`.
- Do not treat process exit code alone as run completion.
- Do not continue an experiment loop after blocking validation failure.
- Do not hide network downloads or heavy work in project atom imports.
