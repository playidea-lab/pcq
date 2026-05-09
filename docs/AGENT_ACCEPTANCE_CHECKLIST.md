# Agent Acceptance Checklist

## Purpose

This checklist defines whether `pcq` is usable by a coding agent as an
agent-operable experiment library.

Passing this checklist means an agent can start from a user goal, create or
modify a CQ ML experiment, validate it, run it through the CQ contract, and use
the resulting RunRecord for the next decision.

## Acceptance Levels

### Level 0: Readable

The agent can understand the project without running training.

- [ ] `pcq resolve --json` identifies project root, `cq.yaml`, command, config,
      metrics, inputs, and output directory.
- [ ] `pcq inspect --json` identifies entrypoint style.
- [ ] `pcq inspect --json` reports project-local atoms when explicitly loaded.
- [ ] `pcq inspect --json` reports output artifacts from the resolved output
      directory.
- [ ] Built-in atoms are clearly marked or documented as reference examples.

### Level 1: Scaffoldable

The agent can create a valid starting experiment.

- [ ] `pcq init-experiment --style script` creates `cq.yaml` and `train.py`.
- [ ] `pcq init-experiment --style trainer --preset ...` creates atom
      infrastructure.
- [ ] `pcq init-experiment --with-pyproject` creates reproducible dependency
      evidence.
- [ ] `pcq init-experiment --agent codex|claude|both` installs agent runtime
      assets through the same path as `pcq agent install`.
- [ ] `pcq agent install --target codex|claude|both --dry-run --json`
      previews writes without modifying files.
- [ ] `pcq agent status --target codex|claude|both --json` reports
      installed, missing, stale, unmanaged, divergent, and partial states
      without modifying files.
- [ ] Generated `cq.yaml` declares metrics and output artifacts.
- [ ] Generated code writes to `pcq.output_dir()`.
- [ ] Generated code ends with `pcq.save_all(...)` or equivalent standard
      artifacts.

### Level 2: Modifiable

The agent can add real project logic without patching `pcq` internals.

- [ ] `pcq atoms scaffold model <name>` creates project-local atom code.
- [ ] `pcq atoms scaffold loss <name>` creates project-local atom code.
- [ ] `pcq atoms validate-local` catches missing contracts.
- [ ] `pcq atoms smoke <kind> <name> --load-project .` verifies a minimal
      executable contract.
- [ ] New project atoms can be referenced from recipes or `Trainer.from_cfg`.
- [ ] Contract scripts can use any third-party ML framework without adapters.

### Level 3: Runnable

The agent can execute or hand off the experiment to a CQ worker.

- [ ] `cq.yaml.cmd` is the only command the worker needs to run.
- [ ] `CQ_CONFIG_JSON` overrides `cq.yaml.configs` without losing top-level
      `name`, `cmd`, `inputs`, or metrics.
- [ ] Relative `output_dir` resolves against project root.
- [ ] Nested cwd execution still finds the correct `cq.yaml`.
- [ ] `pcq.save_all(finalize=True)` writes all standard artifacts into one
      resolved output directory.
- [ ] `pcq.log(...)` emits declared stdout metrics.
- [ ] `pcq run --json` emits parseable JSON only on stdout and captures child
      stdout/stderr to explicit log paths in the JSON envelope.

### Level 4: Verifiable

The agent and service can determine whether the run is complete.

- [ ] `pcq finalize <output_dir>` preserves `cq.yaml` metadata.
- [ ] `pcq validate-run <output_dir> --json` verifies manifest evidence.
- [ ] `pcq describe-run <output_dir> --json` returns status, target metric,
      mode, best/last, artifacts, validation status, reproducibility evidence,
      parent lineage, failure envelope, and policy-free `decision_facts`.
- [ ] `run_record.json` contains execution, source, environment, inputs,
      declared metrics, artifacts, validation, and result summary.
- [ ] `validation_report.json` records blocking failures instead of relying only
      on process exit code.

### Level 5: Iterative

The agent can use previous results to propose the next experiment.

- [ ] `pcq compare-runs` identifies metric/trajectory delta, config/input
      changes, validation/failure differences, artifact/source differences,
      lineage relation, and policy-free `decision_facts`.
- [ ] `pcq lineage` traces parent runs.
- [ ] Parent run ID/path can be stored in the next RunRecord.
- [ ] The agent can explain what changed between runs.
- [ ] The agent can stop when validation fails instead of continuing on
      unreliable evidence.

## Required Regression Scenarios

These scenarios should be covered by automated tests or acceptance scripts.

### Custom Output Directory

Input:

```yaml
configs:
  output_dir: runs/exp001
```

Expected:

- all standard artifacts are under `runs/exp001`
- `pcq inspect` finds that directory
- `pcq validate` checks that directory
- `pcq finalize runs/exp001` preserves `cq.yaml` context

### CQ YAML Only Local Execution

No `CQ_CONFIG_JSON` is set.

Expected:

- `pcq.config()` or the high-level runtime path can use `cq.yaml.configs`
- `pcq.output_dir()` follows `cq.yaml.configs.output_dir`
- `pcq.save_all(finalize=True)` writes a complete run

### Service Override Execution

`cq.yaml` defines:

```yaml
configs:
  output_dir: output
  epochs: 10
```

`CQ_CONFIG_JSON` defines:

```json
{
  "output_dir": "/work/runs/exp001/output",
  "epochs": 3
}
```

Expected:

- env output directory wins
- env epochs wins
- `cq.yaml.name`, `cq.yaml.cmd`, `cq.yaml.inputs`, and metric schema are still
  preserved

### Framework-Agnostic Script

Input:

- script uses sklearn, HF Trainer, TabPFN, PyCaret, or other framework
- script calls `pcq.output_dir()`, `pcq.log(...)`, and `pcq.save_all(...)`

Expected:

- no adapter is required
- standard artifacts exist
- RunRecord describes the run

### Project-Local Atom

Input:

- agent creates `atoms/models.py`
- atom declares metadata and contracts
- recipe references the atom

Expected:

- `pcq atoms validate-local` passes
- `pcq atoms smoke model <name> --load-project .` passes
- run can use the project atom without adding it to `pcq` upstream

### Agent Runtime Installation

Input:

```bash
pcq agent install --target both --path .
```

Expected:

- Codex assets are installed at `AGENTS.md` and
  `.agents/skills/pcq/SKILL.md`
- Claude Code assets are installed at `CLAUDE.md` and
  `.claude/skills/pcq/SKILL.md`
- existing instruction file content is preserved
- divergent existing skill files are skipped unless `--force`
- `--dry-run --json` reports the same planned paths without writing files
- `pcq agent status --target both --json` reports installed assets as
  `installed`
- status reports stale managed blocks and divergent skill files without
  overwriting them

## Release Gate

A release should not claim full agent-operable status unless:

- Level 0 through Level 4 pass.
- custom output directory scenarios pass.
- built-in atom documentation clearly says reference examples only.
- agent runtime assets install for Codex and Claude without destructive writes.
- agent runtime asset status can be checked read-only by agents and services.
- README links to the runtime contract, worker flow, and agent guide.
- service-facing MCP tool names and JSON shapes are documented.
