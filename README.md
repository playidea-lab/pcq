# pcq

> Apache-2.0 Python library for agent-operable experiment evidence and control.
> Bring any training code; `pcq` standardizes the run boundary.

`pcq` turns a project with `cq.yaml` into a reproducible experiment unit. It
loads config, resolves output paths, captures metrics, writes standard
artifacts, finalizes run evidence, and exposes JSON/JSONL surfaces that coding
agents, CI jobs, notebooks, and services can consume.

`pcq` is **not** a training framework, model zoo, adapter matrix, or CQ-only
client. Use PyTorch, Hugging Face Trainer, Lightning, sklearn, TabPFN, PyCaret,
XGBoost, shell scripts, remote jobs, or project-local research code. The
contract is the integration layer.

```text
pcq does not operate the model.
pcq operates the experiment boundary.
```

[SITE](https://playidea-lab.github.io/pcq/) |
[INTRODUCTION](docs/INTRODUCTION.md) |
[V4_DIRECTION](docs/V4_DIRECTION.md) |
[SPEC](docs/SPEC.md) |
[VISION](docs/VISION.md) |
[AGENT_OPERABILITY](docs/AGENT_OPERABILITY.md) |
[RUN_RECORD](docs/RUN_RECORD.md) |
[CQ_YAML_RUNTIME_CONTRACT](docs/CQ_YAML_RUNTIME_CONTRACT.md) |
[JSON_CONTRACTS](docs/JSON_CONTRACTS.md) |
[STRICTNESS](docs/STRICTNESS.md) |
[AGENT_OPERATING_GUIDE](docs/AGENT_OPERATING_GUIDE.md) |
[CHANGELOG](CHANGELOG.md)

Case studies (external evidence):
[mnist-dogfood](docs/case-studies/mnist-dogfood-2026-05-08.md) |
[tabular-dogfood](docs/case-studies/tabular-dogfood-2026-05-09.md) |
[mcp-dogfood](docs/case-studies/mcp-dogfood-2026-05-10.md) |
[cq-worker-dogfood](docs/case-studies/cq-worker-dogfood-2026-05-10.md)

Agent-readable site files:
[llms.txt](site/llms.txt),
[llms-full.txt](site/llms-full.txt),
[agent-manifest.json](site/agent-manifest.json).

## Identity

```text
pcq = open-source experiment evidence/control library
cq  = managed execution + orchestration + dashboard + agent loop
```

CQ service is one managed consumer of the contract. `pcq` remains useful without
CQ: locally, in CI, in notebooks, and inside third-party orchestrators.

## Why pcq

- **Framework-neutral** — keep the training stack that fits the problem.
- **Agent-readable** — use JSON/JSONL instead of terminal scraping.
- **Agent-verifiable** — validate source, config, environment, metrics,
  artifacts, and run records.
- **Agent-operable** — run, observe, validate, describe, compare, lineage, and
  iterate through stable commands.
- **Service-ready** — CQ can consume the same contract for managed execution and
  automatic experiment loops.

## Installation

```bash
uv add pcq
# Optional — to expose pcq as MCP tools to agent runtimes:
uv add 'pcq[mcp]'
```

`pyproject.toml`:

```toml
[project]
dependencies = ["pcq"]              # core only
# or:
dependencies = ["pcq[mcp]"]         # core + Model Context Protocol server
```

For a tag, branch, or private fork:

```toml
[tool.uv.sources]
pcq = { git = "https://github.com/playidea-lab/pcq.git", tag = "v4.1.0" }
```

The PyPI distribution, import name, CLI command, GitHub repository, runtime
workspace, and JSON contract namespace are all `pcq`. Runtime contract names
from CQ remain stable: `cq.yaml`, `CQ_CONFIG_JSON`, and `cq://`.

## Minimal Contract

`cq.yaml` declares the run:

```yaml
name: sklearn-baseline
cmd: uv run python train.py
configs:
  output_dir: output
  seed: 42
  strictness: 3
  monitor: eval_acc
  mode: max
metrics:
  - epoch
  - eval_acc
artifacts:
  - output/
inputs: {}
```

`train.py` can use any framework:

```python
import pickle

import pcq
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cfg = pcq.config()
out = pcq.output_dir()
pcq.seed_everything(cfg.get("seed", 42))

x, y = load_iris(return_X_y=True)
x_train, x_eval, y_train, y_eval = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=int(cfg.get("seed", 42)),
    stratify=y,
)

model = RandomForestClassifier(random_state=int(cfg.get("seed", 42)))
model.fit(x_train, y_train)
eval_acc = float(model.score(x_eval, y_eval))

with (out / "model.pkl").open("wb") as f:
    pickle.dump(model, f)

history = [{"epoch": 0, "eval_acc": eval_acc}]
pcq.log(**history[-1])
pcq.save_all(history=history, artifacts={"model": "model.pkl"})
```

No sklearn adapter is required. The same pattern works for HF Trainer,
Lightning, XGBoost, TabPFN, PyCaret, shell commands, or custom code.

## Agent Command Surface

Read and validate the project:

```bash
pcq resolve --json
pcq inspect . --json
pcq validate . --strictness 2 --json
```

Run the project:

```bash
pcq run --path . --json
pcq run --path . --jsonl
pcq run --path . --events output/events.jsonl --json
```

Validate and summarize outputs:

```bash
pcq validate-run output --strictness 3 --json
pcq describe-run output --json
pcq compare-runs old_output new_output --json
pcq lineage output --json
```

Iterate:

```bash
pcq apply-plan experiment.plan.json --json
```

Agent rule: prefer JSON/JSONL surfaces over scraping human output. `pcq`
reports facts; the agent or service chooses policy.

## Standard Artifacts

A completed run should produce:

- `config.json`
- `metrics.json`
- `manifest.json`
- `run_summary.json`
- `run_record.json`
- `validation_report.json`

`run_record.json` is the canonical completion object. It combines execution,
source, environment, input identity, metric schema, artifact manifest, agent
provenance, validation, and summary evidence.

## Agent Runtime Assets

`pcq` can install its canonical agent instructions and skill into a project.
Package installation itself never mutates project agent files.

```bash
pcq agent install --target codex --path .
pcq agent install --target claude --path .
pcq agent install --target both --path . --dry-run --json
pcq agent status --target both --path . --json
```

To also wire the project for MCP-aware agents (Claude Code, Codex), install
`pcq[mcp]` and pass `--mcp`:

```bash
uv add 'pcq[mcp]'
pcq agent install --target claude --path . --mcp     # writes .mcp.json
pcq mcp serve                                         # stdio (default)
```

This exposes 14 `mcp__pcq__*` tools (`resolve_project`, `validate_run`,
`describe_run`, `compare_runs`, ...) so agents call pcq directly without
subprocess parsing. See [MCP Integration](docs/MCP_INTEGRATION.md).

Reusable assets:

- [templates/AGENTS.pcq.md](templates/AGENTS.pcq.md)
- [skills/pcq/SKILL.md](skills/pcq/SKILL.md)

## v4 Direction

v4 clarifies the product boundary:

- contract-first workflow, not a 3-tier training API
- project-local training code, not built-in production catalogs
- contract scripts, not framework adapters
- run evidence validation, not recipe ownership
- JSON/JSONL facts, not prose parsing

See [pcq v4 Direction](docs/V4_DIRECTION.md).

## Development

```bash
uv run ruff check src/ tests/ scripts/
uv run python -m compileall src/pcq
uv run pytest tests/ -q
bash scripts/release-smoke.sh
```

## License

Apache-2.0.
