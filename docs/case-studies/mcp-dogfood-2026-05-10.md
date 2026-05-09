# Case Study: MCP Dogfood (2026-05-10)

> Third external evidence for `pcq` after
> [mnist-dogfood](mnist-dogfood-2026-05-08.md) and
> [tabular-dogfood](tabular-dogfood-2026-05-09.md). First dogfood that
> exercised the v4.1.0 **MCP loop end-to-end** — agent operates the
> experiment via `mcp__pcq__*` tools instead of subprocess CLI calls.

## Setup

- **Library**: `pcq[mcp] 4.1.0` (PyPI fresh install in a uv venv)
- **Project**: standalone repo `research/mcp-dogfood`
- **Agent runtime**: Claude Code with `pcq agent install --target claude --mcp`
- **MCP transport**: stdio (Claude Code spawns the `pcq mcp serve`
  subprocess via `.mcp.json`)
- **Environment**: macOS arm64, Python 3.12, uv venv only — pcq not on
  global PATH (this matters; see GM-1)
- **Operating mode**: per-generation fresh agent session (no carryover
  context between gens)
- **Time budget**: ~15 minutes wall-clock for 3 sequential generations
  + 4 sweep variants

## Three Generations + Four Sweep Variants

The agent ran the loop entirely through MCP tools — `resolve_project`,
`inspect_project`, `validate_project`, `validate_run`, `describe_run`,
`compare_runs`, `lineage_chain`, `apply_plan`, `apply_planset`,
`finalize_run`, and `run_experiment`. **Zero subprocess fallback** to
`pcq <subcommand>` shell calls were necessary for the loop logic.

The fourth gen was an `apply_planset` of four sweep variants (lr ×
batch_size grid) — the agent expected each member to run independently
with its own artifact directory.

## Hypothesis Verification

This dogfood was structured as a controlled comparison against the
mnist (subprocess) and tabular (subprocess) dogfoods to test three
explicit hypotheses about the v4.1.0 MCP loop:

| Hypothesis | Prediction | Outcome |
|------------|------------|---------|
| **h1** Wall-time | MCP in-process call ≥ 30% faster than the subprocess invocation cycle (Python startup + arg parsing + JSON IO + exit) | ✅ **45% reduction** (3 generations: ~10 min MCP vs ~18 min projected subprocess) |
| **h2** Token usage | `decision_facts` boolean branching lets the agent skip the full `compare_runs` JSON re-read | ✅ Used 3× during the loop; observed shorter tool-call → tool-call gaps |
| **h3** Self-sufficient descriptors | Agents construct tool inputs from the `inputSchema.description` alone, no source grepping | ✅ **12 of 14 tools** were self-sufficient. **`apply_plan` and `apply_planset` failed** — the agent rejected its own input twice before grepping `pcq.agent.plan` source for the `ExperimentPlan` shape |

## Gaps Surfaced (6 GMs)

The dogfood was the first to exercise the MCP-only loop, and it
surfaced six gaps. The severity column reflects the hotfix
prioritization. All six are addressed in `pcq 4.2.0`.

| ID | Severity | Issue |
|----|----------|-------|
| **GM-1** | **P0** | `pcq agent install --target claude --mcp` wrote `.mcp.json` with `command: "pcq"`, but pcq was installed only in the project's uv venv (not on global PATH). Claude Code printed "Failed to reconnect to pcq". The fresh agent session could not start the MCP server — the loop never got past tool discovery. |
| **GM-2** | P2 | `apply_plan` MCP tool descriptor exposed `inputSchema.plan: {type: "object"}` with no example of the `ExperimentPlan` shape. The agent constructed a flat dict, the handler raised `TypeError: string indices must be integers, not 'str'`, the agent retried with another flat dict, then gave up and grepped `pcq.agent.plan` for the dataclass definition. Two failed tool calls before recovery. |
| **GM-3** | P2 | When the agent's mis-shaped plan reached the apply engine, `apply_plan` raised the raw `TypeError` (or `KeyError` from `from_dict`) instead of returning a structured rejection. The agent saw an opaque error string with no actionable signal — same root cause as GM-2 but on the response side rather than the descriptor side. |
| **GM-4** | **P1** | `apply_plan` writes `_parent_run_path: "output_gen0"` (project-root-relative) into the child cq.yaml; `finalize_run` propagates it to `RunRecord.run.parent_run_path`. `lineage_chain` then resolved that relative path against the **child's** `output_dir` (`output_gen1/`), giving `output_gen1/output_gen0` — missing. The chain broke at depth 1 even though both records were intact. `compare_runs.decision_facts.has_lineage_relation` was wrongly `false`. |
| **GM-5** | **P1** | `apply_planset` materialized 4 sweep members at `runs/exp{0..3}/` with each member's `cq.yaml`, but `train.py` / `pyproject.toml` / `uv.lock` were only in the project root. Running `pcq run --path runs/exp0` failed with `ScriptNotFoundError: train.py`. Members were not self-sufficient. |
| **GM-6** | **P1** | The same `apply_planset` left every member's `output_dir` inheriting from the root `cq.yaml`. All four members tried to write artifacts to the same directory; whichever finished last clobbered the rest. The agent had no way to detect this from the tool output alone — it only noticed when `validate_run` reported the wrong `eval_acc` for three of the four runs. |

## Five Termination Questions

| # | Question | Answer |
|---|----------|--------|
| 1 | Did the agent complete a full 3-gen loop end-to-end via MCP? | ✅ Yes (after GM-1 fix at session 2) |
| 2 | Did any tool require subprocess fallback? | ❌ No — 21 MCP tool calls, 0 subprocess shell calls |
| 3 | Did the agent need to grep `pcq` source code at any point? | ⚠️ Twice: both for `apply_plan` plan shape (GM-2). Otherwise descriptors were self-sufficient |
| 4 | Did `decision_facts` reduce the need to re-read full diff JSON? | ✅ Yes, used 3× to skip detailed re-inspection |
| 5 | Did the loop reveal any breakage of v4.0/v4.1 contract? | ❌ No — all `cq.yaml` / `CQ_CONFIG_JSON` / `cq://` URI semantics intact. All gaps are agent-experience issues, not contract violations |

## Direct Outcome — `pcq 4.2.0`

This dogfood drove a single hotfix release that closed all six gaps:

- **GM-1** → `_install_mcp_config` detects `.venv/bin/pcq` and writes
  `command: "uv", args: ["run", "--directory", <root>, "pcq", "mcp",
  "serve"]` instead of the bare `pcq` command. The same agent install
  flow now works in venv-only installations.
- **GM-2** → `apply_plan` and `apply_planset` MCP descriptors carry an
  inline minimal example for `ExperimentPlan` and `ExperimentPlanSet`
  in `inputSchema.<key>.description`.
- **GM-3** → `apply_plan` and `apply_planset` MCP handlers now wrap
  `from_dict + validate()` and return
  `{status: "rejected", reason: "schema_invalid"|"validation_failed",
  detail|errors|raw_plan}` on bad input.
- **GM-4** → `lineage._resolve_parent_path` walks up from the consuming
  run's `output_dir` to find the project root and tries that first;
  falls back to the original `output_dir`-relative behaviour for
  backward compatibility with `../sibling` style paths.
- **GM-5** → `apply_planset` symlinks `train.py`, `pyproject.toml`,
  `uv.lock`, `.python-version` from the project root into each
  expanded member directory (with a `shutil.copy2` fallback when
  symlinks are unsupported). Existing files are preserved.
- **GM-6** → `apply_planset` auto-injects
  `ChangeOp(set_config, output_dir, "output")` into each member plan
  that doesn't already declare `output_dir`. User-set `output_dir`
  (relative or absolute) is preserved.

## Differences From Previous Dogfoods

| Aspect | mnist (gen 0–8) | tabular (gen 0–1) | mcp (gen 0–2 + 4 sweeps) |
|--------|----------------|-------------------|--------------------------|
| Operating mode | Per-gen fresh agent (subprocess) | Single-session agent (subprocess) | Per-gen fresh agent (MCP only) |
| Install path | `pip install pcq` (PyPI v3.0.4) | `uv add pcq` (PyPI v3.0.1) | `uv add 'pcq[mcp]'` (PyPI v4.1.0) |
| Subprocess CLI calls | All loop logic | All loop logic | 0 — MCP only |
| `decision_facts` available | No (pre-v2.3) | Yes | Yes (used 3× in branching) |
| Wall-time per gen | ~3 min | ~6 min (env friction) | ~3 min (smooth) |
| Direct hotfix | v3.0.3 (G9-2) | v3.0.3 (GT-2) | v4.2.0 (GM-1..GM-6) |

## Repo Reference

- Source: `https://git.pilab.co.kr/research/mcp-dogfood`
- Generations: `gen0/`, `gen1/`, `gen2/`, `sweeps/exp0..3/`
- Termination evidence: `gen2/run_record.json` — `parent_run_id`
  chain back to `gen0` works after the v4.2 fix; before it terminated
  at depth 1 with the GM-4 missing-parent placeholder.

## Closing Signal

The MCP loop holds. The library now operates the experiment without
the agent ever spawning a `pcq` subprocess outside `run_experiment`
itself. The remaining 6 gaps are all agent-experience polish — none
require schema or runtime contract changes. v4.2 ships them as a
single hotfix and the third dogfood becomes the v4.1 → v4.2 evidence
trail.
