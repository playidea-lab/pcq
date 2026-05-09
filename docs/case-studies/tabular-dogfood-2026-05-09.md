# Case Study: Tabular Dogfood (2026-05-09)

> Second external evidence for `pcq` after [mnist-dogfood](mnist-dogfood-2026-05-08.md).
> First post-PyPI dogfood on `pcq 3.0.1` — fresh `uv add pcq` install path,
> different domain (tabular), different framework diversity attempt
> (TabPFN / PyCaret / FLAML / XGBoost / sklearn HistGBM).

## Setup

- **Library**: `pcq 3.0.1` (PyPI fresh install — no git URL workaround)
- **Project**: standalone repo `tabular-dogfood`, scaffolded via
  `pcq init-experiment --style script`
- **Dataset**: `sklearn.datasets.load_breast_cancer()` (569 samples × 30 features)
- **Agent runtime**: Claude Code with `pcq agent install --target claude`
- **Environment**: macOS arm64, Python 3.12.10
- **Operating mode**: builder direct (single-session dogfood; not per-gen
  fresh-agent pipeline like mnist)
- **Time budget**: ~35 minutes wall-clock

## Two Generations

| gen | model | eval_acc | train_time | result |
|-----|-------|---------|----------|--------|
| 0 | sklearn LogReg | **0.9649** | 0.011 s | baseline |
| 1 | sklearn HistGradientBoosting | 0.9561 | 2.26 s | regressed -0.009 |

Gen 1 was **the 5th framework attempt**; the first four hit environmental
friction unrelated to `pcq` itself. The full attempt trace is the
case study's most informative artifact.

### Gen 1 — Five Framework Attempts

| # | framework | outcome | root cause |
|---|-----------|---------|------------|
| 1 | TabPFN v7.1.1 (PriorLabs v2 series) | `TabPFNLicenseError` | external API token required (`TABPFN_TOKEN`) |
| 2 | PyCaret 3.3.2 | `RuntimeError` | only supports Python 3.9–3.11 (we run 3.12) |
| 3 | FLAML AutoML 2.6.0 | exit 139 (SIGSEGV) | lgbm/xgb native lib + macOS arm64 |
| 4 | XGBoost 2.1.4 (direct) | exit 139 (SIGSEGV) | macOS arm64 + Python 3.12 |
| 5 | **sklearn HistGradientBoosting** | ✅ accepted | pure-Python sklearn impl |

The dogfood spent ~30 of its 35 minutes on environmental friction outside
`pcq`'s scope. That is itself the dogfood's most concrete signal:
the **library's framework-agnostic claim** holds, but the **practical**
agent experience is dominated by upstream framework compatibility, which
`pcq` neither owns nor diagnoses today.

## What `pcq` Actually Delivered

- ✅ **`uv add pcq`** — single-line PyPI install (v3.0.1's docs simplification
  goal landed end-to-end).
- ✅ **`pcq agent install --target claude`** — `CLAUDE.md` and
  `.claude/skills/pcq/SKILL.md` written without manual edit.
- ✅ **`pcq init-experiment --style script`** — `cq.yaml` and `train.py`
  scaffold, ran without intervention.
- ✅ **`pcq run --path .`** — manual `CQ_CONFIG_JSON` workaround count: **0**
  (vs mnist gen 0–8 that needed it every time before v2.12). The
  v2.12.0 G0-1 / G0-2 fixes ship correctly in v3.0.1.
- ✅ **Failure capture** — exit codes 1 (license / Python-version) and 139
  (segfault) recorded by `pcq run`'s JSON envelope correctly.
- ✅ **`pcq apply-plan plans/gen-1.json`** — `cq.yaml` updated with
  `_parent_run_id` injected automatically.
- ✅ **`pcq validate-run --strictness 3`** — gen 0 and gen 1 both pass with
  full reproducibility evidence (git sha, lockfile, cq.yaml sha, env).
- ✅ **`pcq compare-runs gen0 gen1 --json`** — `metric_direction: regressed`,
  `a_is_ancestor_of_b: true`, `decision_facts` populated immediately.
- ✅ **`pcq lineage`** — chain depth 2, `best_value` populated at every depth.

## What `pcq` Did Not Solve

| ID | Severity | Issue |
|----|----------|-------|
| **GT-1** | P3 | macOS arm64 + Python 3.12 native ML library compatibility friction (xgboost, lightgbm). 4 of 5 framework attempts hit this. **Out of `pcq`'s scope** — but a fresh user meets it on day one. A pre-flight environment check in `pcq init-experiment` could surface the issue before `train.py` runs. |
| **GT-2** | **P1** | `pcq compare-runs A B` returned `config_changes=[]` even though gen 0 → gen 1 had five `set_config` ops (output_dir, model, hgb_max_iter, hgb_max_depth, hgb_lr). Root cause: `cq.yaml` on disk holds the latest state, so reading it for both runs yields the same dict. **This is a re-confirmation of [G9-2](mnist-dogfood-2026-05-08.md#g9-2) from the mnist dogfood; second independent discovery escalated it from P3 to P1.** Fixed in `pcq 3.0.3`. |
| **GT-3** | P2 | TabPFN's `TabPFNLicenseError` and PyCaret's Python-version `RuntimeError` are raised at import time, before `pcq.save_all` is reached. The result is no `RunRecord` written — `pcq run` records `exit_code: 1` in its JSON envelope, but the structured failure envelope from v2.11 doesn't activate. Either `pcq run --json` could synthesize a fallback `RunRecord` for import-time failures, or the `init-experiment` template could include a `try/except` wrapper that calls `pcq.save_all(failure=...)` before re-raising. |

## Direct Outcome — `pcq 3.0.3`

GT-2 was the first dogfood-driven hotfix from this case study:

- `pcq compare-runs` now falls back to `output_dir/config.json` (a snapshot
  written by `pcq.save_config_snapshot()` for every run) when the
  `cq.yaml`-based diff comes back empty.
- Provenance keys (`_git_sha`, `_pcq_version`, etc.) are filtered.
- 3 regression tests added (`tests/test_compare_runs_config.py`).
- v3.0.3 published to PyPI (commit `c9b6048`).

The fix path used by GT-2 — **two independent dogfoods → severity escalation
→ targeted patch with regression tests** — is the loop the case study
demonstrates is now operational.

## Differences from the MNIST Case Study

| dimension | mnist (2026-05-08) | tabular (2026-05-09) |
|----------|---------|---------|
| `pcq` version at start | v2.11.0 (pre-fix) | v3.0.1 (post v2.12 + v3.0 fixes) |
| install path | git URL | PyPI `uv add pcq` |
| operating mode | per-gen fresh agent (sub-agent dispatch) | builder direct |
| evolution depth | 9 generations | 2 generations |
| total wall-clock | ~7–8 min/gen × 9 = ~65 min | ~35 min total (5 attempts) |
| dominant time cost | plan writing + manual `CQ_CONFIG_JSON` workaround | environment friction (4/5 attempts) |
| new gaps surfaced | 21 ranked (P0×1, P1×4, P2×8, P3×4 + info) | 3 (GT-1 P3, GT-2 P1, GT-3 P2) |
| ceiling broken | 0.9583 → 1.0000 (TTA) | 0.9649 → 0.9561 (regressed) |

The tabular case study is **shorter and shallower than mnist by design**.
It was not run to maximize new gap discovery — it was run to verify
that the v2.12.0 → v3.0.1 fixes from the mnist dogfood actually solve
the friction they targeted. Result: they do, with one residual gap
(GT-2 / G9-2) re-surfacing because two-dogfoods is what was needed
to escalate it from P3 to P1.

## Repo

- **tabular-dogfood**: <https://git.pilab.co.kr/research/tabular-dogfood>
- 2 commits, 2 RunRecords, 1 lineage chain
- `dogfood-log.md` covers gen 0 + 5 framework attempts + 3 new gaps

Reproducing:

```bash
git clone https://git.pilab.co.kr/research/tabular-dogfood.git
cd tabular-dogfood
uv sync   # honors locked pcq dependency
pcq lineage output_gen1 --json
pcq compare-runs output_gen0 output_gen1 --json   # config_changes detected (post v3.0.3)
```

## What This Case Study Proves

1. **PyPI install path works**. The v3.0 single-name + PyPI publish move
   produced an external user experience that requires zero git URL,
   zero environment variable wiring, zero manual CLI cobbling.
2. **The dogfood feedback loop closes**. mnist exposed G9-2 at P3.
   Tabular re-exposed it at GT-2 and forced a P1 hotfix (v3.0.3) within
   the same session. The library now responds to its own evidence.
3. **Framework-agnostic ≠ environment-agnostic**. `pcq` is honest about
   not owning framework compatibility, but a fresh user spends most of
   their first hour on it. Pre-flight diagnostics is the next leverage
   point, not yet on the roadmap.

## References

- `docs/case-studies/mnist-dogfood-2026-05-08.md` — first dogfood, source
  of the 21-gap inventory that drove v2.12.x → v3.0.x.
- `docs/CQML_COMPLETION_ROADMAP.md` (renamed `PCQ_COMPLETION_ROADMAP.md`
  in v3.0) — Dogfood Findings section.
- `tabular-dogfood/dogfood-log.md` — primary source per-gen and per-attempt.
- CHANGELOG v3.0.3 entry — fix derived from this case study.
