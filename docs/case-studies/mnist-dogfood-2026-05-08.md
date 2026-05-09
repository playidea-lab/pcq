# Case Study: MNIST Dogfood (2026-05-08)

> First end-to-end dogfood of pcq as an agent-operated experiment contract
> library. 9 fresh agent generations, ML→DL evolution, eval_acc 0.9583 → 1.0.
> The first external evidence pcq has of its own value and friction.

## Setup

- Library: pcq v2.11.0 (commit 9136071, before any dogfood-driven fixes)
- Project: standalone repo `mnist-dogfood`, scaffolded via
  `pcq init-experiment --style script --with-pyproject`
- Dataset: `sklearn.datasets.load_digits()` (8×8, 1797 samples)
- Agent runtime: Claude Code with `pcq agent install --target claude`
- Operating mode:
  - gen 0: builder setup
  - gen 1–2: hybrid (fresh sub-agent decides plan, builder sanity-checks)
  - gen 3–8: pure per-gen fresh (sub-agent runs the full cycle, builder
    is emergency stop only)
- Budget: 5–7 days; actual: completed in one working session

## 9-Generation Evolution

| gen | axis | model / aug | eval_acc | wall |
|-----|------|-------------|----------|------|
| 0 | baseline | sklearn LogReg | 0.9583 | 0.04 s |
| 1 | sklearn nonlinear | SVC rbf | 0.9944 (ML co-best) | 0.01 s |
| 2 | sklearn ensemble | GradientBoosting | 0.9667 (regressed) | 5.7 s |
| 3 | ML → DL | torch MLP | 0.9889 | 3.5 s |
| 4 | DL spatial | CNN-2block | 0.9944 (DL co-best, 73 % fewer params than MLP) | 11 s |
| 5 | DL depth | DeeperCNN-3block | 0.9889 (regressed depth) | 22 s |
| 6 | optim/sched | CNN + OneCycleLR + AdamW | 0.9944 (tied) | 13 s |
| 7-b | augmentation (PlanSet 4-variant) | CNN + RandomAffine | 0.9972 (ceiling broken) | 16 s |
| 8 | TTA | CNN + Affine + TTA-7 | **1.0000 (360/360)** | 11 s |

The 9th generation reached perfect accuracy via test-time augmentation on top
of the gen 7 RandomAffine model. Two apparent ceilings (0.9944 from gens 1, 4,
6 and 0.9972 from gen 7) were each broken in turn.

## What pcq actually delivered (verified, not inferred)

- **Lineage chain depth 9** with `best_value` populated at every depth.
  Each fresh agent saw the full ancestor history and used it to decide the
  next axis without reading any prior agent's prompt.
- **`validate-run --strictness 3`** blocked any run missing reproducibility
  evidence (git sha, lockfile, cq.yaml sha256, environment).
- **`compare-runs direction`** (improved / regressed / tied) gave each
  agent an immediate signal whether the previous generation's axis was
  worth continuing. Three regressions (gen 2, gen 5, gen 7-d) were
  preserved in lineage and informed later decisions.
- **`pcq validate --plan`** caught a real bug (manifest stale lock-in)
  *before* training started in gen 1. Lightning + W&B would have failed
  silently or post-hoc.
- **Failed variants preserved as evidence**: gen 2 GBT and gen 5 DeeperCNN
  remained in the chain with `regressed` direction. Gen 7-d (RandomCrop)
  was visible alongside gen 7-b (RandomAffine) in PlanSet output.
- **Framework freedom** (sklearn → torch in gen 3) crossed without
  changing a single line of `pcq.config`/`pcq.log`/`pcq.save_all` —
  the contract held.
- **`apply-planset`** expanded 4 augmentation variants in one command
  with `parent_run_id` propagation. This is the first PlanSet usage in
  the wild.

## 21 ranked gaps surfaced

The dogfood was not a smoke test. It produced a precise, prioritized list
of friction points that 12 prior releases (v2.0 → v2.11) had not exposed
because they had zero users.

### P0 — fresh user blocked on first command

| ID | Issue |
|----|-------|
| **G7-5** | `pcq.config()` does not fall back to `cq.yaml.configs` when `CQ_CONFIG_JSON` is absent. Every gen needed manual env wiring. PlanSet expand multiplied the cost N×. |

### P1 — every user hits these

| ID | Issue |
|----|-------|
| G0-2 | `pcq run` command absent — fresh users have no first-class entry point. |
| G1-2 | Manifest stale lock-in when an `output_dir` is reused; `validate-run` fails on missing files even after legitimate cleanup. |
| G1-4 | `compare-runs` reports `config_changes=0` despite real cq.yaml.configs differences. |
| G7-1 | `apply-planset` writes relative `output_dir` paths, causing artifacts to land at `runs/genN/runs/genN/`. |

### P2 — recurring friction

`G0-3` (describe-run JSON missing best/artifacts), `G0-4` (output/ absorbs
runtime tmp files into manifest), `G1-1` (no plan op for editing train.py),
`G1-3 = G7-3` (apply-plan/planset doesn't sync artifacts glob with
output_dir), `G3-1 = G5-2` (apply-plan clobbers the inputs section), `G4-1`
(compare-runs returns -0.0 on tied float deltas), `G7-2` (validate
--planset exit code conflated with project-level fails), `G7-4`
(apply-planset doesn't deploy train.py to expanded dirs).

### P3 — informational

`G3-2 / G5-1 / G6-1` (structural ceiling hypothesis on tiny datasets),
`G8-1 / G8-2` (TTA timing measurement nuances).

## 5-question termination summary (agent's own answers)

1. **Setup vs Lightning + W&B** — pcq first run took ~15 min; W&B would
   start "instantly without G0-1 friction." Cumulative manual workarounds
   added 24–40 min across 8 gens.
2. **Stuck points** — 21 gaps, all surfaced by real use, none by inference.
3. **Five-evidence usage** — git_sha ✅, lockfile ✅, cq_yaml_sha256 ✅,
   inputs ❌ (sklearn builtin had no hash), **lineage ✅ — the most
   valuable single field, used by every fresh agent.**
4. **Plan decision time** — ~7–8 min/gen end-to-end (5 min plan write
   + 10 s apply + 2 min G0-1 workaround + train + 30 s × 3 validations).
5. **pcq vs alternative** — "G0-1 (CQ_CONFIG_JSON manual) across all
   8 gens is the #1 pain point. `pcq run` would eliminate it. PlanSet
   expand without train.py deploy and with relative output_dir is a
   metadata contract without an execution contract. If `pcq run` had
   existed in v2.11, ~30 % of manual overhead would disappear."

## Direct outcome — v2.12.0 hotfix

Within the same session the dogfood drove a release. v2.12.0 fixes:

- **G7-5/G0-1** [P0]: `pcq.config()` now falls back to `cq.yaml.configs`
- **G0-2** [P1]: new `pcq run` command
- **G1-2** [P1]: `validate-run --rescan-manifest` flag
- **G1-4** [P1]: `compare-runs` now diffs cq.yaml.configs
- **G7-1** [P1]: `apply-planset` normalizes relative `output_dir`

The 16 P2 + P3 gaps are scheduled for v2.13.

## Meta — the contradiction this case study breaks

> An evidence-first library cannot ship 12 releases with zero
> self-evidence and call its priorities anything other than inference.

The dogfood's purpose was not to validate the agent loop, the lineage
schema, or the strictness levels in isolation. It was to make the next
priority decision quantitative rather than recursive. The 21-gap list
above is now the actual driver of the v2.12 → v2.13 → v2.14 sequence,
replacing the earlier 7-phase ROADMAP that was authored before any
external use.

## Repo

- **mnist-dogfood**: <https://git.pilab.co.kr/research/mnist-dogfood>
- 11 commits, 9 RunRecords, 1 PlanSet, depth-9 lineage chain
- `dogfood-log.md` ~500 lines covering every gen with a uniform schema:
  Result | Decision basis | pcq commands | New gaps | Lightning+W&B
  comparison | Self-evaluation | End-of-gen decision

Reproducing the dogfood from scratch:

```bash
git clone https://git.pilab.co.kr/research/mnist-dogfood.git
cd mnist-dogfood
uv sync   # honors the locked pcq dependency in pyproject.toml
# A fresh project today would simply: uv add pcq. The open-source
# runtime contract library now ships under the `pcq` PyPI name to avoid
# collision with the occupied `cq` package name and the managed CQ service.
# Each output_dir (output/, output_gen1/.../output_gen8/, runs/gen7-{a..d})
# contains a complete RunRecord chain. Inspect with:
pcq lineage output_gen8 --json
pcq describe-run output_gen8 --json
```

## References

- `.cq/runtime/ideas/pcq-mnist-dogfood.md` — original `/pi` idea + termination append
- `docs/PCQ_COMPLETION_ROADMAP.md#dogfood-findings` — gap list embedded in roadmap
- `CHANGELOG.md` — v2.12.0 entry calls out each dogfood-driven fix
- mnist-dogfood/dogfood-log.md — per-gen primary source
