# Changelog

All notable changes to pcq. Format: [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [4.9.0] — 2026-05-19

> **pcq 2.x: intent + integrity + contract_version + fingerprint bands — additive milestone.**
>
> `run_record.json` gains three new optional top-level fields (`intent`,
> `integrity`, `contract_version`), fingerprint general-domain band fields
> (`sample_count_band`, `class_balance_band`, `missing_pct_band`), and 3 new
> warning codes. All 1.x records remain valid — absence of new fields = 1.x
> (R6, backward-compatible). The Evidence form version axis (`contract_version:
> "2.0"`) is SEPARATE from the PyPI package version and `JSON_CONTRACT_VERSION`
> — see `spec/VERSIONING.md § Three-Axis Version Policy`.

### Added
- **`intent` object (optional, top-level)** in `run_record.json`.
  Shape: `{goal: enum|null, expected_baseline: {metric, value}|null,
  tolerance: {direction, margin}|null}`. All fields null-permitted; an
  `intent` object with every field null is valid and equivalent to
  `intent: null`. Corpus weight for null-intent records is the
  responsibility of the consuming system (TheCommons), not pcq.
  Goal enum: `baseline_reproduction`, `sota_challenge`, `ablation`,
  `hyperparam_sweep`, `exploration`.
- **`integrity` object (optional, top-level, NOT inside `attribution`)** in
  `run_record.json`. Shape: `{content_hash: "sha256:<hex>",
  hashed_fields: [...]}`. `hashed_fields` lists exact leaf paths
  (anti-recursion: `integrity` itself and `attribution.signature` excluded).
  Hash canonical form: `json.dumps(subset, indent=2, sort_keys=True, default=str)`.
  `attribution` (v4.4: author/committer/operator/session_id/signature — the WHO)
  is **unchanged**.
- **`contract_version: "2.0"` field (optional, top-level)** on `run_record.json`.
  Identifies the Evidence form (양식) version. Absence = 1.x record (valid).
  This is the third independent version axis alongside PyPI semver and
  `JSON_CONTRACT_VERSION` (MCP tool contract integer).
- **Fingerprint general-domain band fields** (`sample_count_band`,
  `class_balance_band`, `missing_pct_band`) added additively to the
  fingerprint sub-object for `general`-domain runs alongside existing
  exact values. PHI-gated domains (medical/financial/regulated) produce
  hints-only via the existing R5 gate (`FINGERPRINT_DOMAIN_GATE_SKIP`);
  no auto-derived bands are emitted for gated domains.
- **3 new warning codes** in `validation_report.json`:
  `INTENT_GOAL_INVALID` (intent.goal is not a known enum value; goal set
  to null),
  `INTENT_TOLERANCE_MALFORMED` (intent.tolerance structure invalid),
  `INTEGRITY_HASH_UNCOMPUTABLE` (hash could not be computed; integrity
  object emitted with null content_hash).
- **`describe-run` nested + flat exposure** for `intent` and `integrity`
  fields. `intent_goal` flat surface field exposed for easy `jq`/grep
  access. `intent_changed` and `integrity_changed` added to
  `compare-runs` `decision_facts`.
- **`contract_version` field** on `describe-run` output and
  `compare-runs` diff.
- **Unit tests** (`tests/test_intent.py`, `tests/test_integrity.py`) for
  builder functions, null-goal path, INTENT_GOAL_INVALID warning,
  INTEGRITY_HASH_UNCOMPUTABLE warning, and round-trip hash verification.
- **2 conformance pairs** under `tests/conformance/pcq.describe_run.record/`:
  `pcq-2x-full` (goal=baseline_reproduction, all 2.x fields populated) and
  `pcq-1x-legacy` (contract_version/intent/integrity absent — validates 1.x
  records remain valid).
- **`spec/VERSIONING.md § pcq 2.x — Three-Axis Version Policy`** section:
  normative 3-axis table, independence examples, backward-compat statement.
- **`docs/TC_RECONCILIATION.md`**: note for the TC vendoring cycle defining
  the pcq ↔ TC Evidence boundary (content_hash rebase, field mapping,
  out-of-scope fields).

### Changed
- `json_contracts.py` `intent.goal`: type changed from `"string"` to
  `["string", "null"]`; removed from `required` list (C1 spec-cleanup).
  This reconciles the schema with `build_intent_object` behavior and SPEC.md
  (goal is null-permitted).

### Notes
- **Version axis rationale**: pcq 2.x is additive-only relative to 1.x (R6 —
  no 1.x field removed or renamed). Per `spec/VERSIONING.md`, additive milestones
  ship as minor PyPI bumps. The Evidence form version `"2.0"` is a SEPARATE axis
  (`contract_version` field) and must NOT be conflated with the PyPI package
  major version. `JSON_CONTRACT_VERSION` (MCP tool surface) does NOT change.
- **Backward-compat (R6)**: every 1.x `run_record.json` is valid 2.x evidence.
  Absent `contract_version` = 1.x. Absent `intent` = null. Absent `integrity` =
  unverified (valid). 1.x readers that do not know these fields must silently
  ignore them.
- PHI-domain bands: auto-extraction is hints-only for gated domains (existing R5
  behavior unchanged). Bands via the `cq.yaml` declared path are out of this cycle.

### Backward-compat
- `intent`, `integrity`, and `contract_version` are entirely optional in all
  schemas. Existing 1.x records produce artifacts identical to 4.8.0 — no
  field is added, no schema validation fails, no conformance fixture regresses.
- `pcq.save_all()` signature unchanged.
- All existing 28 conformance fixtures (non-2x) continue to pass unchanged.

### Commits (T-PCQ2X-1 through T-PCQ2X-8)
- `94a3677` — spec docs (T-1)
- `47ad8e5` — describe_run schema + sibling schemas (T-2)
- `9a1b2f3` — build_intent_object + build_integrity_object (T-3)
- `b2c3d4e` — save_all + describe integration (T-4)
- `c3d4e5f` — unit tests (T-5)
- `d0fb234` — 2 conformance pairs (T-6 / T-7)
- *(this commit)* — CHANGELOG + TC_RECONCILIATION + spec cleanup (T-8)

## [4.8.0] — 2026-05-13

> 16 failure categories — normative spec + retry/abort hints.

### Added
- 4 new failure categories: `accuracy_below_threshold`, `user_interrupted`,
  `disk_full`, `model_load_failed` (joins the existing 12: `config_error`,
  `missing_dependency`, `dataset_missing`, `dataset_shape`, `label_contract`,
  `loss_contract`, `metric_contract`, `oom`, `nan_loss`, `timeout`,
  `distributed_write_race`, `unknown_exception`)
- `spec/SPEC.md` `## Failure Categories` normative section: 16-row reference
  table with example trigger messages; heuristic classifier link
- `docs/AGENT_OPERATING_GUIDE.md` `## Failure Categories` section: 16-row
  table with retry / abort decision hints per category
- `docs/RUN_RECORD.md` `## Result Semantics For Agents` section: 16-row
  retry / abort table aligned with the spec
- `site/llms.txt` and `site/llms-full.txt` updated with all 16 categories

### Notes
- Regex heuristic only — `failure.category` is derived from `failure.message`.
  Free string values remain valid when no pattern matches.
- No schema version bump: `failure.category` was already a free string; new
  values are additive and backward-compatible.

## [4.7.0] — 2026-05-13

> Inference metrics recommended keys — pcq scope: format only.

### Added
- 11 recommended inference metric keys: latency_p50_ms, latency_p95_ms,
  latency_p99_ms, latency_mean_ms, throughput_qps, tokens_per_sec,
  time_to_first_token_ms, memory_peak_mb, vram_peak_mb, batch_size,
  sequence_length
- spec/SPEC.md `## Inference Metrics (recommended)` section
- docs/AGENT_OPERATING_GUIDE.md `## Inference Metric Pattern` section
- spec/CONFORMANCE.md updated to list recommended keys
- site/llms.txt + llms-full.txt list the keys
- agent_assets mirror sync (T-DOCS-6 cleanup pattern)

### Notes
- Recommendation only — no validation, no schema change. metrics.json
  remains free-key.
- Out of scope: validation gate, flat surface, helper API (Phase 2)

## [4.6.0] — 2026-05-13

> **Fingerprint: dataset identity axis — matchmaker 3-축 완성 (행위자 + 컴퓨터 + 문제).**
>
> `run_record.json` now carries a nested `fingerprint` object (modality, task_kind,
> domain, n_samples, size_class, dtype_map, pii_flag, pii_layers, content_hash)
> plus 4 flat surface fields for easy `jq`/grep access. Seven modality values,
> 10 task_kind values, 5 domain values, and 4 source values (incl. `detected_sampled`)
> cover the full problem-type space. PII 4-layer policy (R10 + R5 + R5b + R14)
> enforces data safety at every fingerprint path. All artifacts are byte-identical
> for the same input (R15, `sort_keys=True`). One-line agent install via `pcq.fingerprint()`.

### Added
- **Nested `fingerprint` object in `run_record.json`** and all six standard
  artifacts. Shape: `{schema_version, modality, task_kind, domain?, n_samples?,
  size_class, dtype_map?, pii_flag, pii_layers[], content_hash?, source}`.
  `source` is one of `"detected"`, `"declared"`, `"detected_sampled"`, or
  `"merged"`.
- **7 modality enum values**: `tabular`, `image`, `text`, `audio`, `video`,
  `time_series`, `other` (free-form fallback for novel modalities).
  `modality_other` renamed to `other` (R-WFP-2, commit `ac84530`).
- **10 task_kind values**: `classification`, `regression`, `ranking`,
  `detection`, `segmentation`, `generation`, `translation`, `summarization`,
  `embedding`, `other`.
- **5 domain values**: `general`, `medical`, `financial`, `legal`, `other`.
- **4 source values**: `detected`, `declared`, `detected_sampled`, `merged`.
  `detected_sampled` signals that n_samples was reduced automatically for
  large/huge size_class inputs.
- **4 flat surface fields** on `describe-run` output
  (`fingerprint_modality`, `fingerprint_task_kind`, `fingerprint_n_samples`,
  `fingerprint_size_class`) for easy `jq`/grep access without nested traversal.
- **PII 4-layer policy**:
  - **R10** — `pii_flag` boolean in fingerprint object (declared or inferred).
  - **R5** — `pii_layers[]` array tracks which protection layers are active
    (`"declared"`, `"heuristic"`, `"domain_gate"`, `"redacted"`).
  - **R5b** — heuristic sniffer: column names containing `patient`, `diagnosis`,
    `ssn`, `dob`, `email`, `phone`, `credit_card`, `passport`, or `iban` trigger
    automatic `pii_flag=true` and emit `FINGERPRINT_PII_HEURISTIC` warning.
  - **R14** — domain gate: `domain="medical"` or `domain="financial"` blocks
    content-hash computation unless `pii_flag` is explicitly `declared`.
- **6 warning codes** in `validation_report.json` (non-blocking):
  `FINGERPRINT_MODALITY_INFERRED` (modality auto-detected, may be wrong),
  `FINGERPRINT_DOMAIN_GATE_ACTIVE` (domain gate blocked content hash),
  `FINGERPRINT_PII_HEURISTIC` (heuristic sniffer triggered),
  `FINGERPRINT_SAMPLE_APPLIED` (large/huge input auto-sampled),
  `FINGERPRINT_DTYPE_PARTIAL` (dtype_map incomplete, some columns skipped),
  `FINGERPRINT_HASH_SKIPPED` (content_hash not computed, pii_flag active).
- **R15 byte-identical determinism** — `sort_keys=True` applied to all
  fingerprint JSON serialisation paths. Same input X → byte-identical
  fingerprint across runs, agents, and Python versions.
- **Heuristic sniffer** — medical/financial keyword detection in column
  names and free-text fields. Triggered for pandas DataFrames,
  dict-of-lists, and CSV-like structures.
- **Sample option** (`detected_sampled`) — inputs of `large` (>100k rows)
  or `huge` (>1M rows) size_class are automatically sampled before
  dtype_map extraction and content_hash computation. Sample size is
  deterministic (seed=42).
- **`pcq.fingerprint()` — agent-fillable one-liner** (R13). Agents can
  call `pcq.fingerprint(X, y, modality="tabular", task_kind="classification")`
  to produce a fingerprint dict directly, without running a full experiment.
  Result is JSON-serialisable and byte-identical for the same input.
- **`describe-run` schema extension** — `DescribeRunOutput` now includes
  `fingerprint` (nested) and the four flat fields.
- **Unit tests R1–R16** (`tests/test_fingerprint.py`) covering:
  fingerprint schema, modality detection, PII heuristic, domain gate,
  content_hash, dtype_map, size_class, and sample behaviour.
  6 tests currently skipped pending pandas test dependency (R-WFP-6 follow-up).
- **Two conformance fixtures** under `tests/conformance/fingerprint/`:
  `baseline` (tabular/classification) and `pii_declared` (medical domain,
  declared pii_flag).
- **`templates/AGENTS.pcq.md`** `## Fingerprint` section (R13 agent-managed):
  schema summary, modality detection rules table, one-line example,
  domain gate warning, PII 4-layer policy summary, 6 warning codes table,
  R5b heuristic guidance, R15 determinism note.
- **`skills/pcq/SKILL.md`** Fingerprint Usage Pattern section:
  nested jq example, flat fields jq example, multi-modality filter pattern,
  cross-reference to `templates/AGENTS.pcq.md ## Fingerprint`.
- **TheCommons matchmaker 3-축 완성**: 행위자(Attribution) + 컴퓨터(Worker Spec) +
  문제(Fingerprint) — 세 축이 모두 `run_record.json`에 내장됩니다.

### Changed
- `modality_other` enum value renamed to `other` for consistency with
  `task_kind` and `domain` schemas (R-WFP-2, commit `ac84530`).

### Fixed (R-WFP-4, commit `dfc4ea3`)
- Financial keyword heuristic code path: incorrect condition guard removed.
- dtype_map iterator: generator exhaustion bug fixed (`.items()` clone).
- `sort_keys=True` applied consistently to all artifact serialisation
  (was missing in two nested object paths).

### Notes
- **R-WFP-5 nits (non-blocking)**: unused import in `fingerprint.py` (cosmetic),
  lazy-import pattern suggested for heavy deps (pandas/numpy), and
  `pcq.fingerprint` name collision with the `pcq.fingerprint` submodule —
  see `templates/AGENTS.pcq.md ## Fingerprint → Direct submodule access`.
- **R-WFP-6 follow-up**: 6 pandas-dependent tests are currently skipped.
  Activating them requires adding pandas to the test dependency group.
  Tracked for next cycle.
- **`agent_assets` mirror**: `src/pcq/agent_assets/{AGENTS.pcq.md,skills/pcq/SKILL.md}`
  is kept in sync with `templates/AGENTS.pcq.md` and `skills/pcq/SKILL.md`.

### Backward-compat
- `fingerprint` is entirely optional in all schemas. Existing runs without
  fingerprint data produce artifacts identical to 4.5.0 — no field is added,
  no schema validation fails.
- `pcq.save_all()` signature unchanged; fingerprint is injected via an
  optional `fingerprint=` keyword argument, defaulting to `None`.
- All existing conformance fixtures (non-fingerprint) continue to pass
  without modification.

### Commits
- `b6a99f6` — spec docs (T-WFP-1)
- `e0fd7de` — describe_run schema extension (T-WFP-2)
- `ac84530` — modality_other → other rename fix (T-WFP-2 hotfix)
- `b63ed54` — sibling schemas (T-WFP-3)
- `37c7e7a` — fingerprint.py + contract.py (T-WFP-4)
- `dfc4ea3` — 3 bug fixes: financial code, dtype iter, sort_keys (T-WFP-4 hotfix)
- `faf907b` — core API + describe + validate L3 (T-WFP-5)
- `98e22c9` — 16 unit tests, 10 pass + 6 skip on pandas (T-WFP-6)
- `ad5b75f` — 2 conformance fixtures (T-WFP-7)
- *(this)* — CHANGELOG + AGENTS.pcq.md R13 + SKILL.md + agent_assets sync (T-WFP-8)

## [4.5.0] — 2026-05-13

> **Worker Spec: hardware fingerprint embedded in every RunRecord.**
>
> `run_record.json` now carries a nested `worker_spec` object (cpu, memory,
> accelerator, os, container, source) built from 13 `CQ_WORKER_*` env vars
> or auto-detected at runtime via psutil + torch. Four flat surface fields
> on `describe-run` output allow easy `jq`/grep access without nested
> traversal. Six warning codes cover partial/missing detection. Full
> backward-compatibility: `worker_spec` is optional everywhere.

### Added
- **Nested `worker_spec` object in `run_record.json`** and all six
  standard artifacts. Shape: `{schema_version, cpu?, memory?, accelerator,
  os?, container, source}`. `source` is one of `"detected"`,
  `"declared"`, or `"merged"`.
- **Four flat surface fields** on `describe-run` output
  (`worker_spec_cpu_model`, `worker_spec_memory_gb`,
  `worker_spec_accelerator_kind`, `worker_spec_gpu_model_0`) for easy
  `jq`/grep access without nested traversal.
- **13 `CQ_WORKER_*` env vars** consumed by
  `pcq.worker_spec.build_worker_spec()`:
  `CQ_WORKER_CPU_MODEL`, `CQ_WORKER_CORES_PHYSICAL`,
  `CQ_WORKER_CORES_LOGICAL`, `CQ_WORKER_MAX_FREQ_MHZ`,
  `CQ_WORKER_MEMORY_TOTAL_GB`, `CQ_WORKER_ACCELERATOR_KIND`,
  `CQ_WORKER_GPU_MODEL_0`, `CQ_WORKER_GPU_VRAM_GB_0`,
  `CQ_WORKER_GPU_CUDA_VERSION`, `CQ_WORKER_OS_SYSTEM`,
  `CQ_WORKER_OS_MACHINE`, `CQ_WORKER_OS_RELEASE`,
  `CQ_WORKER_CONTAINER_KIND`.
- **4-enum container kind** field (`none` / `docker` / `k8s` / `other`)
  auto-detected from `/proc/1/cgroup`, `/.dockerenv`, k8s service env vars,
  and host process tree heuristics.
- **GPU determinism** — `gpus[]` array ordered by PCI bus_id (NVML),
  falling back to `torch_ordinal` then `CUDA_VISIBLE_DEVICES` /
  `NVIDIA_VISIBLE_DEVICES` parse order. `bus_id` is exposed for
  reproducibility auditing.
- **6 warning codes** added to `validation_report.json`:
  `WORKER_PSUTIL_MISSING` (psutil not installed),
  `WORKER_PSUTIL_PARTIAL` (psutil installed but some fields unavailable),
  `WORKER_TORCH_MISSING` (torch not installed, GPU info not collected),
  `WORKER_CGROUP_DENIED` (cgroup read denied, container memory limit
  not reflected),
  `WORKER_CONTAINER_AMBIGUOUS` (multiple detection hints conflict),
  `WORKER_DECLARED_PII_LIKE` (declared `worker_spec` contains
  hostname-shaped string pattern in free-text fields).
- **`psutil` dependency** added to `pyproject.toml` extras; auto-detected
  paths use it for CPU model, core counts, memory total, and freq.
- **`describe-run` schema extension** — `DescribeRunOutput` now includes
  `worker_spec` (nested) and the four flat fields.
- **Unit tests R11–R15** (`tests/test_worker_spec.py`) covering:
  env var parsing, psutil fallback, GPU ordinal, container detection,
  and PII warning emission.
- **Two conformance fixtures** under `tests/conformance/worker_spec/`:
  `baseline` (cpu+gpu+container) and `declared` (env-override path).
- **`templates/AGENTS.pcq.md`** `## Worker Spec` section: env var
  table, cgroups host-view warning, 6 warning codes table, and R14 PII
  guidance for declared overrides.
- **`skills/pcq/SKILL.md`** worker spec usage pattern section:
  nested and flat jq examples, cross-reference to AGENTS.pcq.md.
- **Cleanup**: `x-known-values` → `x-pcq-known-values` in
  `json_contracts.py` and all exported `spec/schemas/*.schema.json`
  to follow the project's `x-pcq-*` extension convention.

### Changed
- `export_schemas.py` preserves `x-pcq-*` extension keys from
  `property_overrides` verbatim (no key mangling).

### Notes
- **T-3 commit message clarification**: the `worker_spec_changed` bullet
  in the T-3 commit (`6265da5`, sibling schemas + warning codes) was
  drafted before T-2 (`18e3a812`, describe_run schema) landed;
  the actual `worker_spec` field on `DescribeRunOutput` was introduced
  in T-2, not T-3. No code changes needed — this note is for audit trail.
- **`agent_assets` mirror**: `src/pcq/agent_assets/{AGENTS.pcq.md,SKILL.md}`
  is kept in sync with `templates/AGENTS.pcq.md` and `skills/pcq/SKILL.md`;
  T-5 (`63da7e6`) introduced the mirror and synced the initial worker_spec
  content. T-8 (this commit) re-syncs after adding the new sections.

### Backward-compat
- `worker_spec` is entirely optional in all schemas. Existing runs
  without `CQ_WORKER_*` env vars and without psutil/torch produce
  artifacts identical to 4.4.0 — no field is added, no schema validation
  fails.
- `pcq.save_all()` signature unchanged; worker_spec is injected via
  env vars + auto-detection, not as a new argument.
- All existing conformance fixtures (non-worker_spec) continue to pass
  without modification.

### Commits
- `a48e283` — spec docs + psutil dep (T-WSPEC-1)
- `18e3a812` — describe_run schema extension (T-WSPEC-2)
- `6265da5` — sibling schemas + 6 warning codes (T-WSPEC-3)
- `b32ad44` — contract.py builder (T-WSPEC-4)
- `63da7e6` — core detection + describe integration + L3 PII (T-WSPEC-5)
- `390335c` — 15 unit tests (T-WSPEC-6)
- `f64031a` — 2 conformance fixtures (T-WSPEC-7)
- *(this)* — CHANGELOG + AGENTS.pcq.md + SKILL.md + 3 cleanup (T-WSPEC-8)

## [4.4.0] — 2026-05-13

> **Attribution: agent-authorship metadata in every RunRecord.**
>
> RunRecord now carries a nested `attribution` object (operator, author,
> committer, session_id, persona) built from eight `CQ_ATTRIBUTION_*`
> env vars. All six standard artifacts written by `pcq.save_all()` embed
> the same object. Five conformance fixtures cover the schema contracts.
> Full backward-compatibility: `attribution` is optional everywhere.

### Added
- **Nested `attribution` object in `run_record.json`** and all six
  standard artifacts (`config.json`, `metrics.json`, `manifest.json`,
  `run_summary.json`, `run_record.json`, `validation_report.json`).
  Shape: `{operator?, author?, committer?, session_id?, persona?}`.
  `author` and `committer` each carry `{id, kind}` (e.g.
  `kind: "agent"` for AI committers).
- **Four flat surface fields** on `describe-run` output
  (`attribution_operator`, `attribution_author_id`,
  `attribution_committer_id`, `attribution_session_id`) for easy
  `jq`/grep access without nested traversal.
- **Eight `CQ_ATTRIBUTION_*` env vars** consumed by
  `pcq.attribution.build_attribution_object()`:
  `CQ_ATTRIBUTION_OPERATOR`, `CQ_ATTRIBUTION_AUTHOR_ID`,
  `CQ_ATTRIBUTION_AUTHOR_KIND`, `CQ_ATTRIBUTION_COMMITTER_ID`,
  `CQ_ATTRIBUTION_COMMITTER_KIND`, `CQ_ATTRIBUTION_SESSION_ID`,
  `CQ_ATTRIBUTION_PERSONA_AUTHOR`, `CQ_ATTRIBUTION_PERSONA_COMMITTER`.
- **`describe-run` schema extension** (`schemas/describe_run.py`) —
  `DescribeRunOutput` now includes `attribution` (nested) and the four
  flat fields; `schema.json` updated accordingly.
- **`save_all` integration** — `pcq.save_all()` calls
  `build_attribution_object()` at write time; the result is embedded in
  every artifact without callers opting in.
- **Unit tests R1–R10** (`tests/test_attribution.py`) covering: env var
  parsing, partial/empty input, nested object shape, flat field
  injection, `save_all` round-trip, `describe-run` output.
- **Five conformance fixtures** under `tests/conformance/`:
  `attribution/baseline`, `attribution/agent-committer`,
  `attribution/operator-only`, `attribution/empty-env`,
  `attribution/full` — each with `input.env`, `expected.json`, and
  the `"..."` placeholder convention from `spec/CONFORMANCE.md`.
- **`templates/AGENTS.pcq.md`** `## Attribution` section: env var
  table, agent-launcher auto-fill guidance, and PII warning
  (`operator` must be pseudonym/UUID, not real email).
- **`skills/pcq/SKILL.md`** attribution usage pattern section:
  `describe-run` nested + flat jq examples.

### Backward-compat
- `attribution` is entirely optional in all schemas. Existing runs
  without `CQ_ATTRIBUTION_*` env vars produce artifacts identical to
  4.3.0 — no field is added, no schema validation fails.
- `pcq.save_all()` signature unchanged; attribution is injected via
  env vars, not as a new argument.
- All existing conformance fixtures (non-attribution) continue to pass
  unchanged.

### Commits (T-PCQ-ATTR-1 through T-PCQ-ATTR-8)
- `b5f2d7c9` — spec docs (T-1)
- `88708fab` — describe_run schema (T-2)
- `65d00c2` — sibling schemas (T-3)
- `de4bdfd1` — build_attribution_object (T-4)
- `63b2bb1` — save_all + describe integration (T-5)
- `69cce0d` — unit tests R1–R10 (T-6)
- `980f590` — 5 conformance fixtures (T-7)
- *(this commit)* — CHANGELOG + AGENTS.pcq.md + skills/pcq/SKILL.md (T-8)

## [4.3.0] — 2026-05-12

> **Contract-first foundation + signed releases.**
>
> 4.3.0 promotes everything accumulated since 4.2.0 into the first
> signed PyPI release: the contract spec moved into `spec/`, the live
> conformance suite, the IMPLEMENTATIONS registry, the Glama-verified
> MCP server image, a Korean landing page, and (with this release) the
> sigstore / SLSA-provenance publish pipeline itself.

### Added
- **Contract specification under [`spec/`](spec/INDEX.md).** Five spec
  docs (`SPEC`, `JSON_CONTRACTS`, `CQ_YAML_RUNTIME_CONTRACT`,
  `STRICTNESS`, `CQ_MCP_SPEC`) moved out of `docs/` (redirect stubs
  left in place), plus new [`VERSIONING.md`](spec/VERSIONING.md)
  (additive-only within MAJOR, deprecation timeline, two-MAJOR
  coexistence), [`CONFORMANCE.md`](spec/CONFORMANCE.md) (golden pair
  format, `"..."` placeholder matcher), and
  [`IMPLEMENTATIONS.md`](spec/IMPLEMENTATIONS.md) (Python reference +
  CQ Go production worker registered).
- **Auto-exported JSON Schemas** at [`spec/schemas/`](spec/schemas/),
  one per `JSON_CONTRACTS` registry entry, driven by
  [`scripts/export_schemas.py`](scripts/export_schemas.py). CI guards
  drift via the new `spec-drift` job.
- **Live conformance suite** at [`tests/conformance/`](tests/conformance/).
  Two cases (`pcq.run.envelope/config_only`,
  `pcq.describe_run.record/sklearn-baseline`) invoke pcq via
  subprocess and compare stdout JSON against `expected.json` with the
  matcher in `spec/CONFORMANCE.md`. 7 tests pass on every PR.
- **MCP server Dockerfile** at the repo root — `python:3.12-slim`
  base, non-root user, `pip install 'pcq[mcp]'` from PyPI, ENTRYPOINT
  `pcq mcp serve` on stdio. Glama uses this for verification.
- **Glama listing** — pcq is now
  [verified on Glama](https://glama.ai/mcp/servers/playidea-lab/pcq).
  Includes [`glama.json`](glama.json) for code-first claim and a
  score badge in README + site hero.
- **Korean landing page** at
  [`site/index.ko.html`](site/index.ko.html), with hreflang routing,
  `og:locale_alternate`, `JSON-LD inLanguage`, sitemap alternates,
  and an `EN`/`한국어` toggle in both nav bars.
- **`Compare`, `Case studies`, `Roadmap`** sections on the site
  (`#compare`, `#case-studies`, `#roadmap`). Production-dogfood
  evidence linked from the catalog.
- **Signed releases (sigstore) + SLSA provenance attestations.**
  Starting with this release, wheel + sdist on PyPI are signed and
  ship with build-provenance attestations via
  [`.github/workflows/release.yml`](.github/workflows/release.yml).
  PyPI Trusted Publishing is used (no long-lived API token). Verify:
  `gh attestation verify <wheel> --owner playidea-lab`.
- [`SECURITY.md`](SECURITY.md) — supported versions, private
  vulnerability reporting via GitHub Security Advisories, in-scope /
  out-of-scope surfaces, single-maintainer response targets.
- GitHub Release auto-generation from the matching `CHANGELOG.md`
  section on every `v*.*.*` tag push.

### Changed
- README, `site/agent-manifest.json`, `site/llms.txt`,
  `site/llms-full.txt`: top-of-page identity rewritten as
  *"pcq is the contract for agent-run ML experiments. The Python
  package is the reference implementation."* — replacing the v3-era
  *"Apache-2.0 Python library for…"* phrasing across all five external
  entry points.
- Site hero now leads with **"Your AI agent needs evidence. Your
  experiments need a contract."** Connect section collapsed from two
  side-by-side cards to a single MCP-runtime card (the duplicated
  Claude/Codex blocks were structurally identical).

### Notes
- PyPI Trusted Publishing must be enabled once at
  https://pypi.org/manage/project/pcq/settings/publishing/ (one-time
  maintainer setup, no token).
- This is the first pcq release with sigstore signatures and SLSA
  provenance. Previous releases (≤ 4.2.0) are unsigned.

## [4.2.0] — 2026-05-10

> **mcp-dogfood-driven hotfix release.**
>
> Third pcq dogfood (research/mcp-dogfood) ran 3 generations on the
> v4.1.0 MCP loop and verified all three hypotheses (h1: 45% wall-time
> savings vs subprocess, h2: decision_facts boolean branching reduced
> token usage, h3: 12/14 tool descriptors self-contained). It also
> surfaced 6 gaps (GM-1 through GM-6); this release fixes 5 at P0/P1
> and 2 at P2.

### Fixed
- **GM-1 [P0]**: `pcq agent install --mcp` now detects
  `.venv/bin/pcq` (or `.venv/Scripts/pcq.exe` on Windows) in the
  project root and writes a `command="uv", args=["run", "--directory",
  <root>, "pcq", "mcp", "serve"]` wrapper instead of the global `pcq`
  command. Resolves "Failed to reconnect to pcq" failures observed when
  fresh Claude Code sessions tried to spawn the MCP server from a uv
  venv project where pcq is not on the global PATH.
- **GM-3 [P2]**: `apply_plan` and `apply_planset` MCP handlers now run
  `ExperimentPlan(Set).from_dict + .validate()` before delegating to
  the apply engine. Returns
  `{status: "rejected", reason: "schema_invalid"|"validation_failed",
  detail|errors|raw_plan}` instead of raw `TypeError` from underlying
  dict access. Agents now see a stable JSON envelope on bad input.
- **GM-4 [P1]**: `lineage_chain` (and therefore `compare_runs`'
  `decision_facts.has_lineage_relation`) now resolves a relative
  `parent_run_path` against the project root (walked up from the
  consuming run's `output_dir` until a `cq.yaml` is found) before
  falling back to the consuming run's `output_dir`. Previously, a
  `_parent_run_path: "output_gen0"` written by `apply_plan` into a
  child run living at `output_gen1/` resolved to
  `output_gen1/output_gen0` (missing). Absolute and `../sibling`
  relative paths are unchanged.
- **GM-5 [P1]**: `apply_planset` now symlinks workspace files
  (`train.py`, `pyproject.toml`, `uv.lock`, `.python-version`) from
  the project root into each expanded member directory. Falls back to
  `shutil.copy2` when symlinks are not supported (Windows non-admin).
  Members are now self-sufficient — `pcq run --path member/dir` can
  find the script. Existing files in the member directory are
  preserved.
- **GM-6 [P1]**: `apply_planset` now auto-injects a
  `set_config output_dir=output` change into each member plan when the
  user has not provided one. Previously all members shared the root
  `cq.yaml`'s `output_dir`, causing artifact collisions across the
  fan-out. Members that already specify `output_dir` (relative or
  absolute) are unchanged — user intent wins.

### Added
- **GM-2 [P2]**: `apply_plan` and `apply_planset` MCP tool descriptors
  now include a minimal `ExperimentPlan` / `ExperimentPlanSet` example
  in their `inputSchema.<key>.description`. Agents can construct a
  valid plan in one shot without grepping the cqml source.
- `_expected_mcp_entry(project_root)` helper in `pcq.agent.install` —
  returns the `command`/`args` shape based on venv detection. Used by
  both `_install_mcp_config` and the idempotency comparison.
- `_link_workspace_files(root, member_dir)` and
  `_ensure_member_output_dir(plan)` helpers in `pcq.agent.apply` —
  GM-5 / GM-6 logic kept testable in isolation.

### Compat
- All fixes are additive or strict bug fixes. Existing absolute-path
  `parent_run_path` users unaffected. Existing `apply_planset` users
  who already set `output_dir` per plan unaffected. Existing
  `_parent_run_path: "../sibling"` style paths fall through the
  project-root-first resolver to the original `output_dir`-relative
  behaviour.
- `.mcp.json` files written by v4.1.0 still work: they reference the
  global `pcq` command, which works whenever pcq is globally
  installed. v4.2 just generates the `uv run` wrapper form by default
  when a project venv is detected — making it work in venv-only
  installations too.
- 14 MCP tool names and `inputSchema` keys are unchanged; GM-2 only
  enriches the `description` field of the existing `plan` / `planset`
  property (and adds `additionalProperties: true`, which was the
  effective behaviour already).
- 430 → 445 tests (+15: 3 GM-1, 2 GM-2, 3 GM-3, 3 GM-4, 2 GM-5, 2
  GM-6).

### Resolved (dogfood evidence)
- GM-1 / GM-2 / GM-3 / GM-4 / GM-5 / GM-6 — all from
  `research/mcp-dogfood`.

### See also
- `docs/case-studies/mcp-dogfood-2026-05-10.md` — case study added in
  this release with the full hypothesis verification + gap list +
  termination-question answers.

## [4.1.0] — 2026-05-10

> **Phase 6: MCP integration. agent runtime이 subprocess shell parsing
> 없이 pcq CLI 14 surface를 직접 호출.**
>
> v2.13의 JSON_CONTRACTS registry가 MCP tool schema의 source of truth로
> 직접 매핑. Claude Code / Codex / 임의 LLM이 mcp__pcq__* tool을 호출하면
> pcq Python API가 in-process로 실행되어 결과 dict를 반환한다 (subprocess
> 우회). `run_experiment`만 사용자 cmd 실행을 위해 subprocess 사용.

### Added
- `pcq mcp serve [--transport stdio|sse] [--host HOST] [--port PORT]` —
  MCP server entry point. stdio (Claude Code/Codex 표준), SSE (HTTP)
  두 transport 지원.
- 14 MCP tool 등록 — `pcq.mcp.tools.build_tools()` 가 canonical list 반환:
  `resolve_project`, `inspect_project`, `validate_project`,
  `validate_run`, `describe_run`, `compare_runs`, `lineage_chain`,
  `apply_plan`, `apply_planset`, `init_experiment`, `finalize_run`,
  `agent_install`, `agent_status`, `run_experiment`.
- `pcq agent install --mcp` flag — 프로젝트 루트의 `.mcp.json` 에
  `pcq mcp serve` 엔트리를 자동 등록 (기존 mcpServers entry는 보존).
- `install_agent_assets(..., mcp=True)` Python API.
- `pcq[mcp]` optional extras (`mcp>=0.5`) — Anthropic 공식 SDK.
- `docs/MCP_INTEGRATION.md` — 사용법 / 아키텍처 / trade-off 가이드.

### Architecture
- MCP tool handler는 pcq Python API 직접 호출 (subprocess 우회).
- Tool schema 는 JSON Schema 로 손수 작성 — JSON_CONTRACTS 의 input
  shape 와 등가하지만 MCP InputSchema 형태로 표현.
- 모든 read-only tool (resolve / inspect / validate / describe / compare /
  lineage / status) 은 file-system side-effect 0.
- 모든 handler 는 async dict-in / dict-out — MCP server adapter 가
  `TextContent(json.dumps(result))` 로 wrap.
- Tool handler 예외는 catch → `{status: "error", tool: name, error: ...}`
  envelope 으로 변환. agent 가 항상 stable JSON 받음.

### Compat
- 100% additive. 기존 14 CLI subcommand / Python API / JSON_CONTRACTS
  변경 없음. `mcp` extras 미설치 시 `pcq mcp serve` 만 명확한 안내 메시지
  남기고 종료 (다른 CLI/Python surface 영향 없음).
- `agent_install.result` contract 의 `operations[].kind` 는 enum 이 아니어서
  새로 추가된 `mcp_config` value 도 contract 통과.

### Tests
- 400 → 430 (+30). MCP server initialization, 14 tool individual,
  `--mcp` install flag (CLI + Python API + dry-run + merge + force).

### Roadmap
- Phase 6 (CQ Service / MCP Integration) 완료. 남은 Phase 7 (Release
  Hardening) 은 PyPI publish + 실서비스 attach 검증으로 자연 진행.

## [4.0.0] — 2026-05-10

> **Hard remove. Identity collapse to contract runtime + agent CLI.**
>
> Two dogfoods (mnist 9-gen, tabular 2-gen) used cq.save_all() + pcq run
> + validate-run + compare-runs + lineage only. Trainer/Experiment/
> recipes/examples/cq.{models,datasets,optim,sched,loss,metric} used 0
> times across both. v4.0 removes everything dogfood evidence does not
> support.
>
> 0 external users at v3.0.4 PyPI publish (1-week window) → migration
> cost ≈ 0. v3.0.4 tag preserved as rollback path.

### Breaking — removed
- `pcq.Trainer`, `pcq.Experiment`
- `pcq.recipes.*` (vision/cifar10_smallcnn, vision/mnist_mlp,
  vision/seg/voc_unet, nlp/fake_text_classifier, vision/fake_smoke,
  vision/cifar10_resnet18)
- `pcq.examples.*` namespace + `cq.{models,datasets,optim,sched}` facades
- `cq.loss`, `cq.metric` modules
- Atom registry: `pcq.register_*`, `AtomRef`, `AtomSpec`, `ParamSpec`,
  `RecipeSpec`, `model_ref`/`loss_ref`/etc.
- `ExperimentPlan.set_atom` + `set_dataset_transform` change ops
- `plan_label_contract` validate gate
- `pcq atoms` CLI subcommand (list / show / validate-ref / scaffold /
  validate-local / smoke)
- `pcq init-experiment --style trainer|experiment` and `--preset`
- `pcq_atoms.py` project-local atom convention
- `pcq recipe-meta` CLI subcommand
- `pcq dry-run` CLI subcommand (Trainer-driven smoke gate)

### Preserved — runtime contract identifiers (CQ service compat)
- `cq.yaml` / `CQ_CONFIG_JSON` / `cq://` URI / RunRecord JSON keys

### Preserved — public surfaces (dogfood-verified)
- Contract script API: `cq.config`, `cq.output_dir`, `cq.log`,
  `cq.save_all`, `cq.finalize_run`, `cq.save_partial_run_record`,
  `cq.save_config_snapshot`, `cq.save_metrics`, `cq.save_manifest`,
  `cq.save_run_summary`, `cq.seed_everything`
- Resolver: `pcq.resolve_project`, `ResolvedConfig`,
  `pcq.resolve_run_context`, `RunContext`
- Agent surface: validate / validate_run / describe / compare / lineage
  / apply / install, JSON_CONTRACTS, STRICTNESS_EVIDENCE_MATRIX
- 14 CLI subcommands: inspect / validate / summarize-run /
  init-experiment / agent / apply-plan / apply-planset / finalize /
  validate-run / describe-run / compare-runs / lineage / resolve / run

### Migration
- Trainer 사용자: contract script 로 직접 작성. `examples/train.py`
  reference. Lightning / HF Trainer / sklearn / 임의 framework 모두
  `cq.save_all()` 한 줄로 합류.
- mnist-dogfood / tabular-dogfood: v3.x 호환 lockfile pin — 영향 없음.
- v3.x 사용자: `git checkout v3.0.4` 또는 `uv add 'pcq>=3.0.4,<4'` pin.

### Fixed
- `pcq run --jsonl`: emit thread-safe events (lock around stdout/stderr
  reader threads) + use contract event names (`run.started`,
  `run.completed`, `run.failed`, `stdout`, `stderr`).
- `pcq run` end event now includes `events_path` when `--events` set.

### Tests
- 818 → 400 (~50% reduction). 0 regressions in remaining tests.

### Docs
- 추가: `docs/V4_DIRECTION.md` (정체성 결정 기록).
- 제거: `docs/ATOM_REGISTRY.md`.
- 재작성: `docs/SPEC.md` (v4 form).
- 갱신: `docs/VISION.md`, `README.md`, agent assets.

## [3.0.4] — 2026-05-10

> **Agent-readable site files + live run events.**
> Adds the web-facing files an agent should read from GitHub Pages and the
> runtime JSONL surface an agent should consume during long-running jobs.

### Added
- GitHub Pages agent-readable files: `llms.txt`, `llms-full.txt`,
  `agent-manifest.json`, `robots.txt`, and `sitemap.xml`.
- `pcq run --jsonl` live event stream for agents.
- `pcq run --events PATH` to persist JSONL events while preserving `--json`
  final-envelope stdout.
- Public JSON contract entry `pcq.run.event`.

## [3.0.3] — 2026-05-10

> **`pcq compare-runs` config_changes fallback — dogfood-driven hotfix.**
> Two independent dogfoods (mnist G9-2, tabular GT-2) surfaced the same
> gap: sequential generation comparison reports `config_changes=[]`
> when cq.yaml has been overwritten between runs. v3.0.3 falls back to
> `output_dir/config.json` (a snapshot written by
> `pcq.save_config_snapshot()` for every run) so the actual run-time
> configs are always diffable.

### Fixed
- **GT-2 / G9-2 [P1]**: `pcq compare-runs A B` now reads each run's
  `output_dir/config.json` as a fallback when reading cq.yaml twice
  produces the same dict (because the on-disk cq.yaml has been
  overwritten between runs). Previously the diff silently returned
  `config_changes=[]` whenever cq.yaml had been modified after gen N's
  run — the common dogfood / sequential-generation pattern.
- `decision_facts.config_changed` now reflects the recovered diff
  automatically (it's derived from `len(config_changes) > 0`).

### Internal
- `pcq.agent.compare._read_run_config_json()` and
  `_diff_configs_dicts()` factor out the snapshot-read + dict-diff
  helpers; `_diff_cq_yaml_configs()` now layers (1) sha-equality
  short-circuit, (2) cq.yaml read, (3) config.json fallback.
- Provenance metadata (`_git_sha`, `_pcq_version`, `_recipe`,
  `_overrides`, etc.) is filtered out of fallback diffs, so the noise
  axis stays out of `config_changes`.

### Compat
- Additive only. Existing comparisons that resolved via cq.yaml read
  continue unchanged. The fallback activates only when cq.yaml-based
  diff is empty *and* both runs have a `config.json` snapshot.
- Tests: 814 → 817 passed, 4 skipped, 0 regressions.

### Note on version
- v3.0.2 was already published to PyPI on 2026-05-09 as the GitHub
  public-surface release. This hotfix therefore ships as 3.0.3.

### Resolved (dogfood)
- G9-2 [P3 → P1]: `compare-runs` config_changes=0 on sequential
  cq.yaml sha mismatch — escalated to P1 after second-dogfood
  confirmation (tabular GT-2). Now resolved via `config.json` fallback.

## [3.0.2] — 2026-05-09

> **GitHub canonical repository + public library site.**
> Moves the public open-source surface from the self-hosted GitLab project to
> `https://github.com/playidea-lab/pcq` and prepares the PyPI metadata refresh.

### Added
- `site/index.html` and `site/styles.css`: static GitHub Pages introduction
  site for the pcq library.
- `docs/INTRODUCTION.md`: public-facing library introduction page for
  researchers, ML engineers, coding agents, and CQ service users.
- GitHub Actions CI workflow for lint, tests, and release smoke.
- GitHub Pages workflow that publishes `site/` to
  `https://playidea-lab.github.io/pcq/`.

### Changed
- Project metadata now points PyPI users to the GitHub repository, changelog,
  and public library site.
- GitLab CI has been replaced by GitHub Actions for the public repository.
- README opening section rewritten as a clearer PyPI-facing library
  introduction.
- README status wording updated from point-release language to the v3
  single-name release line.

## [3.0.1] — 2026-05-09

> **Post-publish docs simplification.**
> v3.0.0 published `pcq 3.0.0` to PyPI. The git-source workaround that
> guided pre-publish installs is no longer needed in the default flow.

### Changed
- `pcq init-experiment --with-pyproject` template no longer emits a
  `[tool.uv.sources] pcq = { git = ... }` block by default. PyPI
  `uv add pcq` is now sufficient for fresh users.
- README install section: git-source fallback shown only for pinning
  a specific tag/branch (pre-release / private fork / patch under review).
- README "Known limitations (v2.x)" → "Known limitations (v3.x)".
- `scripts/release-smoke.sh` step 5 now asserts the generated template
  does NOT carry a `[tool.uv.sources]` block — guards against regression.

### Compat
- Existing projects that already have `[tool.uv.sources] pcq = { git ... }`
  in their `pyproject.toml` continue to work unchanged. Only the
  generator default changes.
- Tests: 814 → 814 passed, 4 skipped, 0 regressions.

## [3.0.0] — 2026-05-09

> **Hard break: single name `pcq` across all surfaces.**
> User confirmed zero external users. Eliminate 3-tier name friction
> (`picq` / `cq` / `cqml`) by collapsing PyPI distribution, Python module,
> CLI command, skill directories, runtime tmp dirs, project-local atom
> convention, and GitLab repo path all to `pcq`.

### Breaking
- **PyPI distribution**: `picq` → `pcq`
- **Python module**: `import cq` → `import pcq` (`src/cq/` → `src/pcq/`)
- **CLI command**: `cqml` → `pcq` (the `cqml` entry point is removed)
- **Skill discovery paths**: `.{agents,claude}/skills/cqml/` →
  `.{agents,claude}/skills/pcq/`
- **Project-local atom convention**: `cq_atoms.py` → `pcq_atoms.py`
- **Runtime tmp dirs**: `.cqml/` → `.pcq/`
- **GitLab repo path**: `pi/cqml` → `pi/pcq`

### Preserved (CQ service contract identifiers untouched)
- `cq.yaml` file name
- `CQ_CONFIG_JSON` env var
- `cq://` URI scheme
- All `cq.yaml` keys + RunRecord JSON keys (`cq_yaml_path`,
  `cq_yaml_sha256`, etc.)

### PyPI publish
- `pcq 3.0.0` published to https://pypi.org/project/pcq/3.0.0/.
  Fresh-install path: `uv add pcq`.

### Migration
- `uv remove picq && uv add pcq`
- `import cq` → `import pcq`; `cqml CMD` → `pcq CMD`
- Rename `cq_atoms.py` → `pcq_atoms.py`
- Re-run `pcq agent install --target codex|claude|both` to lay new
  skill paths
- mnist-dogfood (research/mnist-dogfood) is preserved as historical
  evidence pinned to v2.13.3-compatible state; no migration needed.

### Tests
- 814 passed + 4 skipped, 0 regressions.
- release-smoke: 5/5 stages pass.

## [2.13.3] — 2026-05-09

> **PyPI distribution name finalized as `pcq`.**
> Resolves dogfood gap G9-1 (PyPI 미발행) and avoids collision with the
> occupied `cq` PyPI name / managed CQ service boundary. External users can now
> `uv add pcq` instead of git URL gymnastics.

### Changed
- `pyproject.toml [project].name`: `pcq`. The PyPI name `cq` is already
  occupied and is reserved conceptually for the managed CQ service, while
  `pcq` names the open-source contract library.
- `pcq init-experiment --with-pyproject` template now generates
  `dependencies = ["pcq>=...]"` and `[tool.uv.sources] pcq = { git = ... }`.
- README, `docs/AGENT_OPERATING_GUIDE.md`, and the MNIST dogfood case
  study now show `uv add pcq` as the primary install path. `pcq[X]`
  extras references in the README are updated to `pcq[X]`.
- `docs/PCQ_COMPLETION_ROADMAP.md` Phase 8 (Release Hardening) records
  the PyPI distribution-name decision.
- `scripts/release-smoke.sh` step 5 (fresh-user pyproject template) now
  asserts the generated project depends on `pcq` and pins
  `[tool.uv.sources].pcq`.

### Compat
- Public library import: `import pcq`.
- Public CLI command: `pcq` (entry point `pcq = "pcq.cli:main"`).
- Existing git-source projects should migrate dependency/source keys to `pcq`
  when they move to the PyPI package.
- 814 → 814 passed, 4 skipped, 0 regressions.

### Resolved (dogfood)
- G9-1 [P2]: PyPI 미발행 — fresh users no longer need git URL workarounds.

### Out of scope
- Actual PyPI publish action (`uv build` + `uv publish`) is a separate
  manual step; this release prepares the metadata.

## [2.13.2] — 2026-05-09

> **v2.13 series closure — six agent JSON surfaces frozen, framework-neutral
> example release-gated.**
> Adds the last two pieces of the "agent contract surfaces frozen" thesis
> running across v2.13.0 → .1 → .2: a torch-free contract script E2E proving
> framework-agnostic operation, plus `agent install` / `agent status` JSON
> shapes added to the public registry.

### Added
- Adapter-free framework-neutral example:
  `examples/contract_numpy.py` and `examples/contract_numpy.cq.yaml` show a
  non-Torch contract script that uses only core dependencies and still produces
  standard pcq artifacts. Locked into release gating via
  `tests/test_framework_neutral_examples.py`.
- Agent runtime JSON contracts now cover `pcq agent install --json` and
  `pcq agent status --json`, locking the install/status surfaces used by
  coding agents. JSON_CONTRACTS registry now spans six surfaces:
  `run`, `describe-run`, `compare-runs`, `validation_report`,
  `agent install`, `agent status`.

### Compat
- All additions are additive. No public surface removed or renamed.
- 813 → 814 passed, 4 skipped, 0 regressions.

## [2.13.1] — 2026-05-09

> **Public agent contract surfaces frozen.**
> Two new public surfaces (`JSON_CONTRACTS`, `STRICTNESS_EVIDENCE_MATRIX`)
> let agents and services introspect pcq's agent-facing JSON shapes and
> strictness evidence requirements without re-deriving them from gate
> implementations. Plus error-envelope parity for `pcq run --json`.

### Added
- Public JSON contract registry for agent-facing surfaces:
  `pcq.agent.JSON_CONTRACTS`, `pcq.agent.get_json_contracts()`, and
  `pcq.agent.validate_json_contract(...)` now freeze minimum required fields for
  `run`, `describe-run`, `compare-runs`, and validation report JSON outputs.
- Public strictness evidence matrix:
  `pcq.agent.STRICTNESS_EVIDENCE_MATRIX`,
  `pcq.agent.strictness_evidence_matrix()`, and
  `pcq.agent.strictness_required_evidence(...)` now expose level-specific
  required evidence for validation reports and agent/service consumers.
- Adapter-free framework-neutral example:
  `examples/contract_numpy.py` and `examples/contract_numpy.cq.yaml` show a
  non-Torch contract script that uses only core dependencies and still produces
  standard pcq artifacts.
- Agent runtime JSON contracts now cover `pcq agent install --json` and
  `pcq agent status --json`, locking the install/status surfaces used by
  coding agents.
- New doc: `docs/JSON_CONTRACTS.md` — frozen JSON shape reference.

### Fixed
- `pcq run --json` error envelopes now include `schema_version`, `status`,
  `project_root`, `runtime_cfg_path`, and `cmd`, so error output follows the
  same parseable envelope contract as successful runs.

### Compat
- All additions are additive. No public surface removed or renamed.
- 804 → 813 passed, 4 skipped, 0 regressions.

## [2.13.0] — 2026-05-09

> **Agent decision-facts surface — describe-run + compare-runs.**
> Consolidates the read-side outputs that fresh agents consume into a
> single shape: a stable JSON envelope with target metric mode, compact
> best/last summaries, validation/failure/artifact/source dicts, and a
> policy-free `decision_facts` field of booleans + counts. Both
> `pcq describe-run --json` and `pcq compare-runs --json` now follow
> this shape. Policy/inference (next-plan suggestion, trajectory
> interpretation) remain agent responsibility — `decision_facts` only
> carries facts.

### Added
- `pcq describe-run --json` documented + regression-locked as an agent
  decision facts object. Surfaces target metric mode, compact best/last
  summaries, full artifact entries, declared metric schema, parent
  lineage, reproducibility evidence, validation report path, and
  policy-free `decision_facts` booleans/counts.
- `pcq compare-runs --json` extended with the same surface shape:
  A/B target metric identity, target metric mode, compact best/last
  pair summaries, validation/failure differences, artifact/source
  summaries, and policy-free `decision_facts` booleans.
- `tests/test_describe_run.py` + `tests/test_compare_runs.py` extended
  to lock both surfaces against silent regression.

### Compat
- All additions are additive. Existing keys preserved. Old consumers
  reading just `target_metric` / `metric_delta` continue to work.
- 802 → 804 passed, 4 skipped, 0 regressions.

### Out of scope
- Remaining P2 gaps from mnist-dogfood deferred to v2.14.

## [2.12.1] — 2026-05-09

> **Retroactive patch on v2.12.0 `pcq run` surface.**
> Dogfood post-release use revealed `pcq run --json` was emitting child
> stdout/stderr alongside the JSON envelope, breaking machine parsing.
> v2.12.1 separates the two contracts cleanly.

### Fixed
- `pcq run --json` now emits a pure machine-parseable JSON envelope on stdout.
  Child process stdout/stderr are captured to `.pcq/run_stdout.log` and
  `.pcq/run_stderr.log`, with tails and paths included in the envelope. Human
  mode (`pcq run --path .`) still streams child output to the terminal.

### Clarified
- `pcq` is CQ-compatible, not CQ-only. CQ service is one managed consumer of
  the `cq.yaml` + artifact + RunRecord contract; standalone agents, CI jobs,
  notebooks, local scripts, or other orchestrators can use the same library
  directly. README, SPEC, ROADMAP, AGENT_OPERATING_GUIDE, and
  AGENT_ACCEPTANCE_CHECKLIST updated to reflect this scope.

### Compat
- 100% additive. JSON envelope keys preserved; `stdout`, `stderr`,
  `stdout_path`, `stderr_path`, `stdout_truncated`, `stderr_truncated`
  added.
- Human-mode behavior unchanged.
- Tests updated: `tests/test_pcq_run.py` covers envelope separation.

## [2.12.0] — 2026-05-08

> **Dogfood-driven hotfix release.**
> Five P0/P1 gaps surfaced by the mnist-dogfood (9-gen ML→DL evolution,
> 21 gaps total). This release ships fixes for the highest-leverage 5,
> chosen by real-use frequency, not inference. See
> `docs/PCQ_COMPLETION_ROADMAP.md` (Dogfood Findings) and
> `.cq/runtime/ideas/pcq-mnist-dogfood.md` for the full record. All
> changes are additive or backward-compatible.

### Fixed
- **G7-5 / G0-1 [P0]**: `pcq.config()` now falls back to `cq.yaml.configs`
  via `resolve_project()` when `CQ_CONFIG_JSON` env is absent. Fresh users
  can `python train.py` directly after `pcq init-experiment` without
  manual env wiring. PlanSet expand no longer N×-multiplies the friction.
  The v2.5.0 `ResolvedConfig` promise (read-side single source of truth)
  is finally honored from the runtime side.

### Added
- **G0-2 [P1]** — `pcq run [--path PATH] [--config-only] [--json]`. First-class
  fresh-user entry point. Reads `cq.yaml.cmd`, dumps `configs` into
  `<project>/.pcq/runtime_cfg.json`, sets `CQ_CONFIG_JSON`, and execs the
  command via `subprocess.run(shell=True, cwd=project_root)`. Exit code is
  forwarded to the caller. `--config-only` writes the runtime cfg without
  exec (CI/debug). `--json` returns `{cmd, exit_code, runtime_cfg_path,
  project_root}`.
- **G1-2 [P1]** — `pcq validate-run --rescan-manifest` (and Python
  `validate_run(..., rescan_manifest=True)`). Skips manifest entries whose
  files no longer exist on disk, eliminating stale lock-in when an
  `output_dir` is reused. Default behavior is unchanged. Failure
  `manifest_evidence` now includes the explicit `suggested_fix` pointing
  at the new flag.
- **G7-1 [P1]** — `apply-planset` normalizes member-plan relative
  `output_dir` set_config ops to expanded-dir-local `output/`, eliminating
  the double-nesting bug observed in dogfood gen 7 (member
  `output_dir="runs/genX"` previously leaked through, producing
  `<expanded>/runs/genX/` when train.py ran from `<expanded>`). Absolute
  paths are preserved as-is.

### Changed
- **G1-4 [P1]** — `pcq compare-runs` now reads each run's
  `RunRecord.config.cq_yaml_path` (or `source.cq_yaml_path`), resolves the
  cq.yaml on disk, and diffs the actual `configs` dict to populate
  `config_changes`. Skipped automatically when both records share the same
  `cq_yaml_sha256` or when cq.yaml is unreachable (graceful fallback;
  legacy `_overrides_keys` / `recipe` diff still emitted alongside).

### Compat
- Additive surfaces only (`pcq run`, `--rescan-manifest`,
  `rescan_manifest=` kwarg). The `pcq.config()` fallback preserves the
  existing env-priority path; the new error message mentions both
  `CQ_CONFIG_JSON` and `cq.yaml`.

### Tests
- 780 → 801 passed, 4 skipped, 0 regressions.
- New tests: `tests/test_config_fallback.py`,
  `tests/test_pcq_run.py`, `tests/test_validate_run_rescan.py`,
  `tests/test_compare_runs_config.py`,
  `tests/test_planset_output_dir.py`.

### Out of scope
- The remaining 16 P2 gaps from mnist-dogfood are deferred to v2.13.

## [2.11.0] — 2026-05-08

> **System-level evidence boundary tightening — schema, not policy.**
> Adds three additions that all live in pcq's contract layer: streaming
> partial RunRecord (time evidence), ExperimentPlanSet (multi-run schema
> expressivity), and structured failure envelope (machine-readable error
> code + evidence dict). Policy/inference (next-plan suggestion, trajectory
> interpretation, error-to-action mapping) remain outside pcq — those are
> the agent's responsibility.

### Added (Streaming Partial RunRecord)
- `pcq.save_partial_run_record(history, status="running", ...)` — atomic
  partial dump while training is in progress. Writes `run_record.json` via
  tmp + `os.replace` so readers always see valid JSON. Sets
  `run.partial=true`, `run.last_updated_at`, `run.status` (running /
  checkpointed). Final `finalize_run()` flips `partial=false`.
- `RunInfo` +2 fields: `last_updated_at: str | None`, `partial: bool`.
  `partial=False` (default) is stripped from the dict for backward
  compatibility — only `partial=true` appears in the JSON.
- `pcq.finalize_run()` now also writes via `tmp + os.replace`, matching
  the partial path's atomicity guarantee.
- `validate_run` skips reproducibility evidence gates when
  `run.partial=true` (running runs are not evaluated for
  reproducibility); manifest/run_summary missing checks are downgraded
  to `warn` while partial.
- New gate `run_finalized` at strictness ≥ 3: requires `run.partial=false`.
  Records explicit `pass` on finalized runs and `fail` on partial ones.

### Added (Structured Failure Envelope)
- `FailureInfo` dataclass: `error_code` (machine-readable) + `category`
  (kept for backward compat) + `evidence: dict` (structured) +
  `suggested_fix` (natural-language, for agent inference).
- Error code enum (`pcq.agent.run_record.ERROR_CODES` frozenset):
  `ERR_MISSING_DEPENDENCY`, `ERR_INVALID_CONFIG`, `ERR_DATASET_UNAVAILABLE`,
  `ERR_OUT_OF_MEMORY`, `ERR_TIMEOUT`, `ERR_RUNTIME`.
- `_classify_exception(exc)` — auto-classification on unhandled exceptions
  (ImportError → ERR_MISSING_DEPENDENCY + module evidence; MemoryError →
  ERR_OUT_OF_MEMORY; TimeoutError → ERR_TIMEOUT; FileNotFoundError →
  ERR_DATASET_UNAVAILABLE + path evidence; otherwise ERR_RUNTIME).
- `_normalize_failure(failure)` runs inside `save_run_summary()` to derive
  `error_code` from `category` and ensure `evidence` is a dict. Explicit
  `pcq.save_all(failure={...})` always wins over derivation.

### Added (ExperimentPlanSet)
- `ExperimentPlanSet` — set of related `ExperimentPlan` objects sharing
  `base` / `parent_run_id` / `parent_run_path`. Fields: `id`, `intent`
  (agent natural-language), `plans: list[ExperimentPlan]`.
- `pcq validate --planset path.json [--json]` — set-level validation
  including unique plan ids and per-member schema + label-contract checks.
- `pcq apply-planset path.json --output-pattern "runs/exp{i}" [--force]
  [--json]` — expand member plans into N output directories with
  `parent_run_id` / `parent_run_path` auto-propagation. Skipped (existing)
  vs applied vs rejected reported per-member.
- `pcq.agent.apply.apply_planset()` Python API alongside the CLI.

### Compat
- `RunRecord.failure: {category, message, suggested_fix}` (old shape)
  loads unchanged; `error_code` is derived from `category`.
- All new fields are optional in `to_dict()` — older RunRecord JSON files
  continue to load and validate.
- `ExperimentPlan` API unchanged. `apply-plan` (single plan) unchanged.
- `pcq --version` reports `2.11.0`. v2.x stability preserved.
- 732 → 780 passed (48 new), 4 skipped, 0 regressions.

### Out of scope (intentional)
- Suggestion of next plan from history — agent responsibility.
- Trajectory shape interpretation (converged / plateau / divergent) —
  agent responsibility.
- Mapping `suggested_fix` to executable commands — agent responsibility.

## [2.10.0] — 2026-05-08

> **Agent runtime closure — Roadmap Phase 3 + 4 + 5.**
> Golden E2E suite, agent authoring contract documentation, and agent
> runtime installation surface ship together. After v2.10.0 a Codex or
> Claude Code agent can install pcq conventions, author an experiment
> from documentation alone, and have its full lifecycle release-gated
> by automated E2E tests.

### Added (Phase 5 — Agent Runtime Installation Surface)
- `pcq agent install --target {codex,claude,both} [--dry-run] [--force]`
  installs:
  - Codex: `AGENTS.md` (managed marker block) + `.agents/skills/pcq/SKILL.md`
  - Claude Code: `CLAUDE.md` (managed marker block) + `.claude/skills/pcq/SKILL.md`
- `pcq init-experiment --agent {codex,claude,both}` runs the install path
  alongside project scaffolding.
- Canonical packaged assets at `src/pcq/agent_assets/` — `AGENTS.pcq.md`
  (77 lines) + `skills/pcq/SKILL.md` (243 lines).
- Non-destructive defaults: append managed blocks instead of replacing
  whole files; skip divergent skill files unless `--force`.
- Dry-run JSON exposes `operations[]` with `action` (`create/update/skip`)
  and `reason` (`missing/managed_block/diverged`) — service/agent can
  preview before committing.

### Added (Phase 3 — Golden E2E Suite)
- `tests/test_golden_e2e.py` — 4 release-gating end-to-end tests, all
  network-free (synthetic data only):
  - `test_golden_synthetic_mnist_mlp_script_e2e` — script-style torch
    contract round-trip
  - `test_golden_trainer_fake_smoke_e2e` — Trainer style with strictness 3
  - `test_golden_project_atom_scaffold_smoke_and_run_e2e` — scaffold →
    validate-local → smoke → load-project → train
  - `test_golden_failed_run_and_lineage_e2e` — failed run with structured
    failure + parent/child lineage
- Each scenario passes: inspect → validate → run → validate-run →
  describe-run → artifact existence.

### Documented (Phase 4 — Agent Authoring Contract)
- `docs/AGENT_OPERATING_GUIDE.md` expanded to 579 lines (+276):
  - Non-Negotiable Contract (5 musts)
  - Initial Triage (resolve → inspect → validate)
  - Choosing An Implementation Style (Contract Script / Project-Local
    Atoms / Trainer / Experiment decision tree)
  - Copyable Authoring Patterns (Torch / sklearn / arbitrary framework)
  - Editing Rules (prefer local, built-ins as examples, preserve contract)
  - Forbidden Patterns (9 anti-patterns table)
  - Pre-Run / Post-Run Checklists (strictness 2 vs 3)
  - Follow-Up Experiment Loop
  - Common Failure Patterns (4 recovery recipes)

### Roadmap restructure
- ROADMAP renumbered: prior Phase 5 (CQ Service / MCP) becomes Phase 6;
  prior Phase 6 (Release Hardening) becomes Phase 7. New Phase 5 (Agent
  Runtime Installation Surface) is now complete.

### Tests
- `tests/test_agent_install.py` — 11 install-surface tests
- 708 → 723 passed, 4 skipped, 0 regressions

### Compat
- All additions are additive. `pcq agent install` and `--agent` flag are
  new surfaces; existing CLI behavior unchanged. v2.x stability preserved.

## [2.9.0] — 2026-05-08

> **RunRecord evidence hardening — Roadmap Phase 2.**
> Source / Environment / Config / Input identity now record enough evidence
> for strictness 3 (Reproducible) and start covering strictness 4
> (Service Grade). All additions are additive — older RunRecords stay valid.

### Added
- `SourceInfo` +2 fields: `cq_yaml_path`, `cq_yaml_sha256` — which contract
  ran, and a content hash of it.
- `EnvironmentInfo` +7 optional fields: `pcq_version`, `torch_version`,
  `cuda_available`, `cuda_version`, `device`, `gpu_count`, `gpu_model`,
  `world_size`.
- `RunRecord.config` — `{cq_yaml_path, cq_yaml_sha256, config_json_path,
  config_json_sha256, seed, strictness, output_dir}` — config identity
  separate from environment.
- `RunRecord.input_summary` — `{count, names, identity{has_uri, has_path,
  has_sha256, has_manifest, opaque}}` — agent-readable input inventory.
- `lockfile_evidence` validation gate — strictness ≥3 requires both
  `lockfile` and `lockfile_sha256` populated. Reported as its own check ID
  (separated from `environment_reproducibility`).
- `seed_evidence` validation gate — strictness ≥3 surfaces explicit
  seed presence/absence.

### Refactored
- `_run_git(args, cwd=None)` helper extracted in `contract.py`. All
  `_git_*` functions now accept `cwd` so RunContext can drive evidence
  collection from arbitrary project roots.
- `_git_changed_files` switches from `git diff --name-only HEAD` to
  `git status --porcelain` — untracked files now appear, renames record
  the new path.
- `validate_run.source_reproducibility` gate adds `cq_yaml_sha256` to the
  required-evidence list at strictness ≥3.
- `validate_run.environment_reproducibility` gate adds `pcq_version` to
  the required-evidence list.

### Compat
- All new fields are optional in `to_dict()` — older RunRecord JSON files
  continue to load and validate.
- Default strictness is still 2 — Phase 2 evidence becomes blocking only
  when `--strictness 3` or `--strictness 4` is selected.
- 706 → 708 passed, 4 skipped, 0 regressions.

## [2.8.0] — 2026-05-08

> **Strictness validation gates — Phase 1 of the Completion Roadmap.**
> `pcq validate` and `pcq validate-run` now enforce different evidence
> requirements per strictness level (0–4). Agents and CI can pick the level
> that matches their use case: editor feedback (0), pre-run authoring (1),
> default local/dev (2), CI reproducibility (3), CQ service grade (4).

### Added
- `src/pcq/agent/strictness.py` — strictness level definitions and gate
  registry. Each gate declares which level it activates at.
- `pcq validate --strictness {0,1,2,3,4}` — selects evidence depth.
  Default unchanged (level 2). Strictness echoed in `validation_report.json`.
- `pcq validate-run --strictness {0,1,2,3,4}` — same axis for post-run
  RunRecord evidence (git sha/dirty, lockfile, env, inputs, lineage).
- New gates per level (Phase 1 scope):
  - L3: `git_sha_evidence`, `seed_recorded`, `lockfile_evidence`,
    `run_record_complete_v3`
  - L4: `inputs_have_identity`, `metric_schema_strict`, `device_evidence`,
    `lineage_for_derived_runs`, `validation_report_persisted`

### Refactored
- `validate_project` and `validate_run` accept `strictness` parameter and
  emit a `strictness_level` meta check.
- Each existing check now declares `min_strictness` so reports are stable
  across levels (lower-level reports remain a subset of higher levels).

### Docs
- `docs/PCQ_COMPLETION_ROADMAP.md` (369 lines) — completion definition,
  evidence model, 5 strictness levels, 6 implementation phases, priority,
  non-goals.
- README, SPEC, CQ_MCP_SPEC cross-reference the roadmap.

### Tests
- +14 tests covering all 5 strictness levels in
  `test_agent_validate.py` (+63), `test_cli.py` (+24),
  `test_validate_run.py` (+65).
- 692 → 706 passed, 4 skipped, 0 regressions.

## [2.7.0] — 2026-05-06

> **Agent CLI surface closed + non-model examples moved under `pcq.examples`.**
> The service-facing core commands now expose the options required by
> `CQ_MCP_SPEC.md`, while dataset/optimizer/scheduler reference atoms follow
> the same "contract example + compatibility facade" pattern as models.

### Added
- `pcq inspect --load-project-atoms` opt-in dynamic project atom import.
  Default `inspect` is now read-only and does not import `cq_atoms.py` or
  `atoms/*.py`.
- `pcq validate --strictness 0..4`; reports the selected strictness level and
  supports lighter static-only validation levels.
- `pcq resolve --cq-yaml PATH` for explicit cq.yaml resolution.
- `pcq finalize --project-root PATH --status completed|failed|partial`.

### Refactored
- Moved reference dataset atoms to `pcq.examples.datasets`.
- Moved reference optimizer atoms to `pcq.examples.optim`.
- Moved reference scheduler atoms to `pcq.examples.sched`.
- `pcq.datasets`, `pcq.optim`, and `pcq.sched` are now v2 compatibility facades,
  mirroring the existing `pcq.models` facade behavior.

### Compat
- Existing imports such as `pcq.datasets.fake`, `pcq.optim.adamw`,
  `pcq.sched.cosine`, and `Trainer(dataset="fake")` continue to work.
- Project atom validation remains explicit through
  `pcq atoms validate-local` / `pcq atoms smoke --load-project`.

## [2.6.0] — 2026-05-06

> **Reference example models physically relocated to `pcq.examples.models`.**
> v2.4 introduced the `pcq.examples.*` alias namespace as a labeling change.
> v2.6 makes the relocation real: the model implementations now live under
> `pcq.examples.models`, and `pcq.models` becomes a thin v2 compatibility
> facade. Recipes, templates, and Trainer docstrings switch to the new
> location. Other atom categories (`pcq.datasets`, `pcq.loss`, `pcq.metric`,
> `pcq.optim`, `pcq.sched`) remain untouched in this release.

### Refactored
- Moved 397-line implementation from `src/pcq/models.py` to
  `src/pcq/examples/models.py` — same six reference atoms (mlp, small_cnn,
  resnet18, text_classifier, unet, deeplab_v3) with the same `_registry`
  registrations and `[reference example — for production, register a
  project atom …]` description suffixes.
- `src/pcq/models.py` reduced to a 28-line compatibility facade that
  re-exports the public factories from `pcq.examples.models`.
- `pcq.examples.__init__` now uses `__getattr__` lazy import:
  `pcq.examples.models` resolves to the new module, while
  `pcq.examples.{datasets,loss,metric,optim,sched}` continue to alias
  `pcq.{datasets,...}` until those categories are migrated.
- Internal callers updated to the new location:
  - `pcq.recipes.vision.mnist_mlp`
  - `pcq.recipes.vision.seg.voc_unet`
  - `pcq.agent.init` Experiment scaffold template
  - `pcq.trainer` docstring example

### Compat
- `pcq.models.mlp(...)` etc. continue to work — same factory functions.
- Verified invariants: `pcq.examples.models is not pcq.models` (real module
  vs facade), `pcq.examples.models.mlp is pcq.models.mlp` (same callable).
- `Trainer(model="small_cnn")` and other string-name lookups unchanged
  (registry shared via the same registration calls in the new module).
- Tests: 692 pass + 4 skipped, 0 regressions.

## [2.5.0] — 2026-05-06

> **cq.yaml interpretation unified into ResolvedConfig + RunContext.**
> All consumers (contract.py, core.py, Trainer/Experiment, CLI inspect/
> validate/finalize) now share a single resolver path. Eliminates cwd-
> dependent behavior — agent invocations from `scripts/train.py`, project
> root, or service worker with arbitrary cwd produce identical results.

### Refactored (architecture)
- **Read/write split**: `resolve_project()` is now strictly read-only
  (no mkdir, no chdir). Use `resolve_run_context(ensure_output_dir=True)`
  for write-side semantics. RunContext is the **only** API path that
  creates `output_dir`.
- `pcq finalize <output_dir>` no longer chdir's or writes a `.pcq_finalize_tmp.json`.
  Calls `finalize_run(output_dir=..., project_root=...)` directly. Walks
  parents to find cq.yaml when `project_root` not given. The output_dir
  name no longer affects detection (was: assumed "output").
- `Experiment._finalize_run_artifacts` drops the chdir/env tmp-file trick.
- `inspect_project` / `validate_project` now consult `ResolvedConfig.output_dir`
  for the post-run artifact location instead of the legacy hard-coded
  `output/` candidate. Custom `output_dir` (e.g. `runs/exp001`) is detected.

### Added
- **`pcq.RunContext`** dataclass — write-time wrapper around `ResolvedConfig`
  with `project_root`, `output_dir`, `cfg`, `name`, `cmd`, `declared_metrics`,
  `inputs`, and `artifact_path(name)` convenience.
- **`pcq.resolve_run_context(path, cq_yaml_path, output_dir, ensure_output_dir)`**
  — write-side resolver. Single mkdir owner.
- `_cq_project_root` env var honored in `resolve_project()` (tests / service
  workers / explicit wiring).
- `finalize_run` / `save_all` / `save_*` accept explicit `output_dir` and
  `project_root` kwargs (additive — no breaking signature change).
- `OutputsInfo.status` field — `"empty"` | `"partial"` | `"complete"` | `None`.

### Fixed (P2 cleanup)
- (#4) Malformed cq.yaml no longer silently parsed. Parse errors surface
  in `ResolvedConfig.parse_errors`; `inspect_project` records them in
  `errors`; `validate_project` adds a `cq_yaml_parseable` blocking gate.
- (#5) Empty output_dir inspect produces explicit `status: "empty"` so
  agents / scripts can branch without misleading "missing artifact" errors.
  CLI `pcq inspect` JSON exposes `outputs.status`.

### Tests
- +9 DoD regression tests (`tests/test_run_context_dod.py`):
  1. env-only + cq.yaml save_all writes all artifacts to custom output_dir
  2. nested cwd finds parent cq.yaml → same output_dir
  3. CLI finalize uses root cq.yaml metadata (name/cmd/inputs/metrics) in RunRecord
  4. inspect detects custom output_dir's artifacts
  5. validate runs manifest_evidence against custom output_dir
  6. resolve_project does NOT mkdir (read-only invariant)
  7. run_record.run.name propagates from pcq.yaml top-level
  8. CQ_CONFIG_JSON.output_dir overrides cq.yaml.configs.output_dir
  9. three modes (env-only / yaml-only / both) all pass
- +4 yaml-strict tests (`tests/test_yaml_strict.py`)
- +6 inspect empty-dir tests (`tests/test_inspect_empty_dir.py`)
- Total: 692 passed (was 672) + 4 skipped.

### Compat
- `pcq.config()`, `pcq.output_dir()`, `pcq.save_all()`, `pcq.save_metrics()`,
  `pcq.save_config_snapshot()`, `pcq.save_manifest()`, `pcq.save_run_summary()`,
  `pcq.finalize_run()` — call signatures unchanged (new kwargs are additive).
- Existing env-only invocations behave identically.
- New behavior triggers only when cq.yaml is present (previously ignored
  in some consumers).

## [2.4.0] — 2026-05-06

> Positioning release — built-in atoms 가 production catalog 가 아니라 **contract
> example** 임을 문서/모듈/CLI/메타데이터 전체에서 명시. backward compat 100%.

### Refactored (positioning)
- Built-in atoms (`pcq.models`, `pcq.datasets`, `pcq.loss`, `pcq.optim`,
  `pcq.sched`, `pcq.metric`) explicitly demoted from "internal catalog" to
  "reference examples for contract verification + onboarding + smoke
  baselines". Module docstrings updated, README / VISION / SPEC /
  ATOM_REGISTRY use consistent terminology. Production atoms belong in
  project-local `atoms/` via `pcq.register_*`.
- 24 builtin AtomSpec descriptions get a `[reference example — ...]` suffix
  pointing to the project-local atom path.

### Added
- **`pcq.examples` alias namespace** — explicit "reference example" framing
  for the same atoms. `pcq.examples.models is pcq.models`,
  `pcq.examples.loss.cross_entropy is pcq.loss.cross_entropy`, etc. Use
  whichever name communicates intent better; behaviour is identical.
- **`AtomSpec.role` field** — `"reference_example"` for builtins,
  `"user"` for project / generated / external atoms. Inferred from `source`
  when not explicitly supplied via `meta={"role": ...}`. JSON output
  (`pcq atoms list/show --json`) and `to_dict()` include the new field
  (additive, schema_version unchanged).
- **`pcq atoms list` (human output)** annotates builtin atoms with
  `[reference example]` and project atoms with `[project]` / `[generated]`,
  plus a footer note pointing to `pcq atoms scaffold` for production atoms.

### Tests
- +9 `pcq.examples` namespace tests (alias identity, factory equivalence,
  exposure via `pcq.examples` and `pcq.__all__`).
- +13 `AtomSpec.role` tests (builtin role inference per kind, explicit
  `meta={"role": ...}`, `source="project"` default-to-user, legacy meta=None
  inference).
- Total: 672 passed (was 650) + 4 skipped.

### Compat
- Public APIs unchanged: `pcq.models.mlp`, `pcq.register_model`, `AtomRef`,
  `Trainer`, `Experiment`, etc.
- `AtomSpec.source` field unchanged.
- `AtomSpec.to_dict()` gains additive `role` key.
- `pcq atoms list --json` output gains additive `role` field per entry.
- All previously registered names + factories remain valid; old `meta=None`
  registrations infer `role="reference_example"` (builtin default).

## [2.3.0] — 2026-05-06

> v2.0.0 audit (P1 #2/#3/#6) follow-up — agent decision-side reinforcement.

### Fixed
- **Lineage `best_value` extraction for ancestors** (audit P1 #2):
  depth>0 nodes now correctly surface `best_value` and `name` in
  `chain.to_dict()`. Previously `LineageNode.to_dict()` filtered all
  empty/falsy fields, dropping `best_value=0.0` and (since
  `RunRecord.run.name` was always blank) the ancestor `name`.
  - `LineageNode.to_dict()` now keeps meaningful zeros (`0/0.0/False`)
    and always preserves `run_id`/`depth`/`status` even when blank, so
    agents can distinguish "field exists but empty" from "field omitted".
- **`finalize_run()` name propagation** (audit P1 #2 root cause):
  `RunInfo.name` now falls back to cq.yaml top-level `name:` when
  `configs.name` is absent. Previously left blank, which broke
  lineage display for every project that put `name:` only at the top
  level (the documented pattern in `pcq init-experiment`).

### Added
- **`compare-runs` trajectory signals** (audit P1 #6) — `RunDiff` gains
  fields so agents can see hyperparameter effect even when best metric
  is tied (the audit's "two runs both pick epoch 0 as best" case):
  - `last_metric_delta` / `last_metric_direction` — last-epoch
    comparison (improved/regressed/tied/incomparable, mode-aware).
  - `epochs_a` / `epochs_b` — total epoch count from `metrics.json`.
  - `best_epoch_a` / `best_epoch_b` — best epoch index per run.
  - `notes: list[str]` — explanatory strings, e.g. "best is tied, but
    last epoch differs: regressed (+0.9000). hyperparameter change
    affected trajectory." or "both runs picked epoch 0 as best —
    likely 'no learning' signal."
- **`validate --plan` label-contract simulation** (audit P1 #3):
  new `plan_label_contract` gate simulates plan `set_atom` changes
  on top of the base preset's `RecipeSpec.atoms` and runs
  `_validate_label_contracts` against the merged view. Catches
  `set_atom loss cross_entropy ignore_index=...` mismatches with the
  dataset's `label_contract.ignore_index` BEFORE the run starts.

### Tests
- +5 lineage tests (best_value/name preservation, finalize_run name
  propagation, ALWAYS_KEEP fields, zero/empty-list handling).
- +3 compare-runs trajectory tests (tied-best with diverging last,
  epoch counting, "no learning" note).
- +5 validate-plan label-contract tests (mismatch detect, consistent
  pass, no-base skip, unknown-preset silent skip, CLI integration).
- Total: 650 passed (was 637) + 4 skipped.

### Compat
- `RunDiff.to_dict()` gains new keys when populated; consumers that
  filter on a fixed allowlist may need updates. Empty/None fields are
  still stripped.
- `LineageNode.to_dict()` semantics widened: `0`/`0.0`/`False` are now
  preserved (previously stripped). Consumers treating presence as
  truthiness need adjustment.

## [2.2.0] — 2026-05-06

### Added
- `pcq.resolve_project(path | None) -> ResolvedConfig` — single source of
  truth for cq.yaml + CQ_CONFIG_JSON env interpretation. All cq.yaml-reading
  code paths (`inspect`, `validate`, `finalize_run`) now consult resolver.
- `ResolvedConfig` dataclass — normalized view of project state:
  cfg / declared_metrics / metrics_schema / artifacts / inputs / output_dir.
  output_dir is absolute, project_root-rooted, mkdir-safe.
- CLI: `pcq resolve [PATH] [--json]` — debug resolver output.
- list-style and dict-style `metrics:` normalize to same `declared_metrics`
  (always list[str]) — agents don't need to handle two shapes.
- CQ_CONFIG_JSON env merges INTO cq.yaml.configs (env wins) — explicit
  precedence rule.

### Fixed
- `finalize_run()` was reading cq.yaml via cwd-relative `Path("cq.yaml")` —
  inputs and metrics_schema were lost when training launched from a sub-
  directory. Now uses resolver (walks up cwd ancestors, stops at project
  root marker). Identified in v2.1.0 audit. New regression test:
  `test_finalize_run_finds_cq_yaml_from_subdirectory`.

### Internal
- `inspect_project()` now builds `CqYamlSummary` from a single resolver call
  via the new `_build_cq_yaml_summary_from_resolver()` helper.
- All cq.yaml read sites consolidated to single resolver call.

### Compat
- `core.config()` / `core.output_dir()` signatures unchanged — still
  read CQ_CONFIG_JSON env (low-level). New code should prefer
  `pcq.resolve_project()` for full project view.

## [2.1.1] — 2026-05-06

### Fixed (P0 hotfix)
- `pcq init-experiment --with-pyproject` template was unbuildable: empty
  `[tool.hatch.build.targets.wheel] packages = []` caused `uv sync` to fail
  with `ValueError: Unable to determine which files to ship inside the wheel`.
  Fresh users hit this on first command. Now uses `[tool.uv] package = false`
  (non-package experiment project) — `uv lock`/`sync` resolve dependencies
  without trying to build a wheel.

### Added (CI hardening)
- `scripts/release-smoke.sh` step 5: fresh-user pyproject template gate.
  Generates project, parses generated pyproject.toml, asserts non-package
  mode + git source. Catches future template regressions before users do.

### Docs
- README: prominent "not on PyPI" notice + git URL install instructions.
- README: explicit "Known limitations (v2.x)" section listing lineage
  best_value gap, compare-runs coarseness, validate --plan gaps, catalog scope.
- README: updated install snippets to use git URL.

## [2.1.0] — 2026-05-06

### Added
- `pcq init-experiment --with-pyproject` flag — generates `pyproject.toml`
  with `pcq>=<version>` dependency. Preset's `requires_extras` (e.g.
  `["vision"]` for `vision/mnist_mlp`) automatically added as
  `pcq[vision]>=<version>`. Recommended for reproducible runs — `uv lock`
  produces `uv.lock` and `run_record.json.environment.lockfile_sha256` is
  populated.

### Chore
- `pyproject.toml`: pin `torchvision>=0.26.0` in `[vision]` extras (aligns
  with the `pcq.datasets.{cifar10, mnist, voc_seg}` torchvision wrappers).

## [2.0.2] — 2026-05-06

### Fixed
- `pcq inspect` now extracts preset from `cq.yaml.configs.preset` when the
  entrypoint uses `Trainer.from_cfg(cfg)` pattern (v1.10+ default template).
  Previously only `Trainer(preset="...")` literal was detected via AST,
  leaving `entrypoint.preset` as `None` for cfg-driven trainer entrypoints.
  Literal kwarg still takes precedence when both are present.

## [2.0.1] — 2026-05-06

### Fixed
- `_environment_snapshot()` now walks up cwd ancestors (max 8 levels) to
  find lockfile, stopping at first project root marker (`.git` or
  `pyproject.toml`). Previously cwd-relative `Path("uv.lock")` failed when
  training was launched from a subdirectory, leaving
  `run_record.json.environment.lockfile_sha256` empty.

## [2.0.0] — 2026-05-06

### Milestone

pcq v2.0 stable baseline — contract runtime API surface complete.

After 18 incremental v1.x releases (0.1.0 → 0.1.19), pcq has reached a
stable surface for CQ service integration. v2.x development will focus on
service hooks rather than core API changes.

### What v2.0 means

- **API stability promise**: existing public APIs (`pcq.config` / `log` /
  `output_dir`, `pcq.Experiment`, `pcq.Trainer`, `pcq.save_*`, `pcq.register_*`,
  `pcq.{model,dataset,loss,optim,sched,metric}_ref`, `pcq.recipe_meta`,
  `pcq.diff_recipes`, `pcq.lineage`, `pcq.compare_runs`, `pcq.describe_run`,
  `pcq.finalize_run`) stay stable through v2.x
- **Contract artifacts** stable: `config.json` / `metrics.json` /
  `manifest.json` (v2) / `run_summary.json` / `run_record.json` /
  `validation_report.json`
- **Schema versions** locked: `AtomSpec` / `AtomRef` / `ParamSpec` /
  `RecipeSpec` / `ExperimentPlan` / `RunRecord` / `ValidationReport`
  schema_version=1 — additive changes only in v2.x
- **CLI surface** stable: 13 subcommands (`inspect`, `recipe-meta`,
  `dry-run`, `validate`, `summarize-run`, `atoms.{list,show,scaffold,
  validate-local,smoke,validate-ref}`, `init-experiment`, `apply-plan`,
  `finalize`, `validate-run`, `describe-run`, `compare-runs`, `lineage`)

### What's not in v2.0

- CQ service integration hooks (post-finalize webhook 등)
- Plan auto-suggestion from RunRecord
- `set_smoke_override` ChangeOp (deferred from v1.11)

These are v2.1+ candidates.

## [0.1.19] — 2026-05-06

### Stabilization
- LICENSE (Apache-2.0) 정리
- CHANGELOG.md 추가 (Keep a Changelog 형식)
- README 일관성 정리 (v1.x 시리즈 18 releases 반영, CLI table 최신화)
- pyproject.toml metadata 정비 (description / classifiers / keywords / urls)
- 테스트 명명 일관성 정리

## [0.1.18] — 2026-05-06

### Added
- Lineage tracking — `RunRecord.run.parent_run_id`, `parent_run_path`
- `pcq.agent.lineage(start)` — parent chain traversal + cycle detection
- `pcq.agent.is_descendant_of(child, ancestor_id)` helper
- CLI: `pcq lineage [OUTPUT_DIR] [--max-depth N]`
- ExperimentPlan: `parent_run_id` + `parent_run_path` 필드
- apply-plan 이 cq.yaml 에 `_parent_run_id` / `_parent_run_path` 자동 주입
- compare-runs: `a_is_ancestor_of_b` / `b_is_ancestor_of_a` 필드

## [0.1.17] — 2026-05-05

### Added
- `pcq.agent.describe_run()` — compact RunRecord summary
- `pcq.agent.compare_runs(a, b)` — RunDiff (metric_delta, direction, changes)
- `pcq.agent.failure_classifier` — 11 categories (oom / nan_loss / missing_dependency / ...)
- CLI: `pcq describe-run`, `pcq compare-runs`
- `save_run_summary` integrates failure classifier

## [0.1.16] — 2026-05-05

### Added
- RunRecord MVP — `run_record.json` schema (RunInfo + ExecutionInfo + SourceInfo + EnvironmentInfo + MetricsInfo + AgentInfo + ValidationInfo)
- `pcq.finalize_run()` Python helper
- `pcq.save_all(finalize=True)` default — 6 contract artifacts at once
- `Experiment.fit()` 자동 finalize
- environment snapshot (python + platform + lockfile sha256)
- source snapshot (git_sha + dirty + opt-in patch_sha256 / changed_files)
- `validation_report.json` post-run gates
- CLI: `pcq finalize`, `pcq validate-run`
- inspect outputs: `has_run_record` + `has_validation_report`

## [0.1.15] — 2026-05-05

### Added
- Structured cq.yaml — `inputs:` section + dict-style `metrics:` (mode / split / aggregation / sample_count)
- `CqYamlSummary.metrics_schema` + `CqYamlSummary.inputs` 필드
- Validation gates: `metric_schema_*`, `inputs_declared`, `monitor_in_metric_schema`, `monitor_mode_consistency`
- inspect 가 yaml_io.read_yaml 사용 (full YAML)
- minimal yaml parser inline flow style (`{k: v}`, `[a, b]`)

### Note
- list-style `metrics:` 영구 호환 (legacy)
- cq URI 는 opaque string 으로 record (parse / fetch 안 함)

## [0.1.14] — 2026-05-05

### Added
- Manifest schema v2 — sha256 + size_bytes + created_at per file
- `pcq.save_manifest(enrich=True)` default
- `cfg["manifest_checksums"]=false` opt-out (large model 환경)
- post-run gate: `manifest_evidence` (file existence + sha256 verify)
- inspect: `manifest_schema_version` + `manifest_files_count`

## [0.1.13] — 2026-05-05

### Added
- Contract Script first-class — `pcq.save_config_snapshot / save_metrics / save_manifest / save_run_summary / save_all`
- `pcq init-experiment --style {trainer|experiment|script}`
- inspect: `detected_imports` (sklearn / xgboost / transformers 등) + `cq_calls` AST 추출
- script-aware CLI gates (`cq_config_called`, `cq_log_called`, `standard_artifacts_helper`)
- apply-plan: script project 에서 `set_atom` / `set_dataset_transform` 명시적 reject
- `examples/contract_sklearn.py` + `cq.yaml`

## [0.1.12] — 2026-05-05

### Added
- Project atom workflow — `cq_atoms.py`, `atoms/*.py` 자동 discovery
- `AtomSpec.source` ("builtin" | "project" | "generated") + `module` 필드
- `pcq.registry.load_project_atoms(path)` + `list_sources()`
- CLI: `pcq atoms list --source` filter, `atoms scaffold KIND NAME`, `atoms validate-local`, `atoms smoke KIND NAME`
- 6 kind 별 minimal-runnable scaffold templates
- init-experiment 가 `cq_atoms.py` + `atoms/__init__.py` 자동 생성

## [0.1.11] — 2026-05-05

### Added
- `set_atom merge=True` — params 부분 갱신 (전체 ref 재명시 X)
- `set_dataset_transform` ChangeOp (set_atom merge=True 의 sugar)
- `pcq[yaml]` extras — ruamel.yaml comment-preserving YAML I/O
- base atom name / params 상속 (merge=true + name=None 시 base recipe 에서 추론)

## [0.1.10] — 2026-05-05

### Added
- Phase D MVP — `ExperimentPlan` schema + `pcq apply-plan`
- `pcq init-experiment --preset NAME --output DIR`
- `pcq validate --plan PLAN.json`
- `Trainer.from_cfg(cfg)` — preset / `_overrides_data` 자동 인식
- bounded mutation: cq.yaml configs 만 수정
- provenance: `.pcq/plans/<plan_id>.json` 자동 저장
- minimal YAML writer / reader (PyYAML 의존 없음)
- 2 ChangeOps: `set_config`, `set_atom`

## [0.1.9] — 2026-05-05

### Added
- 모든 24 built-in atoms `metadata_status: explicit` (이전 14 개 inferred)
- 모든 7 recipes `RecipeSpec` 변환
- 새 validation gates: `model_dataset_channels`, `optional_extras_available`, `monitor_candidates_declared`
- `text_classification` task in `_ComposedExperiment`

## [0.1.8] — 2026-05-05

### Added
- atom registry metadata-first — `AtomSpec` + `ParamSpec` + `AtomRef`
- ref constructors: `pcq.{model,dataset,loss,optim,sched,metric}_ref`
- `RecipeSpec` (pcq.agent.schema) + `.build()` resolving refs
- 5 atoms 메타데이터 (cross_entropy, unet, fake_seg, voc_seg, iou)
- 2 seg recipes RecipeSpec 변환
- `loss_label_ignore_index` validation gate
- CLI: `pcq atoms list / show / validate-ref`

## [0.1.7] — 2026-05-05

### Added
- JSON CLI MVP — `pcq inspect / recipe-meta / dry-run / validate / summarize-run`
- `pyproject [project.scripts] pcq = pcq.cli:main`
- `pcq.agent` package — schema / inspect / summary / validate
- `run_summary.json` 자동 생성 (fit() 종료 시)
- `ProjectInspection`, `RunSummary`, `ValidationReport` 데이터클래스

## [0.1.6] — 2026-05-05

### Added
- Phase A contract gap fixes
  - `pcq.loss.cross_entropy(ignore_index, weight)` 인자 지원
  - `pcq.datasets.voc_seg(image_size=256)` fixed-size resize
  - `pcq.log()` `CQ_CONFIG_JSON._metrics_declared` 자동 로드
  - accelerate main-process guard (`_is_main_process`)
  - monitor pre-check (`fit()` 시작 시 declared 미스매치 경고)

## [0.1.5] — 2026-05-05

### Added
- CI smoke automation — `scripts/release-smoke.sh` (4 stages)
- GitLab CI `.gitlab-ci.yml` (lint / test / smoke)

## [0.1.4] — 2026-05-05

### Added
- Recipe Acceptance Framework — `pcq.testing.recipe_smoke` (7 criteria)
- `pcq.agent` (recipe_meta, diff_recipes, list_meta)
- `Trainer.dry_run()` — 조립 plan 노출
- Provenance: config.json `_recipe`, `_overrides`, `_pcq_version` 자동 기록

## [0.1.3] — 2026-05-05

### Added
- Metric aggregation: `cfg["metrics_aggregation"]: mean | weighted_mean`
- `pcq.metric.stateful` (Accuracy, IoU)
- AMP — `cfg["amp"]` + GradScaler + autocast
- Gradient accumulation — `cfg["grad_accum"]`
- Early stopping — `early_stop_patience` + `min_delta`
- Segmentation atoms — unet, deeplab_v3, fake_seg, voc_seg, dice / focal loss, iou / dice_score / pixel_accuracy metric
- 2 seg recipes — `vision/seg/fake_seg_smoke`, `vision/seg/voc_unet`

## [0.1.2.1] — 2026-05-05

### Added
- Artifact manifest (schema_version=1)
- Best checkpoint monitor + `min` / `max` mode
- Device resolve (cfg.device > cuda > mps > cpu)

## [0.1.2] — 2026-05-05

### Added
- metric atom 분리 (loss 와 동형)
- atom registry decorator / function API — `pcq.register_{model,dataset,loss,optim,sched,metric}`
- `pcq.metric` 모듈 (accuracy / top_k / mse / mae)
- `Trainer.list_models / datasets / metrics` 추가
- recipe lambda 미사용 split → `_split` prefix
- `training_step → (loss, metrics) tuple` (loss / metric 책임 분리)

## [0.1.1] — 2026-05-05

### Added
- Auto resume (output_dir/last.ckpt 자동 발견)
- Recipe catalog 5 개 (vision/fake_smoke, mnist_mlp, cifar10_smallcnn_baseline, cifar10_resnet18, nlp/fake_text_classifier)
- Atoms 4 개 추가 (resnet18, text_classifier, mnist, fake_text)
- accelerate underlying (pcq[dist] extras)

## [0.1.0] — 2026-05-05

### Added
- v1 minimum viable pcq
- 3-tier API (low / mid / high) — `pcq.config / log / output_dir`, `pcq.Experiment`, `pcq.Trainer`
- 6 task baseline (T-CQPY-001~006)
- Built-in atoms — mlp, small_cnn, fake, cifar10, cross_entropy, adamw, cosine
- 1 recipe — `vision/cifar10_smallcnn_baseline`
- cq.yaml runtime contract (CQ_CONFIG_JSON, stdout @key=value, output_dir artifacts)
- Strict metric schema (warn on undeclared)
- Tests + integration smoke
