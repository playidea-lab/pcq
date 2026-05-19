# TC Reconciliation Note — pcq 2.x ↔ TheCommons Vendoring Cycle

> Input note for the TC vendoring cycle. Concise. Normative where marked.
> Date: 2026-05-19. Cycle: T-PCQ2X-8.

---

## 1. Single Canonical Source

**`pcq spec/` is the SINGLE canonical source for pcq 2.x.**
`docs/pcq-2.x.md` (inside the TC repository) is a *vendored copy*.

On any divergence between the TC-side copy and `pcq spec/`, **`pcq spec/` wins**.
TC ingestion and TC schema code must be updated to match `pcq spec/`, not the
other way around.

TC should vendor the following pcq artifacts (read-only, no modifications):

- `spec/schemas/pcq.describe_run.record.schema.json` — generated JSON Schema
- `spec/SPEC.md § pcq 2.x Contract` — normative field definitions
- `spec/VERSIONING.md § pcq 2.x — Three-Axis Version Policy` — version axis policy

---

## 2. `content_hash` Rebase (NORMATIVE)

`content_hash` was **rebased**: it is now located at `integrity.content_hash`
(top-level `integrity` object), **NOT** at `attribution.content_hash`.

```json
{
  "integrity": {
    "content_hash": "sha256:<hex>",
    "hashed_fields": ["intent", "config", "metrics", ...]
  },
  "attribution": {
    "author": { "id": "...", "kind": "agent" },
    "committer": { "id": "...", "kind": "human" },
    "operator": "pilab",
    "session_id": "..."
  }
}
```

`attribution` (v4.4: author / committer / operator / session_id / signature —
the **WHO**) is **unchanged**. TC ingestion that reads `attribution.content_hash`
must be updated to read `integrity.content_hash` instead.

---

## 3. TC Proposal Field Mapping (NORMATIVE)

The TC Proposal (`docs/pcq-2.x.md` as of 2026-05-19) introduced three
attribution sub-fields. Their canonical mapping into pcq 2.x:

| TC Proposal field | pcq 2.x canonical | Note |
|---|---|---|
| `attribution.created_at` | Existing `run_record` timestamp (`last_updated_at`, `completed_at`) | Reuse existing; no new field |
| `attribution.contributor_id` | `attribution.operator` | Identical concept; TC reads `attribution.operator` |
| `attribution.pcq_version` | `contract_version` | Top-level field, NOT inside `attribution` |

TC ingestion must read pcq 2.x records using the pcq-canonical names above.
The TC Proposal names are not valid in pcq records.

---

## 4. Out-of-pcq-Scope Fields (NORMATIVE)

The following fields are **explicitly out of pcq scope**. They belong to the TC
Evidence wrapper (`pcq_record` is nested inside the TC envelope):

| Field | Belongs to |
|---|---|
| `tier` (`"real"` / `"synthetic"`) | TC Evidence wrapper |
| `synthetic_source` | TC Evidence wrapper |
| `evidence_id` | TC Evidence wrapper |
| `outreach_origin` | TC Evidence wrapper |

pcq records **real experiments only**. Synthetic evidence (LLM-distilled) is a
TC artifact managed entirely by TC's own schema outside of pcq.

The TC envelope structure (for reference; TC owns the implementation):

```json
{
  "evidence_id": "<string>",
  "tier": "real" | "synthetic",
  "outreach_origin": "<string>",
  "synthetic_source": { ... } | null,
  "pcq_record": {
    <pcq 2.x run_record verbatim>
  }
}
```

The single seam for producers (e.g. cq M4): *build a pcq run_record, then place
it in the `pcq_record` key of the TC envelope*. Sidecar fields on the pcq record
itself are forbidden.

---

## 5. pcq 2.x Additive — TC Vendoring Notes

pcq 2.x is **additive-only** relative to 1.x:

- Every 1.x `run_record.json` is a valid 2.x record (absent fields = null).
- No 1.x required field is removed or renamed.
- `contract_version`, `intent`, and `integrity` are new **optional** fields.
- 1.x readers that do not know these fields must silently ignore them.

TC should vendor the pcq `spec/schemas/*.schema.json` (generated) and the
`spec/SPEC.md § pcq 2.x Contract` section. Do not copy individual field
definitions by hand — use the generated schema as ground truth.

---

*See also: `spec/VERSIONING.md § pcq 2.x — Three-Axis Version Policy`,
`spec/SPEC.md § pcq 2.x Contract`, `spec/SPEC.md § R9`, `spec/SPEC.md § R10`,
`spec/SPEC.md § R12`.*
