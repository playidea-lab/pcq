# pcq.describe_run.record · operator-only

Auto-infer scenario: `run_record.json` has an `attribution` object where operator is
set and author/committer are auto-inferred to the operator's id with `kind="human"`.

This mirrors the `build_attribution_object()` behavior when only
`CQ_ATTRIBUTION_OPERATOR` is set (step 2 auto-infer in `contract.py`).

Structural assertions:

- `attribution.author.kind="human"`, `attribution.author.id="alice-uuid-1234"`.
- `attribution.committer.kind="human"`, `attribution.committer.id="alice-uuid-1234"`.
- `attribution.operator="alice-uuid-1234"` (all three are the same uuid).
- `attribution.session_id` absent (was null, dropped from output).
- Flat fields: `attribution_author_kind="human"`, `attribution_committer_kind="human"`,
  `attribution_operator="alice-uuid-1234"`. `attribution_session_id` absent.
