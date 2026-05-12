# pcq.describe_run.record · without-attribution

Backward compatibility scenario: `run_record.json` has no `attribution` field at all.

Structural assertions:

- `attribution` field is absent from the output (not null — simply omitted, as the
  `RunDescription.to_dict()` drops `None` values for non-ALWAYS_KEEP_KEYS fields).
- All 4 flat attribution fields (`attribution_author_kind`, `attribution_committer_kind`,
  `attribution_operator`, `attribution_session_id`) are also absent from the output.
- All other required fields of `pcq.describe_run.record` are present and valid.

This case ensures agents can handle runs produced before attribution support was added.
