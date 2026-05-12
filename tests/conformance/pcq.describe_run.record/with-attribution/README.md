# pcq.describe_run.record · with-attribution

Full attribution scenario: `run_record.json` contains an attribution object with
all three roles (operator, author, committer) and a session_id explicitly set.

Structural assertions:

- `attribution` nested object is present with `schema_version=1`.
- `attribution.author.kind="human"`, `attribution.author.id="alice-uuid-1234"`.
- `attribution.committer.kind="human"`, `attribution.committer.id="alice-uuid-1234"`.
- `attribution.operator="alice-uuid-1234"`.
- `attribution.session_id="sess-abc-9999"`.
- All 4 flat fields populated: `attribution_author_kind`, `attribution_committer_kind`,
  `attribution_operator`, `attribution_session_id`.
