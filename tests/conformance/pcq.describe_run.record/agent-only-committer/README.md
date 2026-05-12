# pcq.describe_run.record · agent-only-committer

Mixed human/agent scenario: a human authored the change but an AI agent (claude-opus-4-7)
acted as the committer, with a human operator overseeing the process.

This corresponds to the CQ_ATTRIBUTION_* env var combination:
  CQ_ATTRIBUTION_AUTHOR_KIND=human
  CQ_ATTRIBUTION_AUTHOR_ID=alice-uuid-1234
  CQ_ATTRIBUTION_COMMITTER_KIND=agent
  CQ_ATTRIBUTION_COMMITTER_ID=claude-opus-4-7
  CQ_ATTRIBUTION_OPERATOR=alice-uuid-1234

Structural assertions:

- `attribution.author.kind="human"`.
- `attribution.committer.kind="agent"`, `attribution.committer.id="claude-opus-4-7"`.
- `attribution.operator="alice-uuid-1234"`.
- Flat fields: `attribution_author_kind="human"`, `attribution_committer_kind="agent"`,
  `attribution_operator="alice-uuid-1234"`.
