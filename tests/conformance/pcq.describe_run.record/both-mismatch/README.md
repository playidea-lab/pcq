# pcq.describe_run.record · both-mismatch

Tri-party attribution scenario: author, committer, and operator are all distinct
identities. Alice authored the change, Bob committed it, and a shared team UUID
(organization) is the operator.

This corresponds to the CQ_ATTRIBUTION_* env var combination:
  CQ_ATTRIBUTION_AUTHOR_KIND=human
  CQ_ATTRIBUTION_AUTHOR_ID=alice-uuid-1234
  CQ_ATTRIBUTION_COMMITTER_KIND=human
  CQ_ATTRIBUTION_COMMITTER_ID=bob-uuid-5678
  CQ_ATTRIBUTION_OPERATOR=team-org-uuid-9999

Structural assertions:

- `attribution.author.id="alice-uuid-1234"`.
- `attribution.committer.id="bob-uuid-5678"` (different from author).
- `attribution.operator="team-org-uuid-9999"` (organization, distinct from both).
- All three flat fields populated with distinct values.
