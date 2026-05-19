# pcq.describe_run.record · pcq-1x-legacy

`run_record.json`에 pcq 2.x 신규 필드(`intent`/`integrity`/`contract_version`)가 **전혀 없는** 경우의 골든 픽스처 (R6 하위 호환, R8 T-PCQ2X-7).

## 시나리오

`pcq describe-run`이 `intent`/`integrity`/`contract_version` 키가 없는 1.x 형태의 `run_record.json`을 읽을 때:
- 정상적으로 처리되어야 함 (오류 없이 completed 상태로 출력)
- 출력에 `intent`, `intent_goal`, `integrity`, `integrity_content_hash`, `contract_version` 필드가 **없어야** 함

## 픽스처 구조

- `intent`: **없음** (absent, not null) — 1.x 레코드
- `integrity`: **없음** (absent, not null) — 1.x 레코드
- `contract_version`: **없음** (absent, not null) — 1.x 레코드
- `environment.pcq_version`: `"1.5.0"` — 구버전을 시뮬레이션

## Volatile 필드

`expected.json`에서 `"..."` placeholder 처리된 필드:
- `run_id`, `name`, `output_dir`, `last_updated_at`, `git_sha` — volatile (timestamp/random suffix)
- `python`, `platform` — 환경 종속

고정값 필드 (`expected.json`에 literal 지정):
- `status`: `"completed"`
- `validation_status`: `"pass"`
- `decision_facts.run_completed`: `true`

## R6 하위 호환 근거

`pcq 2.x`의 `json_contracts.py`에서 `intent`/`integrity`/`contract_version`은 모두 `optional` 섹션에 있고,
`describe.py`는 `rr.get("intent")`가 `None`이면 해당 필드를 설정하지 않는다.
`RunDescription.to_dict()`는 `None` 값 필드를 출력에서 제거한다.
따라서 1.x 레코드는 이 세 필드 없이도 유효한 `pcq.describe_run.record` 컨트랙트를 만족한다.

매처 규칙 및 placeholder 전체 정책: [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md)
