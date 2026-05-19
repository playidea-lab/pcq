# pcq.describe_run.record · pcq-2x-full

`run_record.json`에 pcq 2.x 신규 필드(`intent`/`integrity`/`contract_version`)가 모두 포함된 경우의 골든 픽스처 (R8 T-PCQ2X-7).

## 시나리오

`pcq describe-run`이 `intent`/`integrity`/`contract_version` 키를 갖는 `run_record.json`을 읽을 때:
- `intent` 중첩 객체를 최상위에 노출
- `intent_goal` 플랫 표면 필드도 함께 노출 (`intent.goal`)
- `integrity` 중첩 객체를 최상위에 노출
- `integrity_content_hash` 플랫 표면 필드도 함께 노출 (`integrity.content_hash`)
- `contract_version` 필드를 최상위에 노출

## 픽스처 구조

- `intent.goal`: `"baseline_reproduction"` — 기저선 재현 실험
- `intent.expected_baseline.metric`: `"eval_acc"`
- `intent.expected_baseline.value`: `0.95`
- `intent.tolerance.direction`: `"higher_is_better"`
- `intent.tolerance.margin`: `0.02`
- `integrity.content_hash`: `sha256:...` — 레코드 무결성 해시
- `integrity.hashed_fields`: leaf-path 목록
- `contract_version`: `"2.0"`

## Volatile 필드

`expected.json`에서 `"..."` placeholder 처리된 필드:
- `run_id`, `name`, `output_dir`, `last_updated_at`, `git_sha` — volatile (timestamp/random suffix)
- `python`, `platform` — 환경 종속
- `integrity.content_hash`, `integrity.hashed_fields`, `integrity_content_hash` — 실행 환경별 해시값

고정값 필드 (`expected.json`에 literal 지정):
- `intent.goal`: `"baseline_reproduction"`
- `intent_goal`: `"baseline_reproduction"`
- `intent.expected_baseline.metric`: `"eval_acc"`
- `intent.expected_baseline.value`: `0.95`
- `intent.tolerance.direction`: `"higher_is_better"`
- `intent.tolerance.margin`: `0.02`
- `contract_version`: `"2.0"`

매처 규칙 및 placeholder 전체 정책: [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md)
