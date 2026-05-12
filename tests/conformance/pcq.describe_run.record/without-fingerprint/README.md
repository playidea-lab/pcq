# pcq.describe_run.record · without-fingerprint

`run_record.json`에 `fingerprint` 키가 없는 경우의 골든 픽스처 (T-WFP-7).

## 시나리오

`pcq describe-run`이 `fingerprint` 키가 없는 `run_record.json`을 읽을 때 (옛 레코드 또는 fingerprint 미사용 경우):
- `fingerprint` 중첩 객체가 출력에 포함되지 않아야 함
- 플랫 표면 필드 4개도 출력에 포함되지 않아야 함:
  - `fingerprint_modality` — absent
  - `fingerprint_task_kind` — absent
  - `fingerprint_n_samples` — absent
  - `fingerprint_size_class` — absent

## 픽스처 구조

- `run_record.json`에 `fingerprint` 키 자체가 없음 (옛 레코드)
- `to_dict` None-strip 패턴: `RunDescription.fingerprint=None`이므로 직렬화 시 키 자체가 제거됨

## Volatile 필드

`expected.json`에서 `"..."` placeholder 처리된 필드:
- `run_id`, `name`, `output_dir`, `last_updated_at`, `git_sha` — volatile (timestamp/random suffix)
- `python`, `platform` — 환경 종속

고정값 필드 (`expected.json`에 literal 지정):
- `status`: `"completed"`
- `validation_status`: `"pass"`
- `decision_facts` 전체 — 실행 결과 고정값

## None-strip 동작

`RunDescription.to_dict()`는 `None` 값을 `_ALWAYS_KEEP_KEYS`에 없으면 출력에서 제거한다.
`fingerprint`, `fingerprint_modality`, `fingerprint_task_kind`, `fingerprint_n_samples`,
`fingerprint_size_class`는 모두 `None`이 기본값이며 `_ALWAYS_KEEP_KEYS`에 속하지 않으므로,
`run_record.json`에 `fingerprint` 키가 없으면 출력에도 포함되지 않는다.

매처 규칙 및 placeholder 전체 정책: [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md)
