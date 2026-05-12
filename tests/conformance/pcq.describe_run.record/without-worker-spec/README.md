# pcq.describe_run.record · without-worker-spec

`run_record.json`에 `worker_spec` 필드가 없는 경우의 골든 픽스처 (T-WSPEC-7).
구 버전 레코드 또는 worker_spec 감지에 실패한 경우를 시뮬레이션한다.

## 시나리오

`pcq describe-run`이 `worker_spec` 키 자체가 없는 `run_record.json`을 읽을 때:
- `worker_spec` 필드를 출력에서 생략 (null 이 아닌 키 자체 부재)
- 4개의 플랫 표면 필드도 모두 생략:
  - `worker_spec_cpu_model` — 키 부재
  - `worker_spec_memory_gb` — 키 부재
  - `worker_spec_accelerator_kind` — 키 부재
  - `worker_spec_gpu_model_0` — 키 부재

## 하위 호환성

`worker_spec` 부재는 `describe.py`의 `to_dict()` 에서 `None` 값이 `_ALWAYS_KEEP_KEYS`에
포함되지 않아 자동으로 생략된다. 따라서 `expected.json`에서 이 필드들은 존재하지 않는다.

`worker_spec_gpu_model_0` 필드는 `null`이 아닌 **omitted (키 자체 부재)** 상태이다.
이는 `to_dict()` 의 None-strip 동작에 의한 것이며, 값이 명시적으로 `null`인 것과 다르다.
(field is omitted, not explicitly null — due to `to_dict()` None-strip behavior)

## Volatile 필드

`expected.json`에서 `"..."` placeholder 처리된 필드:
- `run_id`, `name`, `output_dir`, `last_updated_at`, `git_sha` — volatile
- `python`, `platform` — 환경 종속

매처 규칙 및 placeholder 전체 정책: [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md)
