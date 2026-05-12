# pcq.describe_run.record · with-worker-spec

`run_record.json`에 `worker_spec` 객체가 포함된 경우의 골든 픽스처 (T-WSPEC-7).

## 시나리오

`pcq describe-run`이 `worker_spec` 키를 갖는 `run_record.json`을 읽을 때:
- `worker_spec` 중첩 객체를 그대로 출력 최상위에 노출
- 4개의 플랫 표면 필드도 함께 노출:
  - `worker_spec_cpu_model`: `worker_spec.cpu.model`
  - `worker_spec_memory_gb`: `worker_spec.memory.total_gb`
  - `worker_spec_accelerator_kind`: `worker_spec.accelerator.kind`
  - `worker_spec_gpu_model_0`: `worker_spec.accelerator.gpus[0].model` (없으면 null)

## 픽스처 구조

- `worker_spec.source`: `"detected"` — 자동 감지 결과만 사용한 경우
- `worker_spec.accelerator.kind`: `"mps"` — Apple Silicon MPS 환경
- `worker_spec.accelerator.gpus`: `[]` — MPS에서 GPU 목록은 빈 배열 (discrete GPU 없음)
- `worker_spec_gpu_model_0`: `null` — gpus 배열이 비어 있어 첫 번째 GPU 없음

## Volatile 필드

`expected.json`에서 `"..."` placeholder 처리된 필드:
- `run_id`, `name`, `output_dir`, `last_updated_at`, `git_sha` — volatile (timestamp/random suffix)
- `python`, `platform` — 환경 종속
- `worker_spec.os.release` — OS 릴리즈 버전 (환경 종속)

고정값 필드 (`expected.json`에 literal 지정):
- `worker_spec_cpu_model`: `"Apple M3 Pro"`
- `worker_spec_memory_gb`: `36.0`
- `worker_spec_accelerator_kind`: `"mps"`
- `worker_spec.schema_version`: `1`

매처 규칙 및 placeholder 전체 정책: [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md)
