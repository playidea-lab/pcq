# pcq.describe_run.record · with-fingerprint-tabular

`run_record.json`에 tabular `fingerprint` 객체가 포함된 경우의 골든 픽스처 (T-WFP-7).

## 시나리오

`pcq describe-run`이 `fingerprint` 키를 갖는 `run_record.json`을 읽을 때:
- `fingerprint` 중첩 객체를 그대로 출력 최상위에 노출
- 4개의 플랫 표면 필드도 함께 노출:
  - `fingerprint_modality`: `fingerprint.modality`
  - `fingerprint_task_kind`: `fingerprint.task_kind`
  - `fingerprint_n_samples`: `fingerprint.n_samples`
  - `fingerprint_size_class`: `fingerprint.size_class`

## 픽스처 구조

- `fingerprint.modality`: `"tabular"` — tabular 데이터셋
- `fingerprint.task_kind`: `"classification"` — 분류 태스크
- `fingerprint.n_samples`: `1000` — 1000개 샘플
- `fingerprint.size_class`: `"medium"` — 중간 크기 (10,000 ~ 1,000,000 범위 아래)
- `fingerprint.tabular`:
  - `n_columns`: `10` — 10개 컬럼
  - `type_counts`: `{numeric: 8, categorical: 2, datetime: 0, text: 0}` — 숫자형 8개, 범주형 2개
  - `target_balance`: `0.7` — 다수 클래스 비율 70%
  - `n_classes`: `2` — 이진 분류
  - `missing_ratio_max`: `null` — 결측치 없음
  - `sampled_rows`: `null` — 샘플링 미적용
- `fingerprint.domain`: `"general"` — 일반 도메인 (PII 게이트 비활성)
- `fingerprint.source`: `"detected"` — 자동 감지 결과

## Volatile 필드

`expected.json`에서 `"..."` placeholder 처리된 필드:
- `run_id`, `name`, `output_dir`, `last_updated_at`, `git_sha` — volatile (timestamp/random suffix)
- `python`, `platform` — 환경 종속

고정값 필드 (`expected.json`에 literal 지정):
- `fingerprint_modality`: `"tabular"`
- `fingerprint_task_kind`: `"classification"`
- `fingerprint_n_samples`: `1000`
- `fingerprint_size_class`: `"medium"`
- `fingerprint.schema_version`: `1`
- `fingerprint.tabular.n_columns`: `10`
- `fingerprint.tabular.target_balance`: `0.7`
- `fingerprint.tabular.n_classes`: `2`

매처 규칙 및 placeholder 전체 정책: [`spec/CONFORMANCE.md`](../../../../spec/CONFORMANCE.md)
