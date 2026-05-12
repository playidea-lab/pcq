"""pcq.fingerprint — 모달리티별 데이터 안전 통계 추출.

PII 정책 (R10):
- column 이름은 내부 sniffer(R5b)에서만 사용, 외부 emit 절대 금지.
- raw value, value distribution top-N, sample preview 절대 금지.
- 안전 필드만 emit: shape, type_counts, target_balance ratio, missing_ratio_max.

도메인 게이트 (R5 / R5b):
- domain ∈ {medical, financial, regulated} 시 자동 추출 비활성.
- R5b heuristic sniffer: X.columns에 의료/금융 키워드 포함 + domain=="general" 시
  추출 차단 + FINGERPRINT_DOMAIN_SUSPECTED_MEDICAL 경고 emit.

결정성 (R15): 모든 dict 키는 sorted 순서로 조립.
"""
from __future__ import annotations

# 의료 도메인 heuristic 키워드 (R5b sniffer 전용 — 외부 emit 금지)
MEDICAL_KEYWORDS: frozenset[str] = frozenset({
    "patient", "diagnosis", "hospital", "drug", "mrn", "icd", "dx", "rx",
    "gene", "allele", "prescription", "symptom", "medication", "ehr", "phi",
    "physician", "nurse", "clinical", "biomarker", "vital", "treatment",
})

# 금융 도메인 heuristic 키워드 (R5b sniffer 전용 — 외부 emit 금지)
FINANCIAL_KEYWORDS: frozenset[str] = frozenset({
    "account", "card", "pan", "ssn", "iban", "balance", "txn", "transaction",
    "credit", "debit", "routing", "swift", "wire", "ach", "cvv",
})

# 크기 버킷 (순서 중요: 첫 번째 임계값 미만이면 해당 버킷)
SIZE_BUCKETS: list[tuple[str, int]] = [
    ("small", 10_000),
    ("medium", 1_000_000),
    ("large", 100_000_000),
]  # else → "huge"


def _make_warning(code: str, message: str, field: str | None = None) -> dict:
    """표준 warning 엔트리를 생성합니다."""
    w: dict = {"code": code, "message": message}
    if field is not None:
        w["field"] = field
    return w


def _size_class(n: int) -> str:
    """샘플 수를 SIZE_BUCKETS 기준으로 분류합니다."""
    for name, threshold in SIZE_BUCKETS:
        if n < threshold:
            return name
    return "huge"


def _sniff_domain_keyword(columns: list[str]) -> str | None:
    """column 이름에서 의료/금융 키워드를 탐지합니다 (R5b).

    Substring containment 매칭 — 실제 column 이름은 `patient_id`, `mrn_number`,
    `account_num` 같은 형태가 일반적이므로 exact match는 거의 잡지 못함.
    R-WFP-4 review WEAKNESS 반영.

    Returns:
        "medical" | "financial" | None — 탐지된 도메인 또는 탐지 없음.
    """
    lower_cols = [c.lower() for c in columns]
    # 의료 키워드 확인 (substring)
    for col in lower_cols:
        if any(kw in col for kw in MEDICAL_KEYWORDS):
            return "medical"
    # 금융 키워드 확인 (substring)
    for col in lower_cols:
        if any(kw in col for kw in FINANCIAL_KEYWORDS):
            return "financial"
    return None


def extract_tabular(
    X: object,
    y: object,
    *,
    sample_rows: int = 100_000,
    domain: str = "general",
) -> tuple[dict, list[dict]]:
    """tabular 데이터에서 안전 통계를 추출합니다.

    Args:
        X: pandas DataFrame 또는 numpy array (또는 None).
        y: 타겟 배열 (분류/회귀 레이블).
        sample_rows: 대용량 시 샘플링할 최대 행 수.
        domain: 도메인 컨텍스트 (R5 게이트용).

    Returns:
        (fingerprint_tabular_subobject_dict, warnings) 튜플.
        R10: column 이름, raw value, distribution은 절대 emit 안 함.
        R15: 모든 키는 sorted 순서.
    """
    warnings: list[dict] = []

    # 빈 데이터 처리 (R11)
    if X is None:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 None입니다. tabular 통계를 추출할 수 없습니다.",
        )]

    # len() 시도
    try:
        n = len(X)  # type: ignore[arg-type]
    except TypeError:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X의 길이를 알 수 없습니다. tabular 통계를 추출할 수 없습니다.",
        )]

    if n == 0:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 빈 데이터셋(len=0)입니다. tabular 통계를 추출할 수 없습니다.",
        )]

    # R5b: pandas DataFrame 컬럼 heuristic sniffer
    columns: list[str] = []
    try:
        import pandas as pd  # noqa: PLC0415 — 지연 import

        if isinstance(X, pd.DataFrame):
            columns = list(X.columns)
    except ImportError:
        pass  # pandas 없으면 sniffer 건너뜀

    if columns and domain == "general":
        suspected = _sniff_domain_keyword(columns)
        if suspected is not None:
            code = f"FINGERPRINT_DOMAIN_SUSPECTED_{suspected.upper()}"
            return {}, [_make_warning(
                code,
                f"column 이름에서 {suspected} 도메인 키워드가 감지되었습니다. "
                "domain='general'이지만 민감 데이터가 의심됩니다. "
                "cq.yaml에 domain을 명시하거나 declared 경로를 사용하세요.",
            )]

    # 크기 클래스 결정
    sc = _size_class(n)

    # 대용량 시 샘플링 (large / huge)
    sampled = False
    X_work = X
    if sc in ("large", "huge") and n > sample_rows:
        sampled = True
        try:
            import numpy as np  # noqa: PLC0415
            import pandas as pd  # noqa: PLC0415

            if isinstance(X_work, pd.DataFrame):
                # 분류 타겟이 있으면 stratified sample 시도
                if y is not None:
                    try:
                        from sklearn.model_selection import train_test_split  # noqa: PLC0415

                        idx = list(range(n))
                        sample_idx, _ = train_test_split(
                            idx,
                            train_size=min(sample_rows, n - 1),
                            stratify=list(y),  # type: ignore[call-overload]
                            random_state=42,
                        )
                        X_work = X_work.iloc[sorted(sample_idx)]  # type: ignore[attr-defined]
                    except Exception:  # noqa: BLE001 — stratify 실패 시 random fallback
                        X_work = X_work.sample(n=sample_rows, random_state=42)  # type: ignore[attr-defined]
                else:
                    X_work = X_work.sample(n=sample_rows, random_state=42)  # type: ignore[attr-defined]
            elif isinstance(X_work, np.ndarray):
                rng = np.random.default_rng(42)
                idx = rng.choice(n, size=sample_rows, replace=False)
                X_work = X_work[np.sort(idx)]
        except Exception:  # noqa: BLE001 — 샘플링 실패 시 전체 사용
            sampled = False

        if sampled:
            warnings.append(_make_warning(
                "FINGERPRINT_SAMPLED",
                f"데이터가 큽니다(n={n}). {sample_rows}행으로 샘플링했습니다. "
                "통계는 근사값입니다.",
            ))

    # type_counts 계산 (R15: sorted keys)
    type_counts: dict[str, int] = {
        "categorical": 0,
        "datetime": 0,
        "numeric": 0,
        "text": 0,
    }
    n_columns: int | None = None
    missing_ratio_max: float | None = None

    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(X_work, pd.DataFrame):
            n_columns = len(X_work.columns)
            for col_name, dtype in X_work.dtypes.items():
                dt = str(dtype)
                if "int" in dt or "float" in dt or "complex" in dt:
                    type_counts["numeric"] += 1
                elif "datetime" in dt or "timedelta" in dt:
                    type_counts["datetime"] += 1
                elif "object" in dt or "string" in dt or "category" in dt:
                    # category → categorical, object → 길이 기반 text/categorical
                    if "category" in dt:
                        type_counts["categorical"] += 1
                    else:
                        # 첫 번째 non-null 값의 평균 길이로 text/categorical 구분
                        try:
                            sample_strs = X_work[col_name].dropna().head(100).astype(str)
                            avg_len = sample_strs.str.len().mean()
                            if avg_len > 50:
                                type_counts["text"] += 1
                            else:
                                type_counts["categorical"] += 1
                        except Exception:  # noqa: BLE001
                            type_counts["categorical"] += 1
                elif "bool" in dt:
                    type_counts["categorical"] += 1
                else:
                    type_counts["categorical"] += 1

            # missing_ratio_max — 가장 결측 비율이 높은 컬럼
            try:
                col_missing = X_work.isnull().mean()
                if not col_missing.empty:
                    missing_ratio_max = float(col_missing.max())
            except Exception:  # noqa: BLE001
                pass

    except ImportError:
        # pandas 없으면 numpy 폴백
        try:
            import numpy as np  # noqa: PLC0415

            if isinstance(X_work, np.ndarray):
                if X_work.ndim >= 2:
                    n_columns = X_work.shape[1]
                # numpy array는 dtype 통일됨
                dt = str(X_work.dtype)
                if "int" in dt or "float" in dt or "complex" in dt:
                    type_counts["numeric"] = n_columns or 1
                else:
                    type_counts["categorical"] = n_columns or 1
                # missing
                if "float" in dt:
                    try:
                        col_missing = np.isnan(X_work).mean(axis=0)
                        missing_ratio_max = float(col_missing.max())
                    except Exception:  # noqa: BLE001
                        pass
        except ImportError:
            pass

    # target_balance — 분류 시 majority class ratio
    target_balance: float | None = None
    n_classes: int | None = None
    try:
        import numpy as np  # noqa: PLC0415

        if y is not None:
            y_arr = np.asarray(y)
            unique_vals = np.unique(y_arr)
            n_classes = int(len(unique_vals))
            if n_classes >= 2:
                counts = np.array([float(np.sum(y_arr == v)) for v in unique_vals])
                total = float(counts.sum())
                if total > 0:
                    target_balance = float(counts.max() / total)
    except Exception:  # noqa: BLE001 — y 없거나 타입 불일치
        pass

    # 효과적 샘플 수 (샘플링 이후 크기)
    effective_n: int = n
    try:
        effective_n = len(X_work)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        effective_n = n

    # 결과 조립 (R15: sorted keys)
    result: dict = {
        "missing_ratio_max": missing_ratio_max,
        "n_classes": n_classes,
        "n_columns": n_columns,
        "sampled_rows": effective_n if sampled else None,
        "target_balance": target_balance,
        "type_counts": type_counts,
    }
    # sampled_rows 없을 때 키 제거 (None은 유지 — schema 호환성)
    # → 명세: sampled 아니면 sampled_rows=None, 있으면 effective sample size

    return result, warnings


def extract_image(
    X: object,
    y: object,
) -> tuple[dict, list[dict]]:
    """image 데이터에서 안전 통계를 추출합니다.

    Args:
        X: 이미지 배열 (N, H, W, C) 또는 (N, C, H, W) 형태.
        y: 타겟 레이블 (분류 시 n_classes 추출).

    Returns:
        (fingerprint_image_subobject_dict, warnings) 튜플.
        R10: raw value emit 금지.
        R15: sorted keys.
    """
    warnings: list[dict] = []

    if X is None:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 None입니다. image 통계를 추출할 수 없습니다.",
        )]

    input_shape: list[int] | None = None
    try:
        import numpy as np  # noqa: PLC0415

        arr = None
        # torch tensor 지원
        try:
            import torch  # noqa: PLC0415

            if isinstance(X, torch.Tensor):
                arr = X
                shape = tuple(arr.shape)  # type: ignore[union-attr]
        except ImportError:
            pass

        if arr is None:
            arr = np.asarray(X)
            shape = arr.shape

        # 첫 번째 차원은 N(샘플 수), 나머지가 input_shape
        if len(shape) >= 2:
            input_shape = list(shape[1:])
    except Exception:  # noqa: BLE001
        warnings.append(_make_warning(
            "FINGERPRINT_IMAGE_SHAPE_FAILED",
            "image input_shape 추출 실패.",
            field="image.input_shape",
        ))

    # n_classes — unique y 개수
    n_classes: int | None = None
    try:
        import numpy as np  # noqa: PLC0415

        if y is not None:
            y_arr = np.asarray(y)
            n_classes = int(len(np.unique(y_arr)))
    except Exception:  # noqa: BLE001
        pass

    # R15: sorted keys
    result: dict = {
        "input_shape": input_shape,
        "n_classes": n_classes,
    }
    return result, warnings


def extract_text(
    X: object,
    y: object,
) -> tuple[dict, list[dict]]:
    """text 데이터에서 안전 통계를 추출합니다.

    Args:
        X: 문자열 리스트 또는 배열.
        y: 타겟 (사용 안 함, API 일관성).

    Returns:
        (fingerprint_text_subobject_dict, warnings) 튜플.
        R10: 원문 내용 emit 금지. 토큰 길이 평균만.
        R15: sorted keys.
    """
    warnings: list[dict] = []

    if X is None:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 None입니다. text 통계를 추출할 수 없습니다.",
        )]

    avg_token_len: float | None = None
    vocab_kind: str | None = None

    try:
        # list[str] 또는 numpy array of strings
        texts: list[str]
        try:
            import numpy as np  # noqa: PLC0415

            arr = np.asarray(X)
            texts = [str(s) for s in arr.flat[:1000]]  # 최대 1000개 샘플로 추정
        except Exception:  # noqa: BLE001
            if hasattr(X, "__iter__"):
                texts = [str(s) for s in list(X)[:1000]]  # type: ignore[union-attr]
            else:
                texts = []

        if texts:
            token_lens = [len(s.split()) for s in texts]
            avg_token_len = float(sum(token_lens) / len(token_lens))

            # vocab_kind 추정 (간이 ASCII 기반)
            # 한국어 Unicode 범위: AC00-D7A3 (한글 음절), 1100-11FF (자모)
            sample_concat = "".join(texts[:50])
            korean_chars = sum(
                1 for c in sample_concat if "가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ"
            )
            ascii_chars = sum(1 for c in sample_concat if c.isascii())

            total_chars = len(sample_concat)
            if total_chars == 0:
                vocab_kind = "other"
            elif korean_chars / total_chars > 0.1:
                if ascii_chars / total_chars > 0.4:
                    vocab_kind = "multilingual"
                else:
                    vocab_kind = "korean"
            elif ascii_chars / total_chars > 0.8:
                vocab_kind = "english"
            else:
                vocab_kind = "multilingual"

    except Exception:  # noqa: BLE001
        warnings.append(_make_warning(
            "FINGERPRINT_TEXT_STATS_FAILED",
            "text 통계 추출 실패.",
        ))

    # R15: sorted keys
    result: dict = {
        "avg_token_len": avg_token_len,
        "vocab_kind": vocab_kind,
    }
    return result, warnings


def extract_time_series(
    X: object,
    y: object,
) -> tuple[dict, list[dict]]:
    """time_series 데이터에서 안전 통계를 추출합니다.

    Args:
        X: 시계열 배열 (N, T) 또는 (N, T, F) 형태.
        y: 타겟 (사용 안 함, API 일관성).

    Returns:
        (fingerprint_time_series_subobject_dict, warnings) 튜플.
        R15: sorted keys.
    """
    warnings: list[dict] = []

    if X is None:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 None입니다. time_series 통계를 추출할 수 없습니다.",
        )]

    seq_len: int | None = None
    freq: str = "irregular"

    try:
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(X)
        # 형태: (N, T) 또는 (N, T, F)
        if arr.ndim >= 2:
            seq_len = int(arr.shape[1])
        elif arr.ndim == 1:
            seq_len = int(arr.shape[0])

        # freq 추정 — best-effort (대부분 irregular)
        # pandas DatetimeIndex가 없으면 irregular
        freq = "irregular"

    except Exception:  # noqa: BLE001
        warnings.append(_make_warning(
            "FINGERPRINT_TS_STATS_FAILED",
            "time_series 통계 추출 실패.",
        ))

    # freq 추정 시도 — pandas Index
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(X, (pd.DataFrame, pd.Series)):
            idx = X.index
            if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None:
                freq_str = str(idx.freq)
                # 일반적인 빈도 매핑
                freq_map = {
                    "D": "daily", "H": "hourly", "h": "hourly",
                    "T": "other", "min": "other", "S": "other",
                }
                freq = freq_map.get(freq_str.split("-")[0], "other")
    except Exception:  # noqa: BLE001
        pass  # freq 추정 실패 시 irregular 유지

    # R15: sorted keys
    result: dict = {
        "freq": freq,
        "seq_len": seq_len,
    }
    return result, warnings


def extract_audio(
    X: object,
    y: object,
) -> tuple[dict, list[dict]]:
    """audio 데이터에서 안전 통계를 추출합니다.

    Args:
        X: 오디오 배열 (N, samples) 또는 메타데이터 dict.
        y: 타겟 (사용 안 함, API 일관성).

    Returns:
        (fingerprint_audio_subobject_dict, warnings) 튜플.
        R15: sorted keys.
    """
    warnings: list[dict] = []

    if X is None:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 None입니다. audio 통계를 추출할 수 없습니다.",
        )]

    sample_rate: int | None = None
    avg_duration_sec: float | None = None

    try:
        import numpy as np  # noqa: PLC0415

        # dict 형태 메타데이터 지원 ({"sample_rate": ..., "data": ...})
        if isinstance(X, dict):
            sample_rate = X.get("sample_rate")
            data = X.get("data")
            if data is not None and sample_rate is not None:
                arr = np.asarray(data)
                if arr.ndim >= 2:
                    avg_duration_sec = float(arr.shape[-1]) / float(sample_rate)
                elif arr.ndim == 1:
                    avg_duration_sec = float(arr.shape[0]) / float(sample_rate)
        else:
            arr = np.asarray(X)
            # (N, samples) 형태 추정 — sample_rate 없으면 None
            if arr.ndim == 2:
                avg_duration_sec = None  # sample_rate 없이 계산 불가
            elif arr.ndim == 1:
                avg_duration_sec = None

    except Exception:  # noqa: BLE001
        warnings.append(_make_warning(
            "FINGERPRINT_AUDIO_STATS_FAILED",
            "audio 통계 추출 실패.",
        ))

    # R15: sorted keys
    result: dict = {
        "avg_duration_sec": avg_duration_sec,
        "sample_rate": sample_rate,
    }
    return result, warnings


def extract_graph(
    X: object,
    y: object,
) -> tuple[dict, list[dict]]:
    """graph 데이터에서 안전 통계를 추출합니다.

    Args:
        X: 그래프 객체 — dict({"nodes": ..., "edges": ...}),
           networkx Graph, 또는 edge_index 배열.
        y: 타겟 (사용 안 함, API 일관성).

    Returns:
        (fingerprint_graph_subobject_dict, warnings) 튜플.
        R15: sorted keys.
    """
    warnings: list[dict] = []

    if X is None:
        return {}, [_make_warning(
            "FINGERPRINT_EMPTY_DATA",
            "X가 None입니다. graph 통계를 추출할 수 없습니다.",
        )]

    n_nodes: int | None = None
    n_edges: int | None = None
    n_node_features: int | None = None

    try:
        # dict 형태 지원
        if isinstance(X, dict):
            nodes = X.get("nodes")
            edges = X.get("edges")
            features = X.get("node_features") or X.get("x")
            if nodes is not None:
                n_nodes = int(len(nodes))  # type: ignore[arg-type]
            if edges is not None:
                n_edges = int(len(edges))  # type: ignore[arg-type]
            if features is not None:
                try:
                    import numpy as np  # noqa: PLC0415

                    feat_arr = np.asarray(features)
                    if feat_arr.ndim >= 2:
                        n_node_features = int(feat_arr.shape[-1])
                except Exception:  # noqa: BLE001
                    pass
        else:
            # networkx 지원
            try:
                import networkx as nx  # noqa: PLC0415

                if isinstance(X, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                    n_nodes = int(X.number_of_nodes())
                    n_edges = int(X.number_of_edges())
            except ImportError:
                pass

            # numpy edge_index 형태 — (2, E)
            if n_edges is None:
                try:
                    import numpy as np  # noqa: PLC0415

                    arr = np.asarray(X)
                    if arr.ndim == 2 and arr.shape[0] == 2:
                        n_edges = int(arr.shape[1])
                except Exception:  # noqa: BLE001
                    pass

    except Exception:  # noqa: BLE001
        warnings.append(_make_warning(
            "FINGERPRINT_GRAPH_STATS_FAILED",
            "graph 통계 추출 실패.",
        ))

    # R15: sorted keys
    result: dict = {
        "n_edges": n_edges,
        "n_node_features": n_node_features,
        "n_nodes": n_nodes,
    }
    return result, warnings
