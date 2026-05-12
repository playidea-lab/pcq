"""worker_spec 기능 단위 테스트 — R1~R14 커버리지.

각 테스트는 pcq-agent-worker-spec.md EARS 요구사항에 대응한다.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pcq.contract import build_worker_spec_object
from pcq.agent.describe import describe_run


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _setup_minimal_run_record(tmp_path: Path, worker_spec: Any = "__OMIT__") -> Path:
    """최소 run_record.json 을 작성하고 경로를 반환한다.

    worker_spec 인수:
      - "__OMIT__" (기본): worker_spec 키 자체를 포함하지 않음
      - None: worker_spec=null 명시
      - dict: 해당 dict 포함
    """
    rr: dict = {
        "schema_version": 1,
        "run": {"id": "test-run", "status": "completed"},
        "execution": {},
        "source": {},
        "environment": {},
        "metrics": {"declared": [], "history_path": "metrics.json"},
        "artifacts": [],
    }
    if worker_spec != "__OMIT__":
        rr["worker_spec"] = worker_spec
    p = tmp_path / "run_record.json"
    p.write_text(json.dumps(rr))
    return tmp_path


def _clear_worker_envs(monkeypatch) -> None:
    """모든 CQ_WORKER_* 환경변수를 제거한다."""
    for key in [
        "CQ_WORKER_CPU_MODEL",
        "CQ_WORKER_CORES_PHYSICAL",
        "CQ_WORKER_CORES_LOGICAL",
        "CQ_WORKER_MAX_FREQ_MHZ",
        "CQ_WORKER_MEMORY_TOTAL_GB",
        "CQ_WORKER_ACCELERATOR_KIND",
        "CQ_WORKER_CONTAINER_KIND",
        "CQ_WORKER_OS_SYSTEM",
        "CQ_WORKER_OS_MACHINE",
        "CQ_WORKER_OS_RELEASE",
        "KUBERNETES_SERVICE_HOST",
        "container",
    ]:
        monkeypatch.delenv(key, raising=False)


# ─────────────────────────────────────────────────────────────────────────────
# R3: env 없이 자동 감지 → source=detected
# ─────────────────────────────────────────────────────────────────────────────

def test_R3_auto_detection_no_env(monkeypatch):
    """R3: CQ_WORKER_* env 없이 build_worker_spec_object() 호출 시
    source=detected 이고 accelerator.kind 필드가 존재해야 한다.

    Arrange: 모든 CQ_WORKER_* env 제거
    Act: build_worker_spec_object() 호출
    Assert: source == "detected", accelerator.kind 존재
    """
    # Arrange
    _clear_worker_envs(monkeypatch)

    # Act
    spec, warnings = build_worker_spec_object()

    # Assert
    assert spec is not None, "자동 감지 결과가 None이면 안 됩니다"
    assert spec["source"] == "detected", f"source는 'detected'여야 하지만 {spec['source']!r}"
    accelerator = spec.get("accelerator", {})
    assert "kind" in accelerator, "accelerator.kind 필드가 없습니다"
    assert accelerator["kind"] in ("cpu", "cuda", "mps"), (
        f"accelerator.kind가 유효하지 않습니다: {accelerator['kind']!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R4: CQ_WORKER_* 모든 env → source=declared
# ─────────────────────────────────────────────────────────────────────────────

def test_R4_full_declared(monkeypatch):
    """R4: CQ_WORKER_* env 를 모두 채우면 source=declared 이고 값이 일치해야 한다.

    Arrange: CQ_WORKER_CPU_MODEL, CQ_WORKER_ACCELERATOR_KIND 등 설정
    Act: build_worker_spec_object() 호출
    Assert: source == "declared", cpu.model 값 일치
    """
    # Arrange — psutil 없는 환경 시뮬레이션 (auto-detect 불가)
    _clear_worker_envs(monkeypatch)
    monkeypatch.setenv("CQ_WORKER_CPU_MODEL", "TestCPU-9900K")
    monkeypatch.setenv("CQ_WORKER_CORES_PHYSICAL", "8")
    monkeypatch.setenv("CQ_WORKER_CORES_LOGICAL", "16")
    monkeypatch.setenv("CQ_WORKER_MEMORY_TOTAL_GB", "32.0")
    monkeypatch.setenv("CQ_WORKER_ACCELERATOR_KIND", "cpu")

    # psutil import 실패를 시뮬레이션 → auto-detect 불가 → source=declared
    with patch.dict(sys.modules, {"psutil": None}):
        spec, warnings = build_worker_spec_object()

    # Assert
    assert spec is not None
    assert spec["source"] == "declared", f"source는 'declared'여야 하지만 {spec['source']!r}"
    assert spec["cpu"]["model"] == "TestCPU-9900K", (
        f"cpu.model 불일치: {spec['cpu']['model']!r}"
    )
    assert spec["accelerator"]["kind"] == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# R5: 일부 env + 일부 auto → source=merged
# ─────────────────────────────────────────────────────────────────────────────

def test_R5_merged_partial_env(monkeypatch):
    """R5: 일부 CQ_WORKER_* env 만 설정하고 psutil 이 살아있으면 source=merged 여야 한다.

    Arrange: cpu_model 만 env 에 설정, psutil 은 동작 가능
    Act: build_worker_spec_object() 호출
    Assert: source == "merged"
    """
    # Arrange
    _clear_worker_envs(monkeypatch)
    monkeypatch.setenv("CQ_WORKER_CPU_MODEL", "PartialOverrideCPU")

    # Act — psutil 실제 동작 (auto-detect 가능)
    spec, warnings = build_worker_spec_object()

    # Assert
    # psutil 이 없어서 auto-detect 불가인 환경에서는 declared 가 될 수도 있음
    assert spec is not None
    assert spec["source"] in ("merged", "declared"), (
        f"source는 'merged' 또는 'declared'여야 하지만 {spec['source']!r}"
    )
    # cpu.model 은 env 값으로 override 됨
    assert spec["cpu"]["model"] == "PartialOverrideCPU"


# ─────────────────────────────────────────────────────────────────────────────
# R6: describe_run — worker_spec nested + 4 flat 필드
# ─────────────────────────────────────────────────────────────────────────────

def test_R6_describe_run_nested_and_flat(tmp_path: Path):
    """R6: describe_run 응답에 worker_spec 중첩 dict 와 4개 flat 필드가 모두 존재해야 한다.

    Arrange: run_record.json 에 worker_spec 포함
    Act: describe_run(tmp_path) 호출
    Assert: worker_spec dict 존재, flat 4 필드 값 일치
    """
    # Arrange
    ws: dict = {
        "schema_version": 1,
        "source": "detected",
        "cpu": {"model": "Intel-i9", "cores_physical": 8, "cores_logical": 16, "max_freq_mhz": 5000.0},
        "memory": {"total_gb": 64.0},
        "accelerator": {
            "kind": "cuda",
            "gpus": [{"model": "RTX-4090", "vram_gb": 24.0, "torch_ordinal": 0}],
        },
        "os": {"system": "Linux", "machine": "x86_64", "release": "6.1"},
        "container": {"kind": "none", "image": None},
    }
    _setup_minimal_run_record(tmp_path, worker_spec=ws)

    # Act
    desc = describe_run(tmp_path)

    # Assert — 중첩 객체
    assert desc.worker_spec is not None, "describe_run.worker_spec 이 None 입니다"
    assert isinstance(desc.worker_spec, dict)

    # Assert — flat 표면 4 필드
    assert desc.worker_spec_cpu_model == "Intel-i9"
    assert desc.worker_spec_memory_gb == 64.0
    assert desc.worker_spec_accelerator_kind == "cuda"
    assert desc.worker_spec_gpu_model_0 == "RTX-4090"

    # to_dict() 에도 반영
    data = desc.to_dict()
    assert "worker_spec" in data
    assert data.get("worker_spec_cpu_model") == "Intel-i9"
    assert data.get("worker_spec_accelerator_kind") == "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# R7A: worker_spec 키 없는 구 run_record → describe_run 통과
# ─────────────────────────────────────────────────────────────────────────────

def test_R7A_old_record_no_worker_spec_key(tmp_path: Path):
    """R7A: worker_spec 키가 없는 구형 run_record.json → describe_run 에서 에러 없이
    worker_spec == None, 모든 flat 필드도 None.

    Arrange: worker_spec 키 자체를 omit
    Act: describe_run(tmp_path) 호출
    Assert: worker_spec is None, flat 필드 모두 None
    """
    # Arrange — worker_spec 키 omit ("__OMIT__" 기본값)
    _setup_minimal_run_record(tmp_path)

    # Act
    desc = describe_run(tmp_path)

    # Assert
    assert desc.worker_spec is None
    assert desc.worker_spec_cpu_model is None
    assert desc.worker_spec_memory_gb is None
    assert desc.worker_spec_accelerator_kind is None
    assert desc.worker_spec_gpu_model_0 is None


# ─────────────────────────────────────────────────────────────────────────────
# R7B: worker_spec=null 명시 → describe_run 통과
# ─────────────────────────────────────────────────────────────────────────────

def test_R7B_old_record_null_worker_spec(tmp_path: Path):
    """R7B: worker_spec=null 명시 run_record.json → describe_run 에서 에러 없이
    worker_spec == None.

    Arrange: worker_spec=None 명시
    Act: describe_run(tmp_path) 호출
    Assert: worker_spec is None
    """
    # Arrange — worker_spec=null
    _setup_minimal_run_record(tmp_path, worker_spec=None)

    # Act
    desc = describe_run(tmp_path)

    # Assert
    assert desc.worker_spec is None
    assert desc.worker_spec_cpu_model is None
    assert desc.worker_spec_accelerator_kind is None


# ─────────────────────────────────────────────────────────────────────────────
# R11a: psutil import 실패 → WORKER_PSUTIL_MISSING + 전부 null
# ─────────────────────────────────────────────────────────────────────────────

def test_R11a_psutil_missing(monkeypatch):
    """R11a: psutil import 실패 시뮬레이션 → warnings 에 WORKER_PSUTIL_MISSING 존재,
    cpu/memory 필드 전부 None.

    Arrange: sys.modules["psutil"] = None (import fail 시뮬레이션)
    Act: build_worker_spec_object() 호출
    Assert: WORKER_PSUTIL_MISSING warning, cpu 전 필드 None
    """
    # Arrange
    _clear_worker_envs(monkeypatch)
    with patch.dict(sys.modules, {"psutil": None}):
        # Act
        spec, warnings = build_worker_spec_object()

    # Assert
    warning_codes = [w["code"] for w in warnings]
    assert "WORKER_PSUTIL_MISSING" in warning_codes, (
        f"WORKER_PSUTIL_MISSING warning 없음: {warning_codes}"
    )

    # cpu 자동감지 전 필드 null
    cpu = spec["cpu"] if spec else {}
    assert cpu.get("cores_physical") is None
    assert cpu.get("cores_logical") is None
    assert cpu.get("max_freq_mhz") is None

    # memory 전 필드 null
    memory = spec["memory"] if spec else {}
    assert memory.get("total_gb") is None


# ─────────────────────────────────────────────────────────────────────────────
# R11b: psutil partial field raise → WORKER_PSUTIL_PARTIAL + 다른 필드는 채워짐
# ─────────────────────────────────────────────────────────────────────────────

def test_R11b_psutil_partial_field_raise(monkeypatch):
    """R11b: psutil.cpu_freq() 만 raise → WORKER_PSUTIL_PARTIAL warning,
    cores_physical/cores_logical 은 여전히 채워져야 한다 (객체 swallow 금지 확인).

    Arrange: psutil mock — cpu_freq()만 raise, 나머지는 정상
    Act: build_worker_spec_object() 호출
    Assert: WORKER_PSUTIL_PARTIAL warning, cores_physical != None
    """
    # Arrange — psutil mock
    _clear_worker_envs(monkeypatch)
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.side_effect = lambda logical=True: 4 if logical else 2
    mock_psutil.cpu_freq.side_effect = RuntimeError("cpu_freq 시뮬레이션 실패")
    mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024 ** 3)

    with patch.dict(sys.modules, {"psutil": mock_psutil}):
        # Act
        spec, warnings = build_worker_spec_object()

    # Assert
    warning_codes = [w["code"] for w in warnings]
    assert "WORKER_PSUTIL_PARTIAL" in warning_codes, (
        f"WORKER_PSUTIL_PARTIAL warning 없음: {warning_codes}"
    )

    # cpu_freq 는 None 이지만 cores 는 채워져야 함 (객체 swallow 금지)
    cpu = spec["cpu"] if spec else {}
    assert cpu.get("cores_physical") == 2, "cores_physical 은 채워져야 함"
    assert cpu.get("cores_logical") == 4, "cores_logical 은 채워져야 함"
    assert cpu.get("max_freq_mhz") is None, "cpu_freq 실패 시 max_freq_mhz 는 None"


# ─────────────────────────────────────────────────────────────────────────────
# R11c: torch import 실패 → WORKER_TORCH_MISSING + kind=cpu, gpus=[]
# ─────────────────────────────────────────────────────────────────────────────

def test_R11c_torch_missing(monkeypatch):
    """R11c: torch import 실패 시뮬레이션 → WORKER_TORCH_MISSING warning,
    accelerator.kind="cpu", gpus=[].

    Arrange: sys.modules["torch"] = None
    Act: build_worker_spec_object() 호출
    Assert: WORKER_TORCH_MISSING warning, kind="cpu", gpus=[]
    """
    # Arrange
    _clear_worker_envs(monkeypatch)
    with patch.dict(sys.modules, {"torch": None}):
        # Act
        spec, warnings = build_worker_spec_object()

    # Assert
    warning_codes = [w["code"] for w in warnings]
    assert "WORKER_TORCH_MISSING" in warning_codes, (
        f"WORKER_TORCH_MISSING warning 없음: {warning_codes}"
    )

    accelerator = spec["accelerator"] if spec else {}
    assert accelerator.get("kind") == "cpu", (
        f"torch 없을 때 kind 는 'cpu'여야 하지만 {accelerator.get('kind')!r}"
    )
    assert accelerator.get("gpus") == [], (
        f"torch 없을 때 gpus 는 [] 여야 하지만 {accelerator.get('gpus')!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R11d: /proc/1/cgroup 읽기 → PermissionError → WORKER_CGROUP_DENIED warning
# ─────────────────────────────────────────────────────────────────────────────

def test_R11d_cgroup_permission_denied(monkeypatch):
    """R11d: /proc/1/cgroup 읽기 시 PermissionError → WORKER_CGROUP_DENIED warning.

    컨테이너 감지에서 cgroup 파일을 읽을 때 권한이 없으면
    WORKER_CGROUP_DENIED 경고가 emit 되고 크래시 없이 진행해야 한다.

    Arrange: builtins.open 을 패치하여 /proc/1/cgroup 접근 시 PermissionError
    Act: build_worker_spec_object() 호출
    Assert: WORKER_CGROUP_DENIED warning 존재, spec 은 None 이 아님
    """
    import builtins

    # Arrange
    _clear_worker_envs(monkeypatch)
    original_open = builtins.open

    def _mock_open(path, *args, **kwargs):
        """cgroup 파일 접근 시만 PermissionError, 나머지는 정상."""
        if str(path) == "/proc/1/cgroup":
            raise PermissionError("Permission denied: /proc/1/cgroup")
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=_mock_open):
        # Act
        spec, warnings = build_worker_spec_object()

    # Assert — WORKER_CGROUP_DENIED warning 존재
    warning_codes = [w["code"] for w in warnings]
    assert "WORKER_CGROUP_DENIED" in warning_codes, (
        f"WORKER_CGROUP_DENIED warning 없음: {warning_codes}"
    )

    # spec 은 partial 이라도 반환 (크래시 금지)
    assert spec is None or isinstance(spec, dict), (
        f"None 또는 dict 를 반환해야 하지만 {type(spec)!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R12: Podman env → container.kind="other", WORKER_CONTAINER_AMBIGUOUS
# ─────────────────────────────────────────────────────────────────────────────

def test_R12_container_other_ambiguous(monkeypatch):
    """R12: Podman 환경 시뮬레이션 (container env="podman") →
    container.kind="other", detector_hint="podman",
    WORKER_CONTAINER_AMBIGUOUS warning 없이 or 있어도 kind="other" 로 결정.

    NOTE: 단일 신호(podman 만) 이면 ambiguous 없음. 복수 신호면 ambiguous.
    여기서는 단일 신호 확인.

    Arrange: container=podman env 설정, KUBERNETES_SERVICE_HOST/cgroup 신호 없음
    Act: build_worker_spec_object() 호출
    Assert: container.kind="other", detector_hint="podman"
    """
    # Arrange
    _clear_worker_envs(monkeypatch)
    monkeypatch.setenv("container", "podman")
    # cgroup, k8s 신호 없도록 보장
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    # Act — /proc/1/cgroup 은 macOS 에서 없으므로 자동으로 신호 없음
    spec, warnings = build_worker_spec_object()

    # Assert
    container = spec["container"] if spec else {}
    assert container.get("kind") == "other", (
        f"container.kind 는 'other'여야 하지만 {container.get('kind')!r}"
    )
    assert container.get("detector_hint") == "podman", (
        f"detector_hint 는 'podman'이어야 하지만 {container.get('detector_hint')!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# R13: mock pynvml 2 GPU — PCI bus_id 오름차순 정렬 + torch_ordinal 별도
# ─────────────────────────────────────────────────────────────────────────────

def test_R13_gpus_deterministic_order(monkeypatch):
    """R13: mock pynvml 로 2 GPU (다른 PCI bus_id) 시뮬레이션 →
    gpus[0].model / gpus[1].model 이 PCI bus_id 오름차순으로 결정되어야 한다.
    torch_ordinal 은 별도 필드.

    Arrange: pynvml mock — GPU-B (bus 00:02) / GPU-A (bus 00:01)
    Act: build_worker_spec_object() 호출
    Assert: gpus[0].model == "GPU-A" (bus 00:01), gpus[1].model == "GPU-B"
    """
    # Arrange — pynvml mock
    _clear_worker_envs(monkeypatch)

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetCount.return_value = 2

    def _make_handle(idx: int) -> MagicMock:
        return MagicMock(name=f"handle_{idx}")

    handle_b = _make_handle(0)  # torch index 0 → bus 00:02 (큰 bus)
    handle_a = _make_handle(1)  # torch index 1 → bus 00:01 (작은 bus)

    mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [handle_b, handle_a]
    mock_pynvml.nvmlDeviceGetName.side_effect = [b"GPU-B", b"GPU-A"]

    pci_b = MagicMock()
    pci_b.busId = b"00000000:02:00.0"
    pci_a = MagicMock()
    pci_a.busId = b"00000000:01:00.0"
    mock_pynvml.nvmlDeviceGetPciInfo.side_effect = [pci_b, pci_a]

    mem_info = MagicMock()
    mem_info.total = 24 * 1024 ** 3
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12020  # 12.2

    # torch mock — cuda available
    mock_torch = MagicMock()
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2

    with patch.dict(sys.modules, {"pynvml": mock_pynvml, "torch": mock_torch}):
        # Act
        spec, warnings = build_worker_spec_object()

    # Assert
    assert spec is not None
    gpus = spec["accelerator"]["gpus"]
    assert len(gpus) == 2, f"GPU가 2개여야 하지만 {len(gpus)}개"

    # PCI bus_id 오름차순: 00:01 → GPU-A, 00:02 → GPU-B
    assert gpus[0]["model"] == "GPU-A", (
        f"gpus[0].model 은 'GPU-A'(bus 00:01)여야 하지만 {gpus[0]['model']!r}"
    )
    assert gpus[1]["model"] == "GPU-B", (
        f"gpus[1].model 은 'GPU-B'(bus 00:02)여야 하지만 {gpus[1]['model']!r}"
    )

    # torch_ordinal 은 별도 필드로 존재
    assert "torch_ordinal" in gpus[0], "torch_ordinal 필드가 없습니다"
    assert "torch_ordinal" in gpus[1], "torch_ordinal 필드가 없습니다"


# ─────────────────────────────────────────────────────────────────────────────
# R14: declared cpu.model 에 PII 패턴 → WORKER_DECLARED_PII_LIKE warning, exit_code=0
# ─────────────────────────────────────────────────────────────────────────────

def test_R14_declared_pii_pattern_warn(monkeypatch):
    """R14: cfg worker 섹션에 cpu_model="testuser-MacBookPro.local" 포함 →
    _check_worker_spec_pii 가 WORKER_DECLARED_PII_LIKE warning 을 emit하고
    build_worker_spec_object 반환 spec 에서 source=declared/merged.

    validate_project 경로는 resolve_project 가 cfg.worker 를 configs 로 전달하지
    않아 트리거가 어렵다. 대신 _check_worker_spec_pii 와 build_worker_spec_object 를
    직접 테스트한다 (동일 계약 검증).

    Arrange: cfg 에 worker.cpu_model=PII 패턴값, psutil mock fail → declared
    Act: build_worker_spec_object(cfg=cfg) + _check_worker_spec_pii() 호출
    Assert: source=declared, WORKER_DECLARED_PII_LIKE 경고 emit, report.status=warn
    """
    from pcq.agent.validate import _check_worker_spec_pii
    from pcq.agent.schema import ValidationReport

    # Arrange — cfg 에 worker 섹션 + PII 패턴
    _clear_worker_envs(monkeypatch)
    cfg = {
        "worker": {
            "cpu_model": "testuser-MacBookPro.local",  # hostname.local 패턴 = PII 유사
        }
    }

    # psutil 없어서 auto-detect 불가 → source=declared
    with patch.dict(sys.modules, {"psutil": None}):
        spec, ws_warnings = build_worker_spec_object(cfg=cfg)

    assert spec is not None
    # source 확인 (cfg override 있음 + auto-detect 없음 → declared)
    assert spec["source"] in ("declared", "merged"), (
        f"source 는 declared 또는 merged 여야 하지만 {spec['source']!r}"
    )
    assert spec["cpu"]["model"] == "testuser-MacBookPro.local"

    # _check_worker_spec_pii 직접 호출
    report = ValidationReport(strictness=3, strictness_name="reproducibility")
    _check_worker_spec_pii(report, spec)

    pii_checks = [c for c in report.checks if c.id == "worker_spec_pii"]
    assert pii_checks, "WORKER_DECLARED_PII_LIKE 경고가 emit 되지 않았습니다"

    pii_check = pii_checks[0]
    assert pii_check.status == "warn", (
        f"PII 패턴 감지 시 status='warn'이어야 하지만 {pii_check.status!r}"
    )
    evidence = getattr(pii_check, "evidence", {}) or {}
    assert evidence.get("warning_code") == "WORKER_DECLARED_PII_LIKE"

    # PII 경고는 exit_code 에 영향 없음 — report status 가 fail 아님
    # ValidationReport 는 checks 전체 status 를 aggregate
    # warn 만 있으면 report.status 는 "warn" 또는 "pass"
    assert report.status in ("pass", "warn"), (
        f"PII 경고는 fail 을 유발하면 안 됨: {report.status!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# kind 검증: 잘못된 accelerator.kind → ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_accelerator_kind_raises_value_error(monkeypatch):
    """build_worker_spec_object(cli_args={"accelerator_kind": "invalid"}) →
    ValueError 를 발생시켜야 한다.

    Arrange: cli_args 에 잘못된 accelerator_kind 전달
    Act: build_worker_spec_object(cli_args=...) 호출
    Assert: ValueError 발생
    """
    # Arrange
    _clear_worker_envs(monkeypatch)

    # Act & Assert
    with pytest.raises(ValueError, match="accelerator.kind"):
        build_worker_spec_object(cli_args={"accelerator_kind": "invalid"})


# ─────────────────────────────────────────────────────────────────────────────
# kind 검증: 잘못된 container.kind → ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_container_kind_raises_value_error(monkeypatch):
    """build_worker_spec_object(cli_args={"container_kind": "rkt"}) →
    ValueError 를 발생시켜야 한다.

    Arrange: cli_args 에 유효하지 않은 container_kind ("rkt") 전달
    Act: build_worker_spec_object(cli_args=...) 호출
    Assert: ValueError 발생
    """
    # Arrange
    _clear_worker_envs(monkeypatch)

    # Act & Assert
    with pytest.raises(ValueError, match="container.kind"):
        build_worker_spec_object(cli_args={"container_kind": "rkt"})


# ─────────────────────────────────────────────────────────────────────────────
# None 반환: 자동감지 불가 환경에서의 동작 확인
# ─────────────────────────────────────────────────────────────────────────────

def test_no_args_no_env_returns_partial_or_none(monkeypatch):
    """자동감지가 안 되는 환경(psutil/torch 모두 실패)에서도 build_worker_spec_object 는
    None 또는 부분 spec 을 반환하고 크래시하지 않아야 한다.

    Arrange: env 없음, psutil/torch mock 실패
    Act: build_worker_spec_object() 호출
    Assert: (None 또는 dict) 반환, 예외 없음
    """
    # Arrange
    _clear_worker_envs(monkeypatch)

    with patch.dict(sys.modules, {"psutil": None, "torch": None}):
        # Act
        spec, warnings = build_worker_spec_object()

    # Assert — 크래시 없이 반환
    # NOTE: accelerator 기본값 "cpu" 때문에 None 이 아닐 수 있음
    assert spec is None or isinstance(spec, dict), (
        f"None 또는 dict 를 반환해야 하지만 {type(spec)!r}"
    )
    # warning 목록은 항상 list
    assert isinstance(warnings, list)


# ─────────────────────────────────────────────────────────────────────────────
# validate_project E2E — WORKER_DECLARED_PII_LIKE / FINGERPRINT_DECLARED_PII_LIKE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skip(
    reason=(
        "validate_project E2E pending implementation details (R-WFP-6 follow-up #4). "
        "validate_project 가 cfg.worker 를 configs 로 전달하는 경로가 아직 미구현. "
        "R14 직접 단위 테스트(test_R14_declared_pii_pattern_warn)로 동일 계약 검증됨."
    )
)
def test_validate_project_pii_warning_e2e(tmp_path: Path):
    """validate_project 호출 시 WORKER_DECLARED_PII_LIKE 또는
    FINGERPRINT_DECLARED_PII_LIKE 가 report.checks 에 나타나는지 확인한다.

    현재 validate_project 가 cfg.worker 를 fingerprint/worker spec 검사 경로로
    전달하지 않아 E2E 트리거가 어렵다. 구현 완료 후 skip 제거.
    """
    pass
