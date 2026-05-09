"""pcq.agent.lineage — RunRecord parent chain traversal (v1.18).

pcq 은 parent_run_path 를 path string 으로 따라간다. resolution 은 파일 시스템
기반 — relative path 는 child 의 output_dir 기준으로 해석. 순환 detection +
max_depth truncation + missing parent 에 대해 graceful 동작.

CQ URI (cq://...) 는 opaque — pcq 은 follow 하지 않고, missing parent 로 기록.
URI resolution 은 CQ service 의 책임.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# parent chain 을 따라가는 기본 최대 깊이. 실제 실험 lineage 는 보통 ≤ 50.
_DEFAULT_MAX_DEPTH = 100


# v2.3: 의미 있는 falsy 값 (0/0.0/False) 을 보존하기 위해 항상 직렬화하는 필드.
# run_id/depth/status 는 chain 의 정체성 — 비어있어도 출력해서 "field exists but
# empty" 신호를 agent 에게 전달.
_LINEAGE_NODE_ALWAYS_KEEP = frozenset({"run_id", "depth", "status"})


@dataclass
class LineageNode:
    """chain 의 한 노드. 순환/missing 인 경우 note 로 표기."""

    run_id: str = ""
    output_dir: str | None = None       # absolute path (resolved)
    depth: int = 0
    name: str = ""
    status: str = ""
    target_metric: str | None = None
    best_value: float | None = None
    started_at: str | None = None
    parent_run_id: str | None = None
    note: str | None = None              # "missing", "circular", etc.

    def to_dict(self) -> dict:
        # v2.3: ALWAYS_KEEP 필드는 항상 출력 (빈 값도). 그 외는 빈 컨테이너/
        # None/빈 문자열만 제거하고 0/0.0/False 같은 falsy 숫자/bool 은 보존.
        # 이전 동작은 v in (None, "") 로 0/0.0 도 통과했지만, 더 명시적으로
        # 컨테이너 빈값까지 제거하고 의미 있는 zero 는 보존하도록 정정.
        out: dict = {}
        for k, v in self.__dict__.items():
            if k in _LINEAGE_NODE_ALWAYS_KEEP:
                out[k] = v
                continue
            # None / 빈 문자열 / 빈 list / 빈 dict 만 skip — 0 / 0.0 / False 는 의미 있음.
            if v is None or v == "" or v == [] or v == {}:
                continue
            out[k] = v
        return out


@dataclass
class LineageChain:
    """현재 → parent → ... 순서의 chain."""

    schema_version: int = 1
    chain: list[LineageNode] = field(default_factory=list)
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "chain": [n.to_dict() for n in self.chain],
            "truncated": self.truncated,
            "notes": self.notes,
        }


def _read_record(path: Path) -> dict | None:
    """run_record.json 읽기. path 가 file 또는 dir 모두 허용. 실패 시 None."""
    if path.is_file():
        rr_path = path
    elif path.is_dir():
        rr_path = path / "run_record.json"
        if not rr_path.exists():
            return None
    else:
        return None
    try:
        return json.loads(rr_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _node_from_record(rr: dict, output_dir: Path, depth: int) -> LineageNode:
    """run_record dict + output_dir → LineageNode (best_value 추출 포함)."""
    run = rr.get("run") or {}
    summary = rr.get("summary") or {}
    target = summary.get("target_metric")
    best = summary.get("best") or {}
    best_value: float | None = None
    if target and isinstance(best.get("metrics"), dict):
        bv = best["metrics"].get(target)
        if isinstance(bv, (int, float)):
            best_value = float(bv)
    return LineageNode(
        run_id=str(run.get("id", "")),
        output_dir=str(output_dir),
        depth=depth,
        name=str(run.get("name", "")),
        status=str(run.get("status", "")),
        started_at=run.get("started_at"),
        parent_run_id=run.get("parent_run_id"),
        target_metric=target,
        best_value=best_value,
    )


def _is_remote_uri(path_str: str) -> bool:
    """cq:// 또는 다른 URI scheme 이면 True (opaque, follow 안 함)."""
    if not path_str:
        return False
    # scheme:// 형태 (cq, s3, gs, http(s), file 등)
    if "://" in path_str:
        return True
    return False


def _find_project_root(start: Path) -> Path | None:
    """start 에서 위로 올라가며 첫 번째 cq.yaml 보유 dir 반환.

    v4.2 (GM-4): apply_plan 이 `_parent_run_path` 를 project_root 기준 relative
    (예: "output_gen0") 로 작성하면, child 의 output_dir 기준이 아니라 project
    root 기준으로 resolve 해야 한다. dogfood research/mcp-dogfood 회귀 fix.
    """
    cur = start.resolve()
    # 무한루프 방지 — Path.parents 는 root 까지 자연 종료.
    for candidate in (cur, *cur.parents):
        if (candidate / "cq.yaml").exists():
            return candidate
    return None


def _resolve_parent_path(
    parent_path_str: str, current_output: Path
) -> Path:
    """parent_run_path 를 절대 경로로 해석.

    우선순위:
      1. absolute → 그대로 사용.
      2. project_root 발견되고 해당 경로에 record 있음 → project_root 기준.
      3. 그 외 → current_output 기준 (backward compat).

    v4.2 (GM-4): 이전엔 항상 current_output 기준이라 child 가 output_gen1/
    안에서 _parent_run_path="output_gen0" 을 만났을 때 output_gen1/output_gen0
    로 잘못 해석했음. 이제 project_root/output_gen0 을 먼저 시도.
    """
    p = Path(parent_path_str)
    if p.is_absolute():
        return p

    project_root = _find_project_root(current_output)
    if project_root is not None:
        candidate = (project_root / p).resolve()
        # candidate 가 존재하고 record 가 읽히면 그것을 사용.
        # candidate 가 없거나 record 없는 경우엔 current_output 기준 fallback —
        # 기존 동작 보존 (relative parent_run_path 가 child's output_dir 기준
        # 으로 의도된 케이스 — test_lineage_three_generations_relative_path 처럼
        # ../run_a 같은 명시적 부모 디렉토리 표기).
        if _read_record(candidate) is not None:
            return candidate

    return (current_output / p).resolve()


def lineage(
    start: str | Path,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> LineageChain:
    """parent_run_path 를 따라 chain 생성.

    Args:
        start: output_dir (run_record.json 포함) 또는 run_record.json 직접.
        max_depth: 무한 chain 방지 (default 100).

    Returns:
        LineageChain — 현재 → parent → grandparent ... 순. 실패 시 빈 chain
        + notes 에 사유.
    """
    chain = LineageChain()
    start_path = Path(start).resolve()

    # 첫 노드 — start 가 read 가능해야 함.
    rr = _read_record(start_path)
    if rr is None:
        chain.notes.append(f"no run_record.json at {start_path}")
        return chain

    # output_dir 결정 (start 가 file 이면 parent dir).
    current_output = start_path if start_path.is_dir() else start_path.parent
    current_rr = rr
    seen_ids: set[str] = set()

    depth = 0
    while True:
        node = _node_from_record(current_rr, current_output, depth)

        # 순환 감지 — 같은 run_id 이 다시 등장하면 중단.
        if node.run_id and node.run_id in seen_ids:
            node.note = "circular reference detected"
            chain.chain.append(node)
            chain.notes.append(
                f"chain stopped at depth {depth}: circular reference"
            )
            break

        if node.run_id:
            seen_ids.add(node.run_id)
        chain.chain.append(node)

        # 다음 parent 결정.
        run = current_rr.get("run") or {}
        parent_path_str = run.get("parent_run_path")
        if not parent_path_str:
            break    # chain 끝.

        # CQ URI 등 remote scheme 은 opaque — follow 하지 않고 placeholder 추가.
        if _is_remote_uri(str(parent_path_str)):
            chain.chain.append(
                LineageNode(
                    run_id=str(run.get("parent_run_id") or ""),
                    output_dir=str(parent_path_str),
                    depth=depth + 1,
                    note=f"remote URI not followed: {parent_path_str}",
                )
            )
            chain.notes.append(
                f"remote URI at depth {depth+1}: {parent_path_str}"
            )
            break

        # v4.2 (GM-4): apply_plan 이 작성한 project-root-relative path 를
        # 우선 시도, 없으면 child's output_dir 기준으로 fallback.
        parent_path = _resolve_parent_path(str(parent_path_str), current_output)

        next_rr = _read_record(parent_path)
        if next_rr is None:
            chain.chain.append(
                LineageNode(
                    run_id=str(run.get("parent_run_id") or ""),
                    output_dir=str(parent_path),
                    depth=depth + 1,
                    note=f"parent_run_path not found: {parent_path}",
                )
            )
            chain.notes.append(f"parent missing at depth {depth+1}")
            break

        depth += 1
        if depth >= max_depth:
            chain.truncated = True
            chain.notes.append(f"max_depth={max_depth} reached")
            break

        current_output = (
            parent_path if parent_path.is_dir() else parent_path.parent
        )
        current_rr = next_rr

    return chain


def is_descendant_of(
    child_path: str | Path,
    ancestor_id: str,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> bool:
    """child 가 ancestor_id 의 후손인지 (compare_runs 보조용).

    child 자기 자신은 ancestor 가 아니다 — chain[0] 은 제외하고 검색.
    """
    if not ancestor_id:
        return False
    chain = lineage(child_path, max_depth=max_depth)
    return any(node.run_id == ancestor_id for node in chain.chain[1:])
