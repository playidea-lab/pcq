"""lineage — RunRecord parent chain traversal (v1.18)."""
from __future__ import annotations

import json
from pathlib import Path

from pcq.agent.lineage import (
    LineageChain,
    LineageNode,
    is_descendant_of,
    lineage,
)


def _make_record(
    out_dir: Path,
    run_id: str,
    parent_id: str | None = None,
    parent_path: str | None = None,
    best: float = 0.5,
    target: str = "eval_acc",
    name: str = "",
    status: str = "completed",
) -> Path:
    """run_record.json minimal shape 생성 헬퍼."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rr: dict = {
        "schema_version": 1,
        "run": {"id": run_id, "status": status},
        "execution": {},
        "source": {},
        "environment": {},
        "metrics": {"declared": [], "history_path": "metrics.json"},
        "artifacts": [],
        "summary": {
            "target_metric": target,
            "best": {"epoch": 0, "metrics": {target: best}},
        },
    }
    if name:
        rr["run"]["name"] = name
    if parent_id:
        rr["run"]["parent_run_id"] = parent_id
    if parent_path:
        rr["run"]["parent_run_path"] = parent_path
    rr_path = out_dir / "run_record.json"
    rr_path.write_text(json.dumps(rr), encoding="utf-8")
    return rr_path


def test_lineage_returns_chain_dataclass(tmp_path):
    _make_record(tmp_path / "run_a", "run_a")
    chain = lineage(tmp_path / "run_a")
    assert isinstance(chain, LineageChain)
    assert chain.schema_version == 1


def test_lineage_single_run(tmp_path):
    _make_record(tmp_path / "run_a", "run_a", best=0.42)
    chain = lineage(tmp_path / "run_a")
    assert len(chain.chain) == 1
    node = chain.chain[0]
    assert isinstance(node, LineageNode)
    assert node.run_id == "run_a"
    assert node.depth == 0
    assert node.best_value == 0.42
    assert node.target_metric == "eval_acc"
    assert chain.truncated is False


def test_lineage_two_generations(tmp_path):
    _make_record(tmp_path / "run_a", "run_a", best=0.5)
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_a",
        parent_path=str(tmp_path / "run_a"),
        best=0.7,
    )
    chain = lineage(tmp_path / "run_b")
    assert len(chain.chain) == 2
    assert chain.chain[0].run_id == "run_b"
    assert chain.chain[0].depth == 0
    assert chain.chain[0].best_value == 0.7
    assert chain.chain[1].run_id == "run_a"
    assert chain.chain[1].depth == 1
    assert chain.chain[1].best_value == 0.5


def test_lineage_three_generations_relative_path(tmp_path):
    """parent_run_path 가 상대경로면 child 의 output_dir 기준으로 해석."""
    _make_record(tmp_path / "run_a", "run_a")
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_a",
        parent_path="../run_a",
    )
    _make_record(
        tmp_path / "run_c",
        "run_c",
        parent_id="run_b",
        parent_path="../run_b",
    )
    chain = lineage(tmp_path / "run_c")
    assert [n.run_id for n in chain.chain] == ["run_c", "run_b", "run_a"]


def test_lineage_missing_parent(tmp_path):
    """parent_run_path 가 가리키는 곳에 record 가 없으면 placeholder + note."""
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_ghost",
        parent_path=str(tmp_path / "ghost_dir"),
    )
    chain = lineage(tmp_path / "run_b")
    assert len(chain.chain) == 2
    placeholder = chain.chain[1]
    assert placeholder.run_id == "run_ghost"
    assert placeholder.note and "not found" in placeholder.note
    assert any("missing" in n for n in chain.notes)


def test_lineage_circular(tmp_path):
    """A → B → A 순환 — 두 번째로 만나는 동일 run_id 에서 중단."""
    _make_record(
        tmp_path / "run_a",
        "run_a",
        parent_id="run_b",
        parent_path=str(tmp_path / "run_b"),
    )
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_a",
        parent_path=str(tmp_path / "run_a"),
    )
    chain = lineage(tmp_path / "run_a", max_depth=10)
    # 순환 감지 — 마지막 node 의 note 또는 chain.notes 에 기록.
    has_circular = any(
        "circular" in (n.note or "") for n in chain.chain
    ) or any("circular" in note for note in chain.notes)
    assert has_circular


def test_lineage_max_depth_truncate(tmp_path):
    """max_depth=1 이면 root + parent 1 단계만 따라가고 truncated=True."""
    _make_record(
        tmp_path / "run_a",
        "run_a",
        parent_id="run_p1",
        parent_path=str(tmp_path / "run_p1"),
    )
    _make_record(
        tmp_path / "run_p1",
        "run_p1",
        parent_id="run_p2",
        parent_path=str(tmp_path / "run_p2"),
    )
    _make_record(tmp_path / "run_p2", "run_p2")
    chain = lineage(tmp_path / "run_a", max_depth=1)
    assert chain.truncated is True


def test_is_descendant_of_two_generations(tmp_path):
    _make_record(tmp_path / "run_a", "run_a")
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_a",
        parent_path=str(tmp_path / "run_a"),
    )
    assert is_descendant_of(tmp_path / "run_b", "run_a") is True
    assert is_descendant_of(tmp_path / "run_a", "run_b") is False


def test_is_descendant_of_self_returns_false(tmp_path):
    """run 자기 자신은 자기의 ancestor 가 아니다."""
    _make_record(tmp_path / "run_a", "run_a")
    assert is_descendant_of(tmp_path / "run_a", "run_a") is False


def test_is_descendant_of_empty_ancestor_id(tmp_path):
    _make_record(tmp_path / "run_a", "run_a")
    assert is_descendant_of(tmp_path / "run_a", "") is False


def test_lineage_no_record(tmp_path):
    """존재하지 않는 path — 빈 chain + notes."""
    chain = lineage(tmp_path / "nonexistent")
    assert chain.chain == []
    assert chain.notes
    assert any("no run_record" in n for n in chain.notes)


def test_lineage_accepts_run_record_json_file(tmp_path):
    """start 인자가 run_record.json 직접일 때도 동작."""
    rr_path = _make_record(tmp_path / "run_a", "run_a")
    chain = lineage(rr_path)
    assert len(chain.chain) == 1
    assert chain.chain[0].run_id == "run_a"


def test_lineage_to_dict_serializable(tmp_path):
    _make_record(tmp_path / "run_a", "run_a")
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_a",
        parent_path=str(tmp_path / "run_a"),
    )
    chain = lineage(tmp_path / "run_b")
    d = chain.to_dict()
    # JSON round-trip 으로 직렬화 가능한지 검증.
    serialized = json.dumps(d)
    loaded = json.loads(serialized)
    assert loaded["schema_version"] == 1
    assert len(loaded["chain"]) == 2


def test_lineage_remote_uri_not_followed(tmp_path):
    """parent_run_path 가 cq:// URI 이면 follow 하지 않고 placeholder."""
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_remote",
        parent_path="cq://runs/abc123",
    )
    chain = lineage(tmp_path / "run_b")
    assert len(chain.chain) == 2
    assert chain.chain[1].run_id == "run_remote"
    assert chain.chain[1].note and "remote URI" in chain.chain[1].note


def test_lineage_node_to_dict_strips_empty(tmp_path):
    _make_record(tmp_path / "run_a", "run_a")
    chain = lineage(tmp_path / "run_a")
    node_dict = chain.chain[0].to_dict()
    assert node_dict["run_id"] == "run_a"
    # 비어있는 note / parent_run_id 등은 제거.
    assert "note" not in node_dict
    assert "parent_run_id" not in node_dict


# ── v2.3: Lineage best_value/name 보강 ────────────────────────────────


def test_lineage_extracts_best_value_for_ancestors(tmp_path):
    """v2.3: depth>0 노드도 best_value/name 정상 추출 (audit P1 #2)."""
    _make_record(tmp_path / "run_a", "run_a", best=0.42, name="baseline")
    _make_record(
        tmp_path / "run_b",
        "run_b",
        parent_id="run_a",
        parent_path=str(tmp_path / "run_a"),
        best=0.38,
        name="tuned",
    )
    chain = lineage(tmp_path / "run_b")
    assert len(chain.chain) == 2
    # depth 0 (head)
    assert chain.chain[0].best_value == 0.38
    assert chain.chain[0].name == "tuned"
    # depth 1 (ancestor) — name + best_value 둘 다 살아있어야 함
    assert chain.chain[1].best_value == 0.42
    assert chain.chain[1].name == "baseline"
    # JSON serialization 도 정상 (이전엔 빈 name 만 누락됐을 수 있음)
    d = chain.to_dict()
    assert d["chain"][1]["best_value"] == 0.42
    assert d["chain"][1]["name"] == "baseline"


def test_lineage_node_keeps_zero_best_value(tmp_path):
    """v2.3: 0.0 같은 falsy 값도 to_dict 에서 보존 (이전 to_dict 버그)."""
    _make_record(tmp_path / "run_a", "run_a", best=0.0)
    chain = lineage(tmp_path / "run_a")
    assert chain.chain[0].best_value == 0.0
    d = chain.to_dict()
    # 0.0 이 to_dict 에서 살아있어야 — 이전엔 falsy 로 필터링되지 않았는지 확인.
    assert d["chain"][0].get("best_value") == 0.0


def test_lineage_node_to_dict_keeps_run_id_status_even_when_empty():
    """v2.3: ALWAYS_KEEP 필드 (run_id/depth/status) 는 빈 값도 출력."""
    node = LineageNode()  # 모든 필드 default
    d = node.to_dict()
    assert "run_id" in d
    assert "depth" in d
    assert "status" in d
    # 그 외는 누락.
    assert "name" not in d
    assert "best_value" not in d


def test_lineage_node_to_dict_strips_empty_lists_and_dicts():
    """v2.3: 빈 list/dict 는 제거되되 0/0.0/False 는 보존."""
    node = LineageNode(
        run_id="x", best_value=0.0, depth=2, name="", status="completed"
    )
    d = node.to_dict()
    assert d["best_value"] == 0.0   # 0.0 is meaningful
    assert d["depth"] == 2
    assert d["status"] == "completed"
    assert "name" not in d           # empty string stripped


def test_finalize_run_propagates_yaml_top_level_name(tmp_path, monkeypatch):
    """v2.3: cq.yaml 의 top-level name 이 RunRecord.run.name 에 propagate.

    audit P1 #2 의 root cause — 이전엔 cfg.name 만 읽어 cq.yaml top-level
    name 이 누락됐고, lineage 표시에서 빈 name 이 되었음.
    """
    import pcq

    # cq.yaml 작성 (top-level name 만, configs 에는 name 없음)
    project_root = tmp_path / "proj"
    project_root.mkdir()
    (project_root / "cq.yaml").write_text(
        "name: my-experiment\n"
        "cmd: uv run python train.py\n"
        "configs:\n"
        "  output_dir: output\n",
        encoding="utf-8",
    )
    # train artifact 들 작성
    output_dir = project_root / "output"
    output_dir.mkdir()
    cfg_path = project_root / "cfg.json"
    cfg_path.write_text(
        json.dumps({"output_dir": str(output_dir)}), encoding="utf-8"
    )
    monkeypatch.setenv("CQ_CONFIG_JSON", str(cfg_path))
    monkeypatch.chdir(project_root)
    pcq.save_metrics([{"epoch": 0, "eval_loss": 0.5}])
    pcq.save_run_summary(history=[{"epoch": 0, "eval_loss": 0.5}])
    pcq.save_manifest()
    rr_path = pcq.finalize_run()

    rr = json.loads(rr_path.read_text(encoding="utf-8"))
    # top-level name 이 RunInfo.name 으로 propagate 되어야 함
    assert rr["run"].get("name") == "my-experiment"
