"""inspect_project 가 v1.15 structured cq.yaml 인식.

list-style metrics 영구 호환 + dict-style metrics_schema + inputs 추출.
"""
from __future__ import annotations

import json
import textwrap

from pcq.agent import inspect_project


def _write_project(tmp_path, cq_yaml_content):
    """tmp_path 에 minimal pcq 프로젝트 셋업."""
    (tmp_path / "cq.yaml").write_text(
        textwrap.dedent(cq_yaml_content).lstrip(), encoding="utf-8"
    )
    (tmp_path / "train.py").write_text(
        "import pcq\ncq.config()\n", encoding="utf-8"
    )


def test_inspect_dict_style_metrics_schema(tmp_path):
    _write_project(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs:
          monitor: eval_iou
          mode: max
        metrics:
          eval_iou:
            mode: max
            split: val
            aggregation: macro
          eval_loss:
            mode: min
            split: val
        artifacts:
          - output/
    """)
    insp = inspect_project(tmp_path)
    assert "eval_iou" in insp.cq_yaml.declared_metrics
    assert "eval_loss" in insp.cq_yaml.declared_metrics
    assert insp.cq_yaml.metrics_schema["eval_iou"]["mode"] == "max"
    assert insp.cq_yaml.metrics_schema["eval_iou"]["split"] == "val"
    assert insp.cq_yaml.metrics_schema["eval_iou"]["aggregation"] == "macro"
    assert insp.cq_yaml.metrics_schema["eval_loss"]["mode"] == "min"


def test_inspect_inputs_section(tmp_path):
    _write_project(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          - eval_acc
        artifacts:
          - output/
        inputs:
          dataset:
            name: dental
            version: v12
            uri: cq://datasets/dental/v12
            split: train-val-2026-05-01
    """)
    insp = inspect_project(tmp_path)
    assert "dataset" in insp.cq_yaml.inputs
    ds = insp.cq_yaml.inputs["dataset"]
    assert ds["name"] == "dental"
    assert ds["version"] == "v12"
    # cq URI 는 opaque string — 변형 없이 보존.
    assert ds["uri"] == "cq://datasets/dental/v12"
    assert ds["split"] == "train-val-2026-05-01"


def test_inspect_legacy_list_metrics_still_works(tmp_path):
    """list-style metrics 영구 호환 — metrics_schema 비어 있어야."""
    _write_project(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          - epoch
          - eval_acc
        artifacts:
          - output/
    """)
    insp = inspect_project(tmp_path)
    assert insp.cq_yaml.declared_metrics == ["epoch", "eval_acc"]
    assert insp.cq_yaml.metrics_schema == {}
    assert insp.cq_yaml.inputs == {}


def test_inspect_to_dict_includes_structured_fields(tmp_path):
    _write_project(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          eval_iou:
            mode: max
        artifacts:
          - output/
        inputs:
          dataset:
            name: foo
    """)
    insp = inspect_project(tmp_path)
    d = insp.to_dict()
    # JSON-safe round-trip
    text = json.dumps(d)
    parsed = json.loads(text)
    cq_yaml = parsed["cq_yaml"]
    assert "metrics_schema" in cq_yaml
    assert cq_yaml["metrics_schema"]["eval_iou"]["mode"] == "max"
    assert "inputs" in cq_yaml
    assert cq_yaml["inputs"]["dataset"]["name"] == "foo"


def test_inspect_to_dict_omits_empty_structured_fields(tmp_path):
    """list-style 일 때 metrics_schema/inputs key 가 to_dict 에 없어야 — agent 친화."""
    _write_project(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics:
          - epoch
        artifacts:
          - output/
    """)
    insp = inspect_project(tmp_path)
    cq_yaml = insp.to_dict()["cq_yaml"]
    assert "metrics_schema" not in cq_yaml
    assert "inputs" not in cq_yaml


def test_inspect_dict_artifacts_supported(tmp_path):
    """artifacts 도 dict-style 지원 — keys 가 노출."""
    _write_project(tmp_path, """
        name: t
        cmd: uv run python train.py
        configs: {}
        metrics: [epoch]
        artifacts:
          output/:
            kind: run_output
          checkpoints/:
            kind: model
    """)
    insp = inspect_project(tmp_path)
    assert sorted(insp.cq_yaml.artifacts) == ["checkpoints/", "output/"]
