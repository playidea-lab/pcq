"""pcq.agent.yaml_io — YAML reader/writer.

ruamel.yaml 이 설치되어 있으면 comment / formatting / quoting 을 보존한다
(round-trip). 없으면 v1.10 minimal writer 로 fallback — cq.yaml 의 표준 구조만
다루며 사용자가 손으로 쓴 주석은 보존되지 않는다 (rewrite).

Optional install:
    uv add pcq[yaml]    # ruamel.yaml>=0.17

Minimal writer 지원:
  - top-level scalar (name/cmd 등)
  - 1-2 level nested dict (configs.<key>, configs._overrides_data.<atom>)
  - list of scalars (metrics, artifacts)

Minimal writer 미지원 (의도적):
  - YAML anchors / aliases
  - multiline string literals (|, >)
  - complex list-of-dict 구조 (한 줄 inline 만 부분 지원)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _detect_ruamel():
    """ruamel.yaml 설치 여부 감지. 설치 안 됐으면 None."""
    try:
        from ruamel.yaml import YAML  # type: ignore[import-not-found]
        return YAML
    except ImportError:
        return None


_RUAMEL_YAML = _detect_ruamel()


def _ruamel_instance():
    """ruamel.yaml YAML instance — round-trip 모드 + cq.yaml 스타일.

    ruamel 미설치 시 None 반환.
    """
    YAML = _RUAMEL_YAML
    if YAML is None:
        return None
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    return yaml


# YAML 의 reserved scalar tokens — quoting 강제
_YAML_RESERVED_LOWER = {
    "yes", "no", "true", "false", "null", "~", "on", "off",
}


def _quote_str(s: str) -> str:
    """YAML scalar string quoting — 안전 plain 또는 JSON-style quoted.

    숫자처럼 보이거나 reserved token 또는 특수문자 포함 시 JSON-quote.
    """
    if s == "":
        return '""'
    # reserved 또는 숫자 시작이면 quote
    if s.lower() in _YAML_RESERVED_LOWER:
        return json.dumps(s)
    if s[0].isdigit() or s[0] == "-" and len(s) > 1 and s[1].isdigit():
        return json.dumps(s)
    # plain scalar 허용 문자: 영숫자 + . _ / -
    safe = all(c.isalnum() or c in "._/-" for c in s)
    if safe:
        return s
    return json.dumps(s)


def _emit_scalar(v: Any) -> str:
    """단일 scalar → YAML 표현."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return _quote_str(v)
    # 그 외 (dict/list 가 아닌데 여기로 옴) — JSON 문자열로
    return _quote_str(json.dumps(v))


def _emit_value(value: Any, indent: int) -> list[str]:
    """value (dict/list/scalar) → YAML lines. indent 는 들여쓰기 깊이."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        if not value:
            return [f"{pad}{{}}"]
        for k, v in value.items():
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{pad}{k}: {{}}")
                else:
                    lines.append(f"{pad}{k}:")
                    lines.extend(_emit_value(v, indent + 1))
            elif isinstance(v, list):
                if not v:
                    lines.append(f"{pad}{k}: []")
                else:
                    lines.append(f"{pad}{k}:")
                    lines.extend(_emit_value(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {_emit_scalar(v)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{pad}[]"]
        for item in value:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{pad}- {{}}")
                else:
                    # 첫 키는 - key: 형태로, 나머지는 들여쓰기.
                    sub_lines = _emit_value(item, indent + 1)
                    if sub_lines:
                        first = sub_lines[0].lstrip()
                        lines.append(f"{pad}- {first}")
                        for s in sub_lines[1:]:
                            lines.append(s)
            elif isinstance(item, list):
                # 중첩 list — 인라인 JSON 으로 fallback
                lines.append(f"{pad}- {json.dumps(item)}")
            else:
                lines.append(f"{pad}- {_emit_scalar(item)}")
        return lines
    return [f"{pad}{_emit_scalar(value)}"]


def write_yaml(data: Any, path: Path | str) -> None:
    """data → YAML file.

    ruamel.yaml 이 있으면 사용 (read 한 round-trip object 의 comment/포맷 보존).
    없으면 v1.10 minimal writer fallback.
    """
    p = Path(path)
    if _RUAMEL_YAML is not None:
        yaml = _ruamel_instance()
        with p.open("w", encoding="utf-8") as f:
            yaml.dump(data, f)
        return
    _write_minimal(data, p)


def read_yaml(path: Path | str) -> Any:
    """YAML file → dict (또는 ruamel CommentedMap, dict 호환).

    ruamel.yaml 이 있으면 round-trip object (CommentedMap) 반환 — dict 처럼
    동작하며 isinstance(x, dict) 는 통과한다.
    """
    p = Path(path)
    if _RUAMEL_YAML is not None:
        yaml = _ruamel_instance()
        with p.open("r", encoding="utf-8") as f:
            data = yaml.load(f)
        return data if data is not None else {}
    return _parse_yaml(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────
# Minimal writer (ruamel 미사용 fallback)
# ─────────────────────────────────────────────────────────────────────


def _write_minimal(data: Any, path: Path) -> None:
    """cq.yaml-style minimal YAML 출력. 최상위는 dict.

    표준 키 순서 (cq.yaml convention):
      name → cmd → configs → metrics → artifacts → 기타
    각 top-level 키 사이에 빈 줄.
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"write_yaml: top-level must be dict, got {type(data).__name__}"
        )
    lines: list[str] = []
    ordered_keys = ["name", "cmd", "configs", "metrics", "artifacts"]
    seen: set[str] = set()
    for k in ordered_keys:
        if k in data:
            seen.add(k)
            lines.extend(_emit_top_key(k, data[k]))
            lines.append("")
    for k, v in data.items():
        if k in seen:
            continue
        lines.extend(_emit_top_key(k, v))
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    Path(path).write_text(text, encoding="utf-8")


def _emit_top_key(key: str, value: Any) -> list[str]:
    """top-level 한 키 emit."""
    if isinstance(value, dict):
        if not value:
            return [f"{key}: {{}}"]
        out = [f"{key}:"]
        out.extend(_emit_value(value, 1))
        return out
    if isinstance(value, list):
        if not value:
            return [f"{key}: []"]
        out = [f"{key}:"]
        out.extend(_emit_value(value, 1))
        return out
    return [f"{key}: {_emit_scalar(value)}"]


# ─────────────────────────────────────────────────────────────────────
# Minimal reader (ruamel 미사용 fallback) — write_yaml 출력 round-trip.
# ─────────────────────────────────────────────────────────────────────


def _parse_yaml(text: str) -> dict:
    raw_lines = text.splitlines()
    # 주석/빈 줄 제거 안함 — 라인 인덱스 유지를 위해. parser 가 skip.
    parsed, _end = _parse_block(raw_lines, 0, 0)
    return parsed


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_skip_line(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#")


def _parse_block(lines: list[str], start: int, indent_level: int) -> tuple[dict, int]:
    """블록 파싱. indent_level 은 "들여쓰기 단계 수" (한 단계 = 2 spaces).

    반환: (dict, 다음 처리할 줄 인덱스)
    """
    target_indent = indent_level * 2
    result: dict[str, Any] = {}
    i = start
    while i < len(lines):
        line = lines[i]
        if _is_skip_line(line):
            i += 1
            continue
        leading = _line_indent(line)
        if leading < target_indent:
            break
        if leading > target_indent:
            # 부모가 처리해야 했던 줄 — fallback: skip
            i += 1
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            # 이 위치에서 list 시작은 dict 컨텍스트 위반 — 종료
            break
        # key: value 패턴
        if ":" not in stripped:
            i += 1
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "" or val == "{}" or val == "[]":
            if val == "{}":
                result[key] = {}
                i += 1
                continue
            if val == "[]":
                result[key] = []
                i += 1
                continue
            # 빈 값 → 다음 줄로 nested 또는 list 추정
            next_idx = _next_nonskip(lines, i + 1)
            if next_idx is None:
                result[key] = None
                i += 1
                continue
            nxt = lines[next_idx]
            nxt_lead = _line_indent(nxt)
            if nxt_lead <= target_indent:
                result[key] = None
                i += 1
                continue
            if nxt.lstrip().startswith("- "):
                lst, end = _parse_list(lines, next_idx, indent_level + 1)
                result[key] = lst
                i = end
            else:
                sub, end = _parse_block(lines, next_idx, indent_level + 1)
                result[key] = sub
                i = end
        else:
            result[key] = _parse_scalar(val)
            i += 1
    return result, i


def _parse_list(lines: list[str], start: int, indent_level: int) -> tuple[list, int]:
    """list 파싱. 각 줄은 '- ...' 형태.

    list 내부 dict 는 inline 한 줄 또는 ' - key:\n   key2: ...' 다중 줄.
    """
    target_indent = indent_level * 2
    result: list[Any] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if _is_skip_line(line):
            i += 1
            continue
        leading = _line_indent(line)
        if leading < target_indent:
            break
        if leading > target_indent:
            i += 1
            continue
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        item_str = stripped[2:].strip()
        # dict-in-list 한 줄 inline (예: "- name: foo")
        if (
            item_str
            and ":" in item_str
            and not (item_str.startswith('"') or item_str.startswith("'"))
            and not item_str.startswith("{")
        ):
            d_item, end = _parse_inline_dict_then_continue(
                lines, i, indent_level, item_str,
            )
            result.append(d_item)
            i = end
            continue
        # scalar
        result.append(_parse_scalar(item_str))
        i += 1
    return result, i


def _parse_inline_dict_then_continue(
    lines: list[str], i: int, indent_level: int, first_item_str: str,
) -> tuple[dict, int]:
    """list 항목이 '- key: val' 로 시작하면 dict.

    같은 indent 의 후속 'key: val' 줄도 같은 dict 에 합친다.
    indent_level 은 '- ' 위치의 들여쓰기 단계.
    """
    d: dict[str, Any] = {}
    key, _, val = first_item_str.partition(":")
    d[key.strip()] = _parse_scalar(val.strip()) if val.strip() else None
    target_inner_indent = indent_level * 2 + 2  # '- ' 가 차지한 2칸 만큼 더
    j = i + 1
    while j < len(lines):
        line = lines[j]
        if _is_skip_line(line):
            j += 1
            continue
        leading = _line_indent(line)
        stripped = line.strip()
        if leading < target_inner_indent:
            break
        if stripped.startswith("- "):
            break
        if ":" in stripped:
            k2, _, v2 = stripped.partition(":")
            d[k2.strip()] = _parse_scalar(v2.strip()) if v2.strip() else None
            j += 1
        else:
            j += 1
    return d, j


def _next_nonskip(lines: list[str], start: int) -> int | None:
    j = start
    while j < len(lines):
        if not _is_skip_line(lines[j]):
            return j
        j += 1
    return None


def _parse_scalar(s: str) -> Any:
    """단일 scalar 문자열 → Python 값.

    null/None, bool, int, float, quoted string, inline flow dict/list, plain
    string 순으로 시도. v1.15: inline `{k: v}` / `[a, b]` 도 정상 파싱.
    """
    s = s.strip()
    if not s:
        return None
    if s in ("null", "~", "Null", "NULL"):
        return None
    if s in ("true", "True", "TRUE", "yes", "Yes"):
        return True
    if s in ("false", "False", "FALSE", "no", "No"):
        return False
    # quoted string — JSON 디코드 시도
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s[1:-1]
    # inline flow style — YAML `{k: v}` 또는 `[a, b]`. v1.15 dict-style metrics 지원.
    if s.startswith("{") and s.endswith("}"):
        return _parse_inline_flow_dict(s)
    if s.startswith("[") and s.endswith("]"):
        return _parse_inline_flow_list(s)
    # number
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        pass
    return s


def _split_flow_items(body: str) -> list[str]:
    """flow-style inner body 를 top-level `,` 로 분리. nested {} / [] 보호."""
    items: list[str] = []
    depth = 0
    in_quote: str | None = None
    cur = []
    for ch in body:
        if in_quote:
            cur.append(ch)
            if ch == in_quote and (not cur[-2:] == ["\\", in_quote]):
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            cur.append(ch)
            continue
        if ch in ("{", "["):
            depth += 1
            cur.append(ch)
            continue
        if ch in ("}", "]"):
            depth -= 1
            cur.append(ch)
            continue
        if ch == "," and depth == 0:
            items.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        items.append("".join(cur).strip())
    return [it for it in items if it]


def _parse_inline_flow_dict(s: str) -> dict:
    """`{key: val, key2: val2}` → dict. nested flow 도 재귀 파싱."""
    body = s[1:-1].strip()
    if not body:
        return {}
    out: dict[str, Any] = {}
    for item in _split_flow_items(body):
        if ":" not in item:
            continue
        k, _, v = item.partition(":")
        out[k.strip()] = _parse_scalar(v.strip())
    return out


def _parse_inline_flow_list(s: str) -> list:
    """`[a, b, c]` → list. nested flow 도 재귀 파싱."""
    body = s[1:-1].strip()
    if not body:
        return []
    return [_parse_scalar(item) for item in _split_flow_items(body)]
