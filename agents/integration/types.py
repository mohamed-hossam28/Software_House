"""
agents/integration/types.py
─────────────────────────────
Type-consistency checking helpers extracted from agents/integration.py (Phase 1.4).

Used to verify contract field types against actual annotations in generated code.
"""

import re
from typing import Optional

from agents.integration.constants import _PRIMITIVE_TYPE_SETS


def normalize_api_path(path: str) -> str:
    # Path-typed params ({file_path:path}) are catch-alls — drop the param
    # entirely so `/static/{file_path:path}` normalizes to `/static`.
    path = re.sub(r'\{[^}]*:\w+\}', '', path)
    path = re.sub(r'\{([^}]+)\}', r':\1', path)
    # Parameter NAMES are irrelevant to matching — `{id}` == `{todo_id}`.
    path = re.sub(r':\w+', ':', path)
    return path.strip().rstrip('/')


def strip_leading_api(path: str) -> str:
    """Remove a conventional leading `api/` segment, if present."""
    if path.startswith("/api/"):
        return path[4:]
    return path


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas that are not nested inside [] or ()."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current or not parts:
        parts.append("".join(current).strip())
    return parts


def _type_sets(annotation: str) -> frozenset[str]:
    """Normalize a type annotation/contract type into a set of primitive tokens.
    Handles Optional[x], Union[a, b], a | b, List[x], typing. prefixes and
    JS-style 'string|null'. Unknown names (model classes etc.) are kept as-is
    and compared literally."""
    token = (annotation or "").strip().lower().replace(" ", "")
    token = re.sub(r"\btyping\.", "", token)

    def expand(s: str) -> set[str]:
        s = s.strip()
        if not s:
            return set()
        if s.startswith("optional") and s.endswith("]"):
            return expand(s[len("optional["):-1]) | {"none"}
        if s.startswith("union") and s.endswith("]"):
            out: set[str] = set()
            for part in _split_top_level_commas(s[len("union["):-1]):
                out |= expand(part)
            return out
        if "|" in s:
            out = set()
            for part in s.split("|"):
                out |= expand(part)
            return out
        if s.startswith("list") and s.endswith("]"):
            return expand(s[len("list["):-1]) | {"arr"}
        if s.startswith("dict") and s.endswith("]"):
            return expand(s[4:s.rfind("[")]) | {"obj"}
        if s in _PRIMITIVE_TYPE_SETS:
            return set(_PRIMITIVE_TYPE_SETS[s])
        return {s}

    return frozenset(expand(token))


def _capture_annotation(code: str, field: str) -> Optional[str]:
    """Capture a `field: <annotation>` token, balancing [] so Union/Optional
    generics (which contain commas) are not truncated at depth 0."""
    m = re.search(r"(?<![\w-])\b" + re.escape(field) + r"\b\s*\??\s*:\s*", code)
    if not m:
        return None
    start = m.end()
    i = start
    depth = 0
    while i < len(code):
        ch = code[i]
        if depth == 0 and ch in ",=)#" or (depth == 0 and ch == ":" and i > start):
            break
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        i += 1
    return code[start:i].strip() or None


def check_type_consistency(field: str, expected_type: str, code: str) -> Optional[str]:
    """Check a contract field's declared type against its annotation in code.
    Comparisons are set-based with nullable tolerance: `Optional[str]`,
    `Union[str, None]`, `str | None` all satisfy a contract of `string|null`;
    a nullable actual also satisfies a non-null contract."""
    actual_type = _capture_annotation(code, field)
    if actual_type is None:
        if re.search(r"(?<![\w-])\b" + re.escape(field) + r"\b", code):
            return None
        return f"Field '{field}' is missing entirely."

    exp_set = _type_sets(expected_type)
    act_set = _type_sets(actual_type)
    if not exp_set or not act_set:
        if expected_type.lower().strip() == actual_type.lower().strip():
            return None
        return f"Type mismatch for '{field}': expected '{expected_type}', found '{actual_type}'"

    if exp_set == act_set:
        return None
    if exp_set - {"none"} == act_set - {"none"} and ({"none"} & exp_set or {"none"} & act_set):
        return None
    return f"Type mismatch for '{field}': expected '{expected_type}', found '{actual_type}'"
