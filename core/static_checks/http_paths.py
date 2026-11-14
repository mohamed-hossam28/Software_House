"""
Shared HTTP path utilities for the static-check layer.

Path normalization and shape comparison used by both the JavaScript
frontend checker (URL candidates from fetch/axios calls) and the backend
route extractor (FastAPI route decorators), so both sides compare paths the
same way: scheme/host and query strings stripped, trailing slashes removed,
and `{id}` / `:id` parameter segments reduced to one shared placeholder slot.
"""
import re
from typing import Any

#: Parameter slot in a normalized path shape ("{id}", ":id", "{}" → "{}").
PATH_SLOT = "{}"
_ABSOLUTE_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://[^/]*")

#: PathShape — path reduced to comparable segments, parameters as PATH_SLOT.
PathShape = tuple[str, ...]
#: HTTPIndex — path shape → the HTTP methods defined on it.
HTTPIndex = dict[PathShape, frozenset[str]]


def normalize_path(raw: str) -> str | None:
    """Path portion of *raw*: scheme/host stripped, query and trailing slash removed."""
    path = _ABSOLUTE_URL_RE.sub("", raw or "")
    path = path.split("?", 1)[0]
    if not path.startswith("/"):
        start = path.find("/")
        if start < 0:
            return None
        path = path[start:]
    path = path.rstrip("/") or "/"
    return path


def path_shape(path: str) -> PathShape:
    """Path reduced to comparable segments; parameters become a shared placeholder."""
    segments: list[str] = []
    for segment in path.split("/"):
        if not segment:
            continue
        if (
            (segment.startswith("{") and segment.endswith("}"))
            or segment.startswith(":")
            or segment == "{P}"
        ):
            segments.append(PATH_SLOT)
        else:
            segments.append(segment.lower())
    return tuple(segments)


def shapes_match(call: PathShape, known: PathShape) -> bool:
    """True when the two path shapes align, with parameter slots matching any value."""
    return len(call) == len(known) and all(
        a == b or PATH_SLOT in (a, b) for a, b in zip(call, known)
    )


def contract_index(architecture: Any) -> HTTPIndex:
    """The Architecture Agent's API contracts as path shape → HTTP methods.

    Empty when the architecture is absent or defines no contracts.
    """
    contracts = getattr(architecture, "api_contracts", None) or []
    index: HTTPIndex = {}
    for contract in contracts:
        method = str(getattr(contract, "method", "") or "").upper()
        path = str(getattr(contract, "path", "") or "")
        if not method or not path:
            continue
        shape = path_shape(normalize_path(path))
        index[shape] = index.get(shape, frozenset()) | {method}
    return index


def contract_entries(architecture: Any) -> list[tuple[str, str, PathShape]]:
    """(method, normalized path, path shape) per API contract, message-ready."""
    entries: list[tuple[str, str, PathShape]] = []
    for contract in getattr(architecture, "api_contracts", None) or []:
        method = str(getattr(contract, "method", "") or "").upper()
        raw_path = str(getattr(contract, "path", "") or "")
        path = normalize_path(raw_path)
        if method and path:
            entries.append((method, path, path_shape(path)))
    return entries
