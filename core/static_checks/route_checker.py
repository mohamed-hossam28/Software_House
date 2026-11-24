"""
Backend route extraction and the two-way HTTP contract check.

Extraction — FastAPI route decorators (@app.get("/x"), @router.post("/y/{id}"),
@router.api_route(..., methods=[...]), @router.add_api_route(...)) are
AST-extracted from backend Python files as method + normalized path shape.
A `name = APIRouter(prefix="...")` assignment (module level or in the same
scope) prepends its prefix to paths registered through that router. Decorators
whose path is not a string literal are skipped — dynamic registration cannot
be judged statically — and files that do not parse are skipped, since the
Python checker already reports their syntax errors.

Two-way check — a contract the backend never implements (no extracted route
matches its method and path shape) is an ERROR finding. Each finding is
attributed to the manifest-planned backend file that should own routes so the
repair loop can regenerate or extend that file; when no backend route file is
planned, the finding stays file-less and fails the run honestly (file-less
errors cannot be repaired). A route the backend implements without a contract
is not flagged: the backend is the source of truth for what exists, and the
contracts guide the frontend.

Known limitations (dynamic registration cannot be judged statically):
`app.include_router(router, prefix=...)` adds no prefix, and routers mounted
inside routers collect none of their parents' prefixes — routes registered
through either are indexed under their bare decorator path, which can only
produce a false-positive warn (never a missed error or missed frontend match).
"""
import ast
from typing import Any

from core.state import FindingSeverity, StaticFinding
from core.static_checks.http_paths import (
    HTTPIndex,
    PathShape,
    contract_entries,
    normalize_path,
    path_shape,
    shapes_match,
)

_ROUTE_DECORATORS = {"get", "post", "put", "delete", "patch", "head", "options"}
_METHOD_LIST_DECORATORS = {"api_route", "add_api_route"}


def check_backend_routes(
    files: dict[str, str],
    manifest: Any,
    architecture: Any = None,
) -> list[StaticFinding]:
    """Error for every API contract no extracted backend route implements.

    Findings carry the manifest-planned route file as their repair target;
    `file=""` only when the manifest plans no backend route file at all.
    """
    contracts = contract_entries(architecture)
    if not contracts:
        return []
    routes = extract_backend_routes(files)
    target = _route_file_target(manifest)
    findings: list[StaticFinding] = []
    for method, path, shape in contracts:
        if _implemented(method, shape, routes):
            continue
        findings.append(StaticFinding(
            file=target,
            check="backend_routes",
            severity=FindingSeverity.ERROR,
            line=0,
            message=(
                f"Backend never implements contract {method} {path} "
                f"(no FastAPI route matches method and path)"
            ),
        ))
    return findings


def _route_file_target(manifest: Any) -> str:
    """Manifest path of the backend file planned to own FastAPI routes; "" when none.

    Priority: a backend .py whose planner purpose mentions a route/endpoint
    role, then one whose path contains a route/api segment, then a main/app
    entry file. First sorted match wins, so the target is deterministic.
    """
    if manifest is None:
        return ""
    entries = getattr(manifest, "files", {}) or {}
    purpose_hit = path_hit = entry_hit = ""
    for filepath in sorted(entries):
        if not filepath.endswith(".py"):
            continue
        fm = entries[filepath]
        if str(getattr(fm, "agent", "") or "").lower() != "backend":
            continue
        purpose = str(getattr(fm, "purpose", "") or "").lower()
        norm = filepath.replace("\\", "/").lower()
        name = norm.rsplit("/", 1)[-1]
        if not purpose_hit and any(w in purpose for w in ("route", "endpoint")):
            purpose_hit = filepath
        if not path_hit and any(seg in norm for seg in ("/route", "/api")):
            path_hit = filepath
        if not entry_hit and name in ("main.py", "app.py"):
            entry_hit = filepath
    return purpose_hit or path_hit or entry_hit or ""


def _implemented(method: str, shape: PathShape, routes: HTTPIndex) -> bool:
    """True when a route implements *method* on *shape* (parameter slots match any value).

    Slot matching, not exact equality: a `/users/{id}` route serves a `/users/me`
    contract in FastAPI, so it implements it.
    """
    return any(
        method in methods
        for route_shape, methods in routes.items()
        if shapes_match(shape, route_shape)
    )


# ── Route extraction ─────────────────────────────────────────────────────────


def extract_backend_routes(files: dict[str, str]) -> HTTPIndex:
    """FastAPI routes in *files* as path shape → HTTP methods.

    Covers every .py file passed in; frontend/other files are ignored.
    """
    routes: HTTPIndex = {}
    for filepath, code in sorted(files.items()):
        if not filepath.endswith(".py") or not code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue                                # python_checker already reports this
        _collect_routes(tree, routes)
    return routes


def _collect_routes(tree: ast.AST, routes: HTTPIndex) -> None:
    """Merge every FastAPI route registered in *tree* into *routes*."""
    prefixes: dict[str, str] = {}
    for node in ast.walk(tree):
        prefix = _router_prefix(node)
        if prefix is not None:
            prefixes.update(prefix)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            for method, raw_path in _decorator_routes(decorator, prefixes):
                shape = path_shape(normalize_path(raw_path))
                routes[shape] = routes.get(shape, frozenset()) | {method}


def _router_prefix(node: ast.AST) -> dict[str, str] | None:
    """{name: prefix} for `name = APIRouter(prefix="...")` assignments; else None."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    value = node.value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)):
        return None
    if value.func.id != "APIRouter":
        return None
    prefix = _string_keyword(value, "prefix")
    if prefix is None:
        return None
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = [t.id for t in targets if isinstance(t, ast.Name)]
    return {name: prefix for name in names} if names else None


def _decorator_routes(
    decorator: ast.AST,
    prefixes: dict[str, str],
) -> list[tuple[str, str]]:
    """(METHOD, path) pairs one route decorator registers; [] when not a route."""
    if not isinstance(decorator, ast.Call):
        return []
    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return []
    method_names = (
        {func.attr.upper()} if func.attr in _ROUTE_DECORATORS else set()
    ) | (
        {"GET"} if func.attr in _METHOD_LIST_DECORATORS else set()
    )
    if not method_names:
        return []
    path = _path_argument(decorator)
    if path is None:
        return []
    if func.attr in _METHOD_LIST_DECORATORS:
        declared = _string_list_keyword(decorator, "methods")
        if declared:
            method_names = {m.upper() for m in declared}
        elif _keyword_present(decorator, "methods"):
            return []                              # dynamic methods list — not judgeable
        # else: FastAPI's default of GET applies
    router_name = getattr(func.value, "id", None)
    prefix = prefixes.get(router_name, "")
    return [(method, prefix + path) for method in sorted(method_names)]


def _path_argument(call: ast.Call) -> str | None:
    """Route path of a decorator call: first positional or path= keyword string."""
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None                                # dynamic path — not judgeable
    return _string_keyword(call, "path")


def _string_keyword(call: ast.Call, keyword: str) -> str | None:
    """String value of keyword *keyword* on *call*; None when absent or dynamic."""
    for kw in call.keywords:
        if kw.arg == keyword and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                return kw.value.value
    return None


def _string_list_keyword(call: ast.Call, keyword: str) -> list[str] | None:
    """List of string values passed as keyword *keyword*; None when absent/dynamic."""
    for kw in call.keywords:
        if kw.arg != keyword:
            continue
        value = kw.value
        if isinstance(value, (ast.List, ast.Tuple)):
            items = [
                element.value for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if len(items) == len(value.elts):
                return items
    return None


def _keyword_present(call: ast.Call, keyword: str) -> bool:
    """True when keyword *keyword* is passed on *call*, whatever its value."""
    return any(kw.arg == keyword for kw in call.keywords)
