"""
agents/integration/checks.py
───────────────────────────
Phase A cross-layer semantic checks for the Integration Agent — standalone,
stateless functions over ProjectState. Static parsing only; generated code is
never executed.

Checks:
  check_startup_assets             A1  — StaticFiles/FileResponse paths resolve
  check_model_schema_consistency   A2  — ORM columns / schema ctor kwargs exist
  check_layer_wiring               A3  — schemas/ single-source, no dead modules
  check_frontend_backend_contracts A4  — fetch() calls match real backend routes
  check_frontend_dom               A5  — DOM ids/classes/asset wiring
  check_file_set                   A6  — required files planned/written/on disk
  check_file_corruption            A7  — corruption detector over generated code
  check_duplicate_foundation       A8  — engine/Base/SessionLocal defined once
  check_router_prefixes            A9  — no doubled include_router+APIRouter prefix
  check_contract_coverage          A10 — API contracts implemented & covered
"""

import logging
import os
import re
from typing import Any, Optional

from core.state import FileManifest, ProjectState

from agents.utils import detect_corruption, normalize_output_path

from agents.integration.constants import (
    OUTPUT_DIR,
    _FOUNDATION_SYMBOLS,
)
from agents.integration.parsers import (
    parse_backend_routes,
    parse_css_classes,
    parse_frontend_fetches,
    parse_html_classes,
    parse_html_ids,
    parse_js_classes,
    parse_js_response_field_usage,
    parse_js_used_ids,
    parse_pydantic_schemas,
    parse_python_imports,
    parse_sqlalchemy_models,
)
from agents.integration.resolvers import (
    build_internal_prefixes,
    layer_root_for,
    match_backend_route,
    resolve_import_path,
)
from agents.integration.types import (
    check_type_consistency,
    normalize_api_path,
    strip_leading_api,
)

_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def _resolve_css_target(
    cls: str,
    current_js: str,
    css_paths: list[str],
    js_paths: list[str],
    html_paths: list[str],
    html_code: dict[str, str],
) -> str:
    """Attribute an undefined CSS class to the file best placed to define it:
    a CSS file when one exists; else the JS file using it when the class also
    appears in HTML (inline classList usage); else the first JS or HTML file.
    (Attributing to app.js/index.html blindly just reworked the wrong file.)"""
    if css_paths:
        return css_paths[0]
    if any(cls in (html_code.get(fp, "") or "") for fp in html_code):
        return current_js
    return js_paths[0] if js_paths else html_paths[0]

def _module_path_for(src: str) -> str:
    """Layer-root-relative dotted Python module for a manifest path:
    'output/backend/db/models.py' → 'db.models'; 'output/backend/db/__init__.py' → 'db'."""
    root = layer_root_for(src)
    rel = src[len(root):] if root and src.startswith(root) else src.lstrip("/").lstrip("\\")
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")].rstrip("/")
    return rel.replace("/", ".").replace("\\", ".")


def get_file_code(filepath: str, fm: Optional[FileManifest], state: ProjectState) -> str:
    bucket = getattr(state.generated_code, fm.agent, {}) if fm else {}
    code = bucket.get(filepath, "") if isinstance(bucket, dict) else ""

    if not code:
        abs_path = os.path.join(_root_dir, normalize_output_path(filepath))
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception:
                pass
    return code


def _set_generated_code(state: ProjectState, filepath: str, agent: str, code: str) -> None:
    """Store generated code under state.generated_code[<agent>][filepath] when the bucket exists."""
    bucket = getattr(state.generated_code, agent, None)
    if isinstance(bucket, dict):
        bucket[filepath] = code


def collect_backend_routes(state: ProjectState) -> tuple[list[dict], set[str]]:
    """
    Ground-truth backend route list. When routers are mounted under an api
    prefix (app.include_router(..., prefix=...)), only the PREFIXED routes
    are actually served — raw router paths are excluded so frontend calls
    that miss the prefix get flagged instead of silently matching.
    Returns (served_routes, api_prefixes).
    """
    api_prefixes: set[str] = set()
    for path in state.generated_code.backend:
        code = get_file_code(path, state.code_manifest.files.get(path), state)
        if not code:
            continue
        for m in re.finditer(r"include_router\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
            api_prefixes.add(m.group(1).rstrip("/"))

    raw_routes: list[dict] = []
    for path in state.generated_code.backend:
        code = get_file_code(path, state.code_manifest.files.get(path), state)
        if not code:
            continue
        static = ("FileResponse" in code or "HTMLResponse" in code or "StaticFiles" in code)
        # Only ROUTER files (those defining an APIRouter) are mounted under
        # api prefixes via include_router — app-level routes (app.get, ...)
        # are not prefixed.
        prefixed = "APIRouter" in code
        for r in parse_backend_routes(code):
            raw_routes.append({**r, "static": static, "prefixed": prefixed, "filepath": path})
        # StaticFiles mounts serve GET (html=True → the dir index too). They
        # are recorded so HTML-serving contracts (e.g. `GET /`) are satisfied
        # by the mount instead of flagged as missing.
        for m in re.finditer(
            r"\.mount\s*\(\s*['\"]([^'\"]*)['\"]\s*,\s*StaticFiles\s*\(",
            code,
        ):
            mount_path = m.group(1).rstrip("/") or "/"
            mount_html = bool(re.search(r"html\s*=\s*True", code[m.end():m.end() + 120]))
            raw_routes.append({
                "method": "GET",
                "path": mount_path,
                "request_model": "",
                "response_model": "",
                "static": True,
                "prefixed": False,
                "filepath": path,
                "is_mount": True,
                "mount_html": mount_html,
            })

    if api_prefixes:
        routes: list[dict] = []
        for r in raw_routes:
            if r.get("prefixed"):
                for ap in api_prefixes:
                    routes.append({**r, "path": ap + r["path"]})
            else:
                routes.append(r)
    else:
        routes = raw_routes
    return routes, api_prefixes


def collect_schema_fields(state: ProjectState) -> dict[str, set[str]]:
    """
    {ClassName -> {fields}} across all backend files. When a class name is
    defined in both a schemas/ module and a router, the schemas/ definition
    wins (single-source rule, Phase C).
    """
    by_name: dict[str, tuple[str, set[str]]] = {}
    for path in state.generated_code.backend:
        code = get_file_code(path, state.code_manifest.files.get(path), state)
        if not code:
            continue
        for name, fields in parse_pydantic_schemas(code).items():
            cur = by_name.get(name)
            is_schema_file = "/schemas/" in path
            if cur is None or (is_schema_file and "/schemas/" not in cur[0]):
                by_name[name] = (path, fields)
    return {name: fields for name, (_, fields) in by_name.items()}


def check_startup_assets(state: ProjectState) -> list[dict]:
    """
    A1 (STARTUP) — StaticFiles(directory=...)/FileResponse(...) paths must
    resolve from the layer root the app is run from. Catches static
    directories that point at the wrong base folder.
    """
    conflicts: list[dict] = []
    if not any("frontend" in p for p in state.code_manifest.files):
        return conflicts
    all_written = all(
        fm.written for p, fm in state.code_manifest.files.items() if "frontend" in p
    )
    if not all_written:
        return conflicts  # disk layout not final yet — avoid false positives

    for path, fm in state.code_manifest.files.items():
        if not fm.written or not path.endswith(".py"):
            continue
        root = layer_root_for(path)
        if not root:
            continue
        code = get_file_code(path, fm, state)
        if not code:
            continue
        refs: list[str] = []
        refs += [m.group(1) for m in re.finditer(
            r"StaticFiles\s*\(\s*directory\s*=\s*['\"]([^'\"]+)['\"]", code)]
        refs += [m.group(1) for m in re.finditer(
            r"FileResponse\s*\(\s*['\"]([^'\"]+)['\"]", code)]
        refs += [m.group(1) for m in re.finditer(
            r"HTMLResponse\s*\(\s*(?:content\s*=\s*)?open\s*\(\s*['\"]([^'\"]+)['\"]", code)]
        for ref in refs:
            layer_abs = os.path.join(_root_dir, root.strip("/"))
            cand = os.path.normpath(os.path.join(layer_abs, ref))
            if os.path.exists(cand):
                continue
            alt = os.path.normpath(os.path.join(_root_dir, OUTPUT_DIR, ref.lstrip("/")))
            location = f"'{alt}'" if os.path.exists(alt) else "(nowhere in the workspace)"
            conflicts.append({
                "type": "STARTUP",
                "filepath": path,
                "task_id": fm.task_id,
                "expected": f"Asset path '{ref}' exists relative to the '{root}' layer root",
                "actual": "missing",
                "reason": (
                    f"'{root}{ref.lstrip('/')}' does not exist on disk. Assets appear to "
                    f"live at {location}. Use a path relative to the layer root "
                    f"(e.g. '../frontend/static')."
                ),
            })
    return conflicts


def check_model_schema_consistency(    state: ProjectState,
    schema_fields: Optional[dict[str, set[str]]] = None,
) -> list[dict]:
    """
    A2 (SCHEMA_COLUMN) — every attribute read off an ORM model and every
    constructor kwarg of a model/schema must exist on it. Catches
    `TodoItem.user_id` / `TodoResponse(completed=...)` style drift.
    """
    conflicts: list[dict] = []
    models: dict[str, set[str]] = {}
    schemas: dict[str, set[str]] = {}
    for path in state.generated_code.backend:
        code = get_file_code(path, state.code_manifest.files.get(path), state)
        if not code:
            continue
        models.update(parse_sqlalchemy_models(code))
        schemas.update(parse_pydantic_schemas(code))
    # schemas/ definitions win over router-local duplicates (single source).
    collected_schemas = schema_fields if schema_fields is not None else collect_schema_fields(state)
    for name, fields in collected_schemas.items():
        schemas[name] = fields

    for path in state.generated_code.backend:
        fm = state.code_manifest.files.get(path)
        code = get_file_code(path, fm, state)
        if not code:
            continue
        for model, columns in models.items():
            for m in re.finditer(r"\b" + re.escape(model) + r"\s*\.\s*(\w+)", code):
                attr = m.group(1)
                if attr in columns or attr in {"query", "metadata", "select", "registry", "objects"}:
                    continue
                conflicts.append({
                    "type": "SCHEMA_COLUMN",
                    "filepath": path,
                    "task_id": fm.task_id if fm else "unknown",
                    "expected": f"Column '{attr}' on ORM model '{model}'",
                    "actual": "undefined column",
                    "reason": (
                        f"'{path}' accesses '{model}.{attr}' but the model only defines "
                        f"columns: {sorted(columns) or '(none)'}. Fix the attribute name "
                        f"or add the column to the model."
                    ),
                })
            for m in re.finditer(r"\b" + re.escape(model) + r"\s*\(([^)]*)\)", code):
                for kw in re.finditer(r"(\w+)\s*=", m.group(1)):
                    key = kw.group(1)
                    if key in columns:
                        continue
                    conflicts.append({
                        "type": "SCHEMA_COLUMN",
                        "filepath": path,
                        "task_id": fm.task_id if fm else "unknown",
                        "expected": f"Column '{key}' on ORM model '{model}'",
                        "actual": "undefined column",
                        "reason": (
                            f"'{path}' constructs '{model}' with field '{key}' but the "
                            f"model only defines columns: {sorted(columns) or '(none)'}."
                        ),
                    })
        for schema, fields in schemas.items():
            for m in re.finditer(r"\b" + re.escape(schema) + r"\s*\(([^)]*)\)", code):
                for kw in re.finditer(r"(\w+)\s*=", m.group(1)):
                    key = kw.group(1)
                    if key in fields or key in {"model_config"}:
                        continue
                    conflicts.append({
                        "type": "SCHEMA_COLUMN",
                        "filepath": path,
                        "task_id": fm.task_id if fm else "unknown",
                        "expected": f"Field '{key}' on schema '{schema}'",
                        "actual": "unknown field",
                        "reason": (
                            f"'{path}' constructs '{schema}' with field '{key}' but the "
                            f"schema only defines: {sorted(fields) or '(none)'}."
                        ),
                    })
    return conflicts


def check_layer_wiring(state: ProjectState) -> list[dict]:
    """
    A3 (WIRING) — structural integrity of the backend layer:
    no schema redefinitions next to a schemas/ module, no unimported modules
    when an entrypoint exists, and no inline data-access in routers when a
    services layer exists and is unused.
    """
    conflicts: list[dict] = []
    files = state.code_manifest.files
    backend_paths = [p for p in files if files[p].agent == "backend" and p.endswith(".py")]
    if not backend_paths:
        return conflicts

    manifest_files = list(files.keys())
    internal_prefixes = build_internal_prefixes(manifest_files)

    # 1. Schema duplication vs a shared schemas/ module.
    schema_names: dict[str, list[str]] = {}
    for p in backend_paths:
        code = get_file_code(p, files.get(p), state)
        if not code:
            continue
        for name in parse_pydantic_schemas(code):
            schema_names.setdefault(name, []).append(p)
    for name, owners in schema_names.items():
        schema_file = next((p for p in owners if "/schemas/" in p), None)
        if not schema_file:
            continue
        for p in owners:
            if "/schemas/" in p:
                continue
            code = get_file_code(p, files.get(p), state)
            if re.search(r"from\s+[\w.]*schemas[\w.]*\s+import", code or ""):
                continue
            conflicts.append({
                "type": "WIRING",
                "filepath": p,
                "task_id": files[p].task_id,
                "expected": f"Schema '{name}' imported from the shared schemas module",
                "actual": "redefined locally",
                "reason": (
                    f"'{p}' redefines schema class '{name}' which already exists in "
                    f"'{schema_file}'. Define it once in the schemas module and import "
                    f"it here — never redefine request/response models in routers."
                ),
            })

    # 2. Dead modules — only when the layer has an entrypoint to run.
    entrypoints = {p for p in backend_paths if os.path.basename(p) in ("main.py", "app.py", "server.py")}
    has_entrypoint = bool(entrypoints)
    imported_paths: set[str] = set()
    if has_entrypoint:
        for p in backend_paths:
            code = get_file_code(p, files.get(p), state)
            if not code:
                continue
            for imp in parse_python_imports(code):
                resolved = resolve_import_path(imp, p, manifest_files, internal_prefixes)
                if resolved:
                    imported_paths.add(resolved)
        for p in backend_paths:
            if p in entrypoints or p in imported_paths:
                continue
            if os.path.basename(p) == "__init__.py":
                continue
            # Attribute to the entrypoint: it is the only file whose rework
            # can add the missing import (reworking the dead module itself
            # can never make it imported).
            target = next(iter(sorted(entrypoints))) if entrypoints else p
            conflicts.append({
                "type": "WIRING",
                "filepath": target,
                "task_id": files[target].task_id if target in files else files[p].task_id,
                "expected": f"Module '{p}' is imported by at least one other backend file",
                "actual": "never imported",
                "reason": (
                    f"'{p}' is not imported by any other backend file — it is dead "
                    f"code. Either wire it into the entrypoint/router or remove it."
                ),
            })

    # 3. Inline data-access in routers while an unused services layer exists.
    service_paths = [p for p in backend_paths if "/services/" in p]
    if service_paths and (not has_entrypoint or not any(p in imported_paths for p in service_paths)):
        business = re.compile(
            r"\.commit\(|\.rollback\(|db\.query|SessionLocal|create_engine|sessionmaker|"
            r"mock_db|\.append\(|\.pop\(|\.save\(|\.delete\(|\.filter\("
        )
        for p in backend_paths:
            if not any(k in p.lower() for k in ("/routers/", "/routes/", "controller")):
                continue
            code = get_file_code(p, files.get(p), state)
            if not code or ("@router." not in code and "@app." not in code):
                continue
            if business.search(code):
                conflicts.append({
                    "type": "WIRING",
                    "filepath": p,
                    "task_id": files[p].task_id,
                    "expected": "Routers delegate data access to the services layer",
                    "actual": "inline data access in router",
                    "reason": (
                        f"'{p}' performs data-access/business logic inline while a "
                        f"services layer exists ({', '.join(service_paths)}) that is "
                        f"never imported. Routers must be thin: import and call the "
                        f"service functions."
                    ),
                })
    return conflicts


def check_router_prefixes(state: ProjectState) -> list[dict]:
    """
    A9 (WIRING) — a router mounted under app.include_router(prefix=...) that
    ALSO declares its own APIRouter(prefix=...) produces doubled served
    paths (e.g. '/api' + '/api/todos' → '/api/api/todos'). Flag the router
    file so rework removes one of the two prefixes; contracts can then be
    satisfied at the canonical path.
    """
    conflicts: list[dict] = []
    files = state.code_manifest.files
    include_prefixes: set[str] = set()
    for p in files:
        if files[p].agent != "backend" or not p.endswith(".py"):
            continue
        code = get_file_code(p, files.get(p), state)
        if not code:
            continue
        for m in re.finditer(r"include_router\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
            include_prefixes.add(m.group(1).rstrip("/"))
    if not include_prefixes:
        return conflicts

    for p in files:
        fm = files[p]
        if fm.agent != "backend" or not p.endswith(".py"):
            continue
        code = get_file_code(p, fm, state)
        if not code or "APIRouter" not in code:
            continue
        for m in re.finditer(r"APIRouter\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
            rp = m.group(1).rstrip("/")
            for ip in include_prefixes:
                if rp != ip and not rp.startswith(ip + "/"):
                    continue
                conflicts.append({
                    "type": "WIRING",
                    "filepath": p,
                    "task_id": fm.task_id,
                    "expected": f"Router mounted under include_router(prefix='{ip}') without its own prefix",
                    "actual": f"duplicate prefix: APIRouter(prefix='{rp}')",
                    "reason": (
                        f"'{p}' declares APIRouter(prefix='{rp}') while the layer mounts it via "
                        f"include_router(prefix='{ip}') — every route is served under the doubled "
                        f"path '{ip}{rp}'. Remove the router's own prefix (keep bare paths like "
                        f"'/todos') and let the include_router prefix supply the API root."
                    ),
                })
                break
    return conflicts


def check_duplicate_foundation(state: ProjectState) -> list[dict]:
    """
    Phase 3 (WIRING) — two+ backend files defining the same DB foundation
    symbol (engine / Base / SessionLocal) produce duplicate connections and
    two declarative bases; the import graph can never converge. Flag the
    non-canonical file so the owning agent removes the duplicate definition
    and imports the symbol from the canonical module instead.
    """
    conflicts: list[dict] = []
    files = state.code_manifest.files
    owners: dict[str, list[str]] = {s: [] for s in _FOUNDATION_SYMBOLS}
    for p in files:
        fm = files[p]
        if fm.agent != "backend" or not p.endswith(".py"):
            continue
        code = get_file_code(p, fm, state)
        if not code:
            continue
        for sym in _FOUNDATION_SYMBOLS:
            if re.search(r"^" + sym + r"\s*=", code, re.MULTILINE):
                owners[sym].append(p)

    for sym, paths in owners.items():
        if len(paths) < 2:
            continue
        canonical = canonical_foundation_file(state, paths)
        mod = _module_path_for(canonical)
        for p in paths:
            if p == canonical:
                continue
            conflicts.append({
                "type": "WIRING",
                "filepath": p,
                "task_id": files[p].task_id,
                "expected": f"'{sym}' defined once (in '{canonical}')",
                "actual": "duplicate foundation definition",
                "reason": (
                    f"'{p}' redefines '{sym}' which is already defined in '{canonical}'. "
                    f"DB foundation modules must define engine/Base/SessionLocal ONCE — "
                    f"remove the duplicate definition here and import it: "
                    f"'from {mod} import {sym}'."
                ),
            })
    return conflicts


def canonical_foundation_file(state: ProjectState, paths: list[str]) -> str:
    """
    Deterministic pick of the single file that owns DB foundation symbols:
    a dedicated foundation module name first (session.py / database.py /
    db.py / base.py / init_db.py), then the file defining the most
    foundation symbols, then the lexicographically smallest path.
    """
    dedicated = {"session.py", "database.py", "db.py", "base.py", "init_db.py"}
    for p in sorted(paths):
        if os.path.basename(p).lower() in dedicated:
            return p

    def count(p: str) -> int:
        code = get_file_code(p, state.code_manifest.files.get(p), state)
        return sum(
            1 for s in _FOUNDATION_SYMBOLS
            if code and re.search(r"^" + s + r"\s*=", code, re.MULTILINE)
        )

    return max(paths, key=lambda p: (count(p), -len(p), p))


def check_frontend_backend_contracts(    state: ProjectState,
    routes_ctx: Optional[tuple[list[dict], set[str]]] = None,
    schema_fields: Optional[dict[str, set[str]]] = None,
) -> list[dict]:
    """
    A4 (FRONTEND_HTTP) — frontend fetch calls must match the backend's ACTUAL
    routes: path (incl. /api prefix), HTTP verb, request-body fields and
    response fields read by the UI.
    """
    conflicts: list[dict] = []
    frontend_paths = [
        p for p, fm in state.code_manifest.files.items()
        if fm.agent == "frontend" and p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".html"))
    ]
    if not frontend_paths:
        return conflicts
    if routes_ctx is None or schema_fields is None:
        collected_routes, collected_prefixes = collect_backend_routes(state)
        if routes_ctx is None:
            routes_ctx = (collected_routes, collected_prefixes)
    schemas = schema_fields if schema_fields is not None else collect_schema_fields(state)
    routes, api_prefixes = routes_ctx

    for path in frontend_paths:
        fm = state.code_manifest.files.get(path)
        if not fm or not fm.written:
            continue
        code = get_file_code(path, fm, state)
        if not code or "fetch" not in code:
            continue
        for fetch in parse_frontend_fetches(code):
            route, close = match_backend_route(fetch["path"], fetch["method"], routes)
            prefix_issue = None
            if route is None and api_prefixes:
                # Missing /api prefix? e.g. calls '/todos' but served at '/api/todos'.
                # Fall through to body/static checks against the prefixed match.
                for ap in api_prefixes:
                    match, _ = match_backend_route(ap + fetch["path"], fetch["method"], routes)
                    if match:
                        route = match
                        prefix_issue = ap
                        break
            if route is None:
                close_str = "; ".join(f"{c['method']} {c['path']}" for c in close[:3])
                conflicts.append({
                    "type": "FRONTEND_HTTP",
                    "filepath": path,
                    "task_id": fm.task_id,
                    "expected": f"Backend route for '{fetch['method']} {fetch['path']}'",
                    "actual": "no matching route",
                    "reason": (
                        f"Frontend '{path}' calls '{fetch['method']} {fetch['path']}' but "
                        f"no backend route matches."
                        + (f" Closest backend routes: {close_str}." if close_str else "")
                    ),
                })
                continue

            if prefix_issue:
                conflicts.append({
                    "type": "FRONTEND_HTTP",
                    "filepath": path,
                    "task_id": fm.task_id,
                    "expected": f"Backend route for '{fetch['method']} {fetch['path']}'",
                    "actual": "missing api prefix",
                    "reason": (
                        f"Frontend '{path}' calls '{fetch['method']} {fetch['path']}' "
                        f"but the backend mounts its routers under '{prefix_issue}' — use "
                        f"'{prefix_issue}{fetch['path']}' instead."
                    ),
                })

            # Static/HTML endpoint parsed as JSON in the fetch's own handling?
            if route.get("static") and ".json()" in fetch.get("after", ""):
                conflicts.append({
                    "type": "FRONTEND_HTTP",
                    "filepath": path,
                    "task_id": fm.task_id,
                    "expected": f"JSON response from '{route['method']} {route['path']}'",
                    "actual": "endpoint returns HTML",
                    "reason": (
                        f"Frontend '{path}' parses the response of "
                        f"'{route['method']} {route['path']}' as JSON (.json()), but the "
                        f"backend endpoint serves static/HTML content."
                    ),
                })

            # Request-body fields vs the route's request model.
            if fetch["body_keys"] and route.get("request_model"):
                fields = schemas.get(route["request_model"])
                if fields:
                    for key in fetch["body_keys"]:
                        if key in fields:
                            continue
                        conflicts.append({
                            "type": "FRONTEND_HTTP",
                            "filepath": path,
                            "task_id": fm.task_id,
                            "expected": f"Request field '{key}' accepted by '{route['method']} {route['path']}'",
                            "actual": "unknown field",
                            "reason": (
                                f"Frontend sends body field '{key}' to "
                                f"'{route['method']} {route['path']}', but request model "
                                f"'{route['request_model']}' only accepts: {sorted(fields)}."
                            ),
                        })

            # Response fields the UI reads vs the route's response model.
            # Only checked when THIS fetch parses JSON — otherwise the used
            # fields come from an unrelated fetch and the flag is noise.
            if (
                route.get("response_model") and schemas.get(route["response_model"])
                and ".json()" in fetch.get("after", "")
            ):
                resp_fields = schemas[route["response_model"]]
                for var, used in parse_js_response_field_usage(code).items():
                    for f in used:
                        if f in resp_fields:
                            continue
                        conflicts.append({
                            "type": "FRONTEND_HTTP",
                            "filepath": path,
                            "task_id": fm.task_id,
                            "expected": f"Response field '{f}' on '{route['response_model']}'",
                            "actual": "unknown field",
                            "reason": (
                                f"Frontend reads field '{var}.{f}' from the response of "
                                f"'{route['method']} {route['path']}', but response model "
                                f"'{route['response_model']}' only has: {sorted(resp_fields)}."
                            ),
                        })
    return conflicts


def check_contract_coverage(    state: ProjectState,
    routes_ctx: Optional[tuple[list[dict], set[str]]] = None,
    schema_fields: Optional[dict[str, set[str]]] = None,
) -> list[dict]:
    """
    A8 (CONTRACT_COVERAGE) — parse-based replacement for the legacy substring
    `_check_api_contracts`. Every architecture API contract must be
    implemented by a PARSED backend route (incl. StaticFiles mounts, which
    satisfy HTML-serving contracts). Field/type conformance is checked only
    against the route's owning file plus the files defining its
    request/response models — never against README/css/requirements text.
    Frontend coverage: a backend-implemented contract not called from any
    frontend fetch is flagged (static/asset routes are exempt — the browser
    references them, not fetch()).
    """
    conflicts: list[dict] = []
    contracts = state.architecture.api_contracts or []
    if not contracts:
        return conflicts

    if routes_ctx is None:
        routes_ctx = collect_backend_routes(state)
    schemas = schema_fields if schema_fields is not None else collect_schema_fields(state)
    routes, _api_prefixes = routes_ctx

    # Class name → defining file (schemas/ wins where duplicated).
    model_to_file: dict[str, str] = {}
    for path in state.generated_code.backend:
        code = get_file_code(path, state.code_manifest.files.get(path), state)
        if not code:
            continue
        for name in parse_pydantic_schemas(code):
            if "/schemas/" in path or name not in model_to_file:
                model_to_file[name] = path

    # All fetch calls across written frontend files.
    frontend_fetches: list[dict] = []
    for path, fm in state.code_manifest.files.items():
        if (
            fm.agent != "frontend" or not fm.written
            or not path.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"))
        ):
            continue
        code = get_file_code(path, fm, state)
        if not code:
            continue
        for fetch in parse_frontend_fetches(code):
            frontend_fetches.append({**fetch, "filepath": path, "task_id": fm.task_id})

    for c in contracts:
        method = c.method.upper()
        contract_path = c.path.strip()
        path_key = normalize_api_path(contract_path)

        route, candidates = match_backend_route(contract_path, method, routes, allow_api_prefix=True)
        if route is None:
            # Attribute to the closest-matching route's OWNER when a
            # similar route exists (e.g. path/prefix drift): the router file
            # is the only file whose rework can fix the path. Fall back to
            # the backend ENTRYPOINT so the conflict always lands in a real
            # file's per-file rework instructions (a virtual "backend" key
            # is never delivered to any file).
            owner = "backend"
            hint = ""
            if candidates:
                owner = candidates[0].get("filepath") or "backend"
                hint = f" Closest route is '{candidates[0]['method']} {candidates[0]['path']}' in '{owner}' — align the path (prefix/params), don't duplicate."
            if owner == "backend" or owner not in state.code_manifest.files:
                owner = next(
                    (p for p in state.code_manifest.files
                     if p.endswith(("main.py", "app.py", "server.py")) and "backend" in p),
                    "backend",
                )
            conflicts.append({
                "type": "CONTRACT_COVERAGE",
                "filepath": owner,
                "task_id": state.code_manifest.files.get(owner).task_id if owner in state.code_manifest.files else "unknown",
                "expected": f"Backend implementation for endpoint '{method} {contract_path}'",
                "actual": "Missing",
                "reason": f"API contract endpoint '{method} {contract_path}' is not implemented in any backend file.{hint}",
            })
            continue

        owner = route.get("filepath") or "backend"
        if route.get("is_mount"):
            # An HTML static mount satisfies an HTML-serving contract — the
            # page itself is the frontend's domain, so no field checks and
            # no fetch() requirement.
            continue
        if route.get("static"):
            # File/HTML-serving routes are referenced by the browser (link/
            # script/img), not via fetch() — skip the frontend-call check.
            pass

        scope_files = set()
        if owner in state.code_manifest.files:
            scope_files.add(owner)
        for model in (route.get("request_model") or "", route.get("response_model") or ""):
            if model and model in model_to_file:
                scope_files.add(model_to_file[model])
        scoped_code = "\n".join(
            get_file_code(p, state.code_manifest.files.get(p), state) or ""
            for p in scope_files
        )
        if scoped_code.strip():
            for field, ftype in c.request_body.items():
                err = check_type_consistency(field, ftype, scoped_code)
                if err:
                    conflicts.append({
                        "type": "CONTRACT_COVERAGE",
                        "filepath": owner,
                        "task_id": state.code_manifest.files.get(owner).task_id if owner in state.code_manifest.files else "unknown",
                        "expected": f"Request field '{field}' ({ftype})",
                        "actual": err,
                        "reason": f"Backend file '{owner}' for path '{contract_path}' has contract field issue: {err}.",
                    })
            for field, ftype in c.response_body.items():
                err = check_type_consistency(field, ftype, scoped_code)
                if err:
                    conflicts.append({
                        "type": "CONTRACT_COVERAGE",
                        "filepath": owner,
                        "task_id": state.code_manifest.files.get(owner).task_id if owner in state.code_manifest.files else "unknown",
                        "expected": f"Response field '{field}' ({ftype})",
                        "actual": err,
                        "reason": f"Backend file '{owner}' for path '{contract_path}' has contract field issue: {err}.",
                    })

        # Frontend coverage: a backend-implemented contract must be called
        # from some frontend fetch — unless it is a static/asset route.
        if route.get("static"):
            continue
        called = any(
            fetch["method"].upper() == method
            and (
                normalize_api_path(fetch["path"]) == path_key
                or strip_leading_api(normalize_api_path(fetch["path"])) == path_key
                or normalize_api_path(fetch["path"]) == strip_leading_api(path_key)
            )
            for fetch in frontend_fetches
        )
        if not called:
            # Attribute to the first written frontend module so the conflict
            # is delivered through per-file rework instructions.
            frontend_target = next(
                (f["filepath"] for f in frontend_fetches),
                next(
                    (p for p, fm in state.code_manifest.files.items()
                     if fm.agent == "frontend" and fm.written
                     and p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"))),
                    "frontend",
                ),
            )
            conflicts.append({
                "type": "CONTRACT_COVERAGE",
                "filepath": frontend_target,
                "task_id": state.code_manifest.files.get(frontend_target).task_id if frontend_target in state.code_manifest.files else "unknown",
                "expected": f"Frontend call/reference to endpoint '{method} {contract_path}'",
                "actual": "Missing",
                "reason": f"API contract endpoint '{method} {contract_path}' is implemented in backend but not called/referenced in any frontend file.",
            })

    return conflicts


# ── Phase 1D: Cross-layer semantic checks (A1–A6) ─────────────────────────


def check_frontend_dom(state: ProjectState) -> list[dict]:
    """
    A5 (FRONTEND_DOM) — JS element ids must exist in HTML; classes used in
    HTML/JS must be defined in CSS; every JS/CSS file must be loaded by an
    HTML entrypoint (and every referenced asset must exist).
    """
    conflicts: list[dict] = []
    files = state.code_manifest.files
    html_paths = [p for p in files if p.endswith((".html", ".htm"))]
    js_paths = [p for p in files if p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"))]
    css_paths = [p for p in files if p.endswith((".css", ".scss", ".sass"))]
    if not html_paths:
        return conflicts

    html_code: dict[str, str] = {}
    html_ids: set[str] = set()
    html_classes: set[str] = set()
    for p in html_paths:
        code = get_file_code(p, files.get(p), state)
        if code:
            html_code[p] = code
            html_ids |= parse_html_ids(code)
            html_classes |= parse_html_classes(code)

    # 1. JS-referenced element ids must exist.
    for p in js_paths:
        fm = files.get(p)
        code = get_file_code(p, fm, state)
        if not code:
            continue
        for i in parse_js_used_ids(code):
            if i in html_ids:
                continue
            conflicts.append({
                "type": "FRONTEND_DOM",
                "filepath": p,
                "task_id": fm.task_id if fm else "unknown",
                "expected": f"Element with id='{i}' in an HTML file",
                "actual": "missing",
                "reason": (
                    f"JS '{p}' references element id '{i}' but no HTML file defines it. "
                    f"Defined ids: {sorted(html_ids) or '(none)'}."
                ),
            })

    # 2. Classes used in HTML/JS must be defined in CSS.
    css_classes: set[str] = set()
    for p in css_paths:
        code = get_file_code(p, files.get(p), state)
        if code:
            css_classes |= parse_css_classes(code)
    if css_classes:
        used = set(html_classes)
        for p in js_paths:
            code = get_file_code(p, files.get(p), state)
            used |= parse_js_classes(code or "")
        for c in sorted(used):
            if c in css_classes:
                continue
            css_target = _resolve_css_target(c, p, css_paths, js_paths, html_paths, html_code)
            conflicts.append({
                "type": "FRONTEND_DOM",
                "filepath": css_target,
                "task_id": files.get(css_target).task_id if css_target in files else "unknown",
                "expected": f"CSS class '.{c}'",
                "actual": "undefined in CSS",
                "reason": (
                    f"Class '{c}' is used in HTML/JS but never defined in any CSS "
                    f"file (defined: {sorted(css_classes)[:12] or '(none)'})."
                ),
            })

    # 3. Asset loading: every JS/CSS file must be referenced by some HTML.
    asset_exts = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".css", ".scss", ".sass")
    loaded: set[str] = set()
    for p in html_paths:
        code = html_code.get(p, "")
        layer = layer_root_for(p)
        for m in re.finditer(r"(?:src|href)\s*=\s*['\"]([^'\"]+)['\"]", code):
            ref = m.group(1)
            if ref.startswith(("http://", "https://", "//", "data:", "#")) or "{{" in ref:
                continue
            if ref.startswith("/"):
                # Web-root-relative reference — resolve against the layer root
                # (output/frontend/ for index.html → /static/... works).
                base_dir = layer or os.path.dirname(p)
                ref = ref.lstrip("/")
            else:
                base_dir = os.path.dirname(p)
            loaded.add(os.path.normpath(os.path.join(base_dir, ref)).replace("\\", "/"))
    for p in list(js_paths) + list(css_paths):
        fm = files.get(p)
        normalized = p.lstrip("./")
        if p in loaded or normalized in {x.lstrip("./") for x in loaded}:
            continue
        conflicts.append({
            "type": "FRONTEND_DOM",
            "filepath": p,
            "task_id": fm.task_id if fm else "unknown",
            "expected": f"'{p}' is loaded by an HTML entrypoint",
            "actual": "not referenced",
            "reason": (
                f"Script/stylesheet '{p}' is never referenced by any HTML file — no "
                f"<script src> or <link href> points at it. Wire it into the HTML "
                f"entrypoint (and do NOT duplicate its logic inline in HTML)."
            ),
        })
    return conflicts


def check_file_set(state: ProjectState) -> list[dict]:
    """
    A6 (FILE_SET) — every file the planner committed to (state.required_files)
    must be planned, written, and present on disk.
    """
    conflicts: list[dict] = []
    for req in (state.required_files or []):
        fm = state.code_manifest.files.get(req)
        if not fm:
            conflicts.append({
                "type": "FILE_SET",
                "filepath": req,
                "task_id": "planner",
                "expected": f"Required project file '{req}' is planned in the manifest",
                "actual": "missing",
                "reason": f"Required project file '{req}' is not planned in the code manifest.",
            })
        elif not fm.written:
            conflicts.append({
                "type": "FILE_SET",
                "filepath": req,
                "task_id": fm.task_id,
                "expected": f"Required project file '{req}' is written",
                "actual": "planned but not written",
                "reason": f"Required project file '{req}' is planned but was never written by its owning agent.",
            })
        else:
            abs_p = os.path.join(_root_dir, normalize_output_path(req))
            if not os.path.exists(abs_p):
                conflicts.append({
                    "type": "FILE_SET",
                    "filepath": req,
                    "task_id": fm.task_id,
                    "expected": f"Required project file '{req}' exists on disk",
                    "actual": "missing on disk",
                    "reason": f"Required project file '{req}' is marked written but missing on disk.",
                })
    return conflicts


def check_file_corruption(state: ProjectState) -> list[dict]:
    """
    A7 — statically detect corrupted generated content (repeated-block
    duplication / JSON-escape leakage) in every in-memory file.
    """
    conflicts: list[dict] = []
    code_map = getattr(state, "generated_code", None)
    if code_map is None:
        return conflicts
    for layer in ("backend", "frontend", "database", "devops"):
        for filepath, code in (getattr(code_map, layer, {}) or {}).items():
            conflicts.extend(detect_corruption(code or "", filepath))
    return conflicts


def run_phase_a_checks(state: ProjectState) -> list[dict]:
    """Run all cross-layer semantic checks (A1–A7); never crash on a bad fixture."""
    # Parse-once context: backend routes and schema fields are collected at
    # most once per Phase A pass and shared across the checks that need them.
    ctx: dict[str, Any] = {}

    def _routes() -> tuple[list[dict], set[str]]:
        if "routes_ctx" not in ctx:
            ctx["routes_ctx"] = collect_backend_routes(state)
        return ctx["routes_ctx"]

    def _schemas() -> dict[str, set[str]]:
        if "schema_fields" not in ctx:
            ctx["schema_fields"] = collect_schema_fields(state)
        return ctx["schema_fields"]

    checks = [
        ("startup-assets", check_startup_assets),
        ("model-schema", lambda s: check_model_schema_consistency(s, schema_fields=_schemas())),
        ("layer-wiring", check_layer_wiring),
        ("router-prefixes", check_router_prefixes),
        ("duplicate-foundation", check_duplicate_foundation),
        ("frontend-backend", lambda s: check_frontend_backend_contracts(s, routes_ctx=_routes(), schema_fields=_schemas())),
        ("contract-coverage", lambda s: check_contract_coverage(s, routes_ctx=_routes(), schema_fields=_schemas())),
        ("frontend-dom", check_frontend_dom),
        ("file-set", check_file_set),
        ("file-corruption", check_file_corruption),
    ]
    conflicts: list[dict] = []
    for name, check in checks:
        try:
            conflicts.extend(check(state))
        except Exception as exc:
            logger.error("[IntegrationAgent] Phase A check '%s' failed: %s", name, exc)
    # Deduplicate identical findings (e.g. the same bad column used 4 times).
    seen: set[tuple] = set()
    unique: list[dict] = []
    for c in conflicts:
        key = (c.get("type"), c.get("filepath"), c.get("expected"), c.get("actual"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique
