"""
agents/planner.py
─────────────────
Planner Agent — TWO-PASS planning (Problem 1).

PASS 1 — Single skeleton LLM call.
  One LLM call produces the COMPLETE project file tree (agent / file / exports /
  imports_from). Global visibility means it cannot invent phantom imports.

PASS 2 — Deterministic task generation (no LLM).
  Topological sort over the import graph assigns depends_on + priority, then
  FileManifest entries are built. Zero LLM creativity in ordering.
"""

import logging
from threading import Lock
from typing import Any

from .base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF
from providers.llm.base import BaseLLMProvider
from core.workspace import (
    BACKEND_ROOT, FRONTEND_ROOT, OUTPUT_DIR, connection_module_path, dotted_import, relative_import, reroot,
    workspace_path,
)
from core.state import (
    ProjectState, Task, TaskList, FileManifest, CodeManifest, APIContract, ArchData,
    CONNECTION_MODULE_EXPORTS, build_connection_module, build_env_example,
    persistence_mandate, render_persistence_contract,
)
import os
import re

logger = logging.getLogger(__name__)

VALID_AGENTS: frozenset[str] = frozenset({"database", "backend", "frontend"})
AGENT_PRIORITY: dict[str, int] = {"database": 1, "backend": 2, "frontend": 2}
AGENT_PREFIX: dict[str, str] = {"database": "db", "backend": "be", "frontend": "fe"}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Pass-1 system + user prompts                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_SKELETON_SYSTEM = """\
You are a senior engineering project manager. Design the COMPLETE file structure for the ENTIRE
project in a single response. You have GLOBAL visibility over every file, so you MUST list every
file — including each file that any other file imports from.

Respond with a SINGLE JSON object — no prose outside it:
{"skeleton": [
  {"id":"<ID>","agent":"<database|backend|frontend>","title":"<short>",
   "description":"<what this file does>","file":"<repo-relative path>",
   "exports":["<name>", ...],
   "imports_from":{"<source file path>":["<name>", ...], ...}}
]}

Rules:
- ID prefix by owner agent: db-*, be-*, fe-* (e.g. be-001).
- exports: ONLY cross-module names — what OTHER files import from THIS file.
- imports_from: the exact source file path(s) this file imports from, and which names it imports.
  This is how the system computes build order — be precise and exhaustive.
- EVERY file referenced in imports_from MUST itself appear in the skeleton. No phantom imports.
- Include entrypoint/wiring files: backend main.py (registers routers) and frontend index.html.
- Unique, non-empty, repo-relative file paths.

Layer roots (fixed — every path MUST start with one of them):
- backend/  — every backend AND database file (ORM models, the DB connection module, any .sql),
  plus requirements.txt and .env.example. The entrypoint is backend/main.py. Python imports are
  relative to backend/ (backend/db/session.py is imported as `db.session`).
- frontend/ — every frontend file, including package manifest and bundler config. The entrypoint
  is frontend/index.html.
- README.md is the only file outside a Layer root.
"""

SKELETON_USER_TEMPLATE = """\
### Software Requirements (Functional)
{srs_bullets}

### Tech Stack
{tech_stack}

### API Contracts
{api_contracts}

### Database Schema Overview
{database_schema_overview}

{persistence_contract}

{complexity_constraint}
---
Generate the complete project skeleton. JSON object only:
```json
"""


def _srs_bullets(srs: Any) -> str:
    """Render functional requirements (compact) for the skeleton prompt."""
    reqs = getattr(srs, "functional_requirements", None)
    if not reqs:
        return "  (none provided)"
    return "\n".join(f"- {r}" for r in reqs)


def _fmt_tech_stack(stack: dict[str, str]) -> str:
    if not stack:
        return "  (none)"
    return "\n".join(f"- {k}: {v}" for k, v in stack.items())


def _complexity_constraint(tier: str) -> str:
    """Prompt-level constraint calibrating file count to user intent (Problem 4).

    Every tier carries the persistence mandate: every run has a database.
    """
    tier = (tier or "STANDARD").upper()
    limits = {"MINIMAL": "8-12", "STANDARD": "15-30", "ENTERPRISE": "30-60+"}
    return (
        f"## Complexity: {tier}\n"
        f"- Target total project file count: {limits.get(tier, '15-30')}.\n"
        + {
            "MINIMAL": "- No auth/user tables unless explicitly requested.\n"
                       "- Single DB table where possible. Minimal CSS. No state management.\n"
                       "- Do NOT create notification/modal/loading components unless requested.",
            "STANDARD": "- Full CRUD, optional auth, moderate styling.\n"
                        "- Only create files that map to stated requirements — no speculative scaffolding.",
            "ENTERPRISE": "- Full auth/RBAC/monitoring as warranted.\n"
                          "- Every file must map to a stated requirement.",
        }.get(tier, "- Only create files that map to stated requirements.")
        + "\n" + persistence_mandate(tier)
    )


# ── Formatting helpers ───────────────────────────────────────────────────────────

def _fmt_api_contracts(contracts: list[APIContract]) -> str:
    if not contracts:
        return "  (none)"
    lines = []
    for c in contracts:
        lock = "🔒" if c.auth_required else "🔓"
        lines.append(
            f"  {lock} {c.method:7} {c.path}\n"
            f"         desc         : {c.description}\n"
            f"         request_body : {c.request_body}\n"
            f"         response_body: {c.response_body}"
        )
    return "\n".join(lines)


def _extract_table_names(schema_overview: str) -> str:
    """Extract just table names from db schema overview for backend depends_on prediction."""
    if not schema_overview:
        return "(none)"
    lines = []
    counter = 0
    for line in schema_overview.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Take just the table name (first word before parenthesis or space)
        table = line.split("(")[0].split(" ")[0].strip()
        if table:
            counter += 1
            lines.append(f"  db-{counter:03d}: {table}")
    return "\n".join(lines) or "(none)"


# ── Validation helpers ───────────────────────────────────────────────────────────

def _has_cycle(tasks: list[Task]) -> bool:
    """Detect cycles via iterative DFS (paint algorithm)."""
    graph: dict[str, list[str]] = {t.id: list(t.depends_on) for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {t.id: WHITE for t in tasks}

    def dfs(start: str) -> bool:
        stack = [(start, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                color[node] = BLACK
                continue
            if color[node] == GRAY:
                return True
            if color[node] == BLACK:
                continue
            color[node] = GRAY
            stack.append((node, True))
            for neighbour in graph.get(node, []):
                if neighbour in color:
                    if color[neighbour] == GRAY:
                        return True
                    if color[neighbour] == WHITE:
                        stack.append((neighbour, False))
        return False

    return any(dfs(t.id) for t in tasks if color[t.id] == WHITE)


def _validate_tasks(raw_list: list[Any]) -> tuple[list[Task], list[str]]:
    """Parse raw dicts into Task objects with full validation."""
    warnings: list[str] = []
    tasks:    list[Task] = []
    seen_ids: set[str]   = set()

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            warnings.append(f"task[{i}] is not a dict — skipped")
            continue
        tid   = str(item.get("id", "")).strip()
        if not tid:
            warnings.append(f"task[{i}] missing required field 'id' — skipped")
            continue

        agent = str(item.get("agent", "")).strip().lower()
        if not agent:
            # Infer agent from ID prefix (e.g. be-001 -> backend, db-001 -> database, fe-001 -> frontend)
            if tid.startswith("be-"):
                agent = "backend"
            elif tid.startswith("db-"):
                agent = "database"
            elif tid.startswith("fe-"):
                agent = "frontend"

        if agent not in VALID_AGENTS:
            warnings.append(f"task '{tid}': unknown or missing agent '{agent}' — skipped")
            continue
        if tid in seen_ids:
            warnings.append(f"duplicate task ID '{tid}' — skipped")
            continue
        seen_ids.add(tid)
        tasks.append(Task(
            id=tid, agent=agent,
            title=str(item.get("title", "")),
            description=str(item.get("description", "")),
            depends_on=[str(d) for d in item.get("depends_on", [])],
            priority=AGENT_PRIORITY[agent], status="pending",
        ))

    # Pass 2: validate depends_on references
    valid_ids = {t.id for t in tasks}
    for task in tasks:
        bad = [d for d in task.depends_on if d not in valid_ids]
        if bad:
            warnings.append(f"task '{task.id}': depends_on references unknown IDs {bad} — removed")
            task.depends_on = [d for d in task.depends_on if d in valid_ids]

    # Pass 3: frontend must not depend on backend
    backend_ids = {t.id for t in tasks if t.agent == "backend"}
    for task in tasks:
        if task.agent != "frontend":
            continue
        be_deps = [d for d in task.depends_on if d in backend_ids]
        if be_deps:
            warnings.append(f"task '{task.id}' (frontend) depends on backend task(s) {be_deps} — removed")
            task.depends_on = [d for d in task.depends_on if d not in backend_ids]

    # Pass 4: cycle detection
    if _has_cycle(tasks):
        warnings.append("CYCLE DETECTED — clearing all depends_on to recover")
        logger.error("[PlannerAgent] circular dependency detected — clearing all depends_on")
        for task in tasks:
            task.depends_on = []

    return tasks, warnings


#: Project-level file that stays at the top of `output/`, outside any Layer root.
_TOP_LEVEL_FILES = frozenset({"README.md"})


def _place(path: str, agent: str, errors: list[str] | None = None) -> str:
    """
    Workspace path of a skeleton path under its owning coding agent's Layer root.
    A path that names the other layer's directory is re-rooted and logged.
    """
    rel = workspace_path(path)[len(OUTPUT_DIR) + 1:]
    if rel in _TOP_LEVEL_FILES:
        return f"{OUTPUT_DIR}/{rel}"
    placed, misplaced = reroot(path, agent)
    if misplaced and errors is not None:
        errors.append(
            f"skeleton path '{path}' (owner: {agent}) names the other layer's directory "
            f"— re-rooted under its Layer root as '{placed}'"
        )
    return placed


def _build_manifest(
    tasks: list[Task], raw_list: list[Any], arch: Any = None, errors: list[str] | None = None,
) -> dict[str, FileManifest]:
    """
    Build CodeManifest entries from validated tasks + raw LLM dicts.

    Every path is re-rooted under its owning coding agent's Layer root
    (`output/backend/` for database and backend, `output/frontend/` for
    frontend), whatever layout the skeleton used. Misplaced paths are
    appended to *errors*.
    """
    raw_by_id: dict[str, dict] = {}
    for item in raw_list:
        if isinstance(item, dict) and item.get("id"):
            raw_by_id[str(item["id"]).strip()] = item

    manifest_files: dict[str, FileManifest] = {}
    placed_paths: dict[str, str] = {}   # skeleton path (workspace form) → re-rooted path
    for task in tasks:
        raw_item  = raw_by_id.get(task.id, {})
        file_path = str(raw_item.get("file", "")).strip().lstrip("/").lstrip("\\")

        if not file_path:
            # Fallback file path if LLM omitted 'file' field
            sanitized_title = re.sub(r'[^a-zA-Z0-9_]', '_', task.title.lower()).strip('_') or f"task_{task.id}"
            if task.agent == "database":
                folder, ext = arch.persistence_contract.model_dir, ".py"
            elif task.agent == "backend":
                folder, ext = "app/api", ".py"
            else:
                folder, ext = "src", ".js"
            file_path = f"{folder}{sanitized_title}{ext}"
            logger.info("[PlannerAgent] task '%s' missing 'file' field — created fallback path: %s", task.id, file_path)

        raw_path = file_path
        file_path = _place(raw_path, task.agent, errors)

        if file_path in manifest_files:
            # Disambiguate duplicate file paths by suffixing task_id
            parts = file_path.rsplit(".", 1)
            file_path = f"{parts[0]}_{task.id}.{parts[1]}" if len(parts) == 2 else f"{file_path}_{task.id}"
            logger.info("[PlannerAgent] disambiguated duplicate file path to '%s'", file_path)

        manifest_files[file_path] = FileManifest(
            agent=task.agent, task_id=task.id, filepath=file_path,
            purpose=str(raw_item.get("description", ""))[:120],
            exports=[str(e) for e in raw_item.get("exports", [])],
        )
        placed_paths[workspace_path(raw_path)] = file_path

    # Persist the Pass-1 import graph (imports_from) into each manifest entry.
    # The coding agents read this to know their EXACT allowed imports — the LLM
    # must never guess import paths. Sources are normalized to the manifest's
    # workspace-relative form and validated against the final file set.
    for task in tasks:
        raw_item = raw_by_id.get(task.id, {})
        fm = next((m for m in manifest_files.values() if m.task_id == task.id), None)
        if fm is None:
            continue
        imports_from: dict[str, list[str]] = {}
        for src_path, names in (raw_item.get("imports_from") or {}).items():
            src = workspace_path(src_path)
            src = placed_paths.get(src, src)
            if src == fm.filepath or src not in manifest_files:
                if src not in manifest_files:
                    logger.warning(
                        "[PlannerAgent] task '%s' imports unknown file '%s' (phantom import) — dropped from import contract",
                        task.id, src,
                    )
                continue
            imports_from[src] = [str(n) for n in names if str(n).strip()]
        fm.imports_from = imports_from

    # Phase 2 — deterministic cross-agent field contracts (no LLM). Backend agents
    # read these to match DB columns / API bodies while the DB agent runs in parallel.
    _populate_contracts(manifest_files, arch)

    return manifest_files


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 2 — deterministic cross-agent field contracts (no LLM)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

#: Class-name suffixes stripped when matching an export to a DB table or API
#: resource keyword (e.g. 'TodoItem' → 'todo', 'TodoCreate' → 'todo').
_ENTITY_SUFFIXES: tuple[str, ...] = (
    "create", "new", "insert", "in", "update", "put", "patch", "request",
    "payload", "form", "response", "out", "read", "view", "item", "record",
    "entity", "model", "table", "orm", "dto", "schema", "detail", "result",
)

_REQUEST_SUFFIXES: tuple[str, ...] = (
    "create", "new", "insert", "in", "update", "put", "patch", "request", "payload", "form",
)

_RESPONSE_SUFFIXES: tuple[str, ...] = (
    "response", "out", "read", "view", "item", "detail", "result",
)


def _norm_keyword(name: str) -> str:
    """'TodoItem' → 'todoitem' (lowercase, alphanumerics only)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _entity_forms(name: str) -> set[str]:
    """
    Normalized matching forms of a name: itself, its singular, and its
    suffix-stripped stem (e.g. 'TodoItem' → {'todoitem', 'todo'},
    'users' → {'users', 'user'}). Empty forms are dropped.
    """
    kw = _norm_keyword(name)
    forms = {kw}
    if kw.endswith("s"):
        forms.add(kw[:-1])
    stripped = kw
    for suf in _ENTITY_SUFFIXES:
        if stripped.endswith(suf):
            stripped = stripped[: -len(suf)]
            break
    forms.add(stripped)
    if stripped.endswith("s"):
        forms.add(stripped[:-1])
    return {f for f in forms if f}


def _match_entity(export: str, candidates: dict[str, Any]) -> str | None:
    """
    Match an export/class name to EXACTLY ONE candidate key (table or API resource).
    Deterministic only-match rule: no match, or >1 candidate → None (never guess).
    """
    export_forms = _entity_forms(export)
    matched = [k for k in candidates if _entity_forms(k) & export_forms]
    return matched[0] if len(matched) == 1 else None


def _parse_schema_overview(overview: str) -> dict[str, dict[str, str]]:
    """
    Parse the architecture agent's schema-overview lines into {table: {column: type}}.
    Handles e.g. 'users (id UUID PK, email VARCHAR UNIQUE, created_at TIMESTAMP)'.
    """
    tables: dict[str, dict[str, str]] = {}
    for line in (overview or "").splitlines():
        line = line.strip()
        if not line or "(" not in line:
            continue
        name = line.split("(", 1)[0].strip()
        if not name:
            continue
        inner = line.split("(", 1)[1].rsplit(")", 1)[0]
        fields: dict[str, str] = {}
        for part in inner.split(","):
            tokens = part.strip().split()
            if len(tokens) >= 2:
                fields[tokens[0]] = " ".join(tokens[1:])
        if fields:
            tables[name] = fields
    return tables


def _is_request_schema(export: str) -> bool:
    kw = _norm_keyword(export)
    return any(kw.endswith(s) for s in _REQUEST_SUFFIXES)


def _is_response_schema(export: str) -> bool:
    kw = _norm_keyword(export)
    return any(kw.endswith(s) for s in _RESPONSE_SUFFIXES)


def _is_schema_file(fm: FileManifest) -> bool:
    """Backend file defining request/response models (schemas/models module, or schema-style exports)."""
    path = (fm.filepath or "").lower()
    if "/schemas/" in path or "/models/" in path or path.endswith(("schemas.py", "models.py")):
        return True
    for export in fm.exports or []:
        kw = _norm_keyword(export)
        if any(kw.endswith(s) for s in _ENTITY_SUFFIXES):
            return True
    return False


def _populate_contracts(manifest_files: dict[str, FileManifest], arch: Any = None) -> None:
    """
    Deterministically attach a field contract to each DB-model and schema file:

      - database-agent files → fields from `database_schema_overview` (table → columns),
                               matched to model exports by normalized name
                               (e.g. table 'todos' → export 'TodoItem').
      - backend schema files → fields from the matching API contract's
                               request/response bodies, matched by path resource
                               (e.g. '/api/todos' → export 'TodoCreate').

    No LLM involvement — a name that matches nothing (or several things) is skipped.
    """
    if arch is None:
        return

    tables = _parse_schema_overview(getattr(arch, "database_schema_overview", "") or "")

    resources: dict[str, list[Any]] = {}
    for c in (getattr(arch, "api_contracts", None) or []):
        key = ""
        for seg in str(getattr(c, "path", "") or "").strip("/").split("/"):
            if seg and seg.lower() != "api":
                key = seg
                break
        if key:
            resources.setdefault(key, []).append(c)

    for fm in manifest_files.values():
        if fm.agent == "database" and tables:
            for export in list(fm.exports or []):
                table = _match_entity(export, tables)
                if table is None:
                    continue
                fm.contract[export] = dict(tables[table])
        elif fm.agent == "backend" and _is_schema_file(fm):
            for export in list(fm.exports or []):
                key = _match_entity(export, resources)
                if key is None:
                    continue
                fields: dict[str, str] = {}
                for c in resources[key]:
                    req = dict(getattr(c, "request_body", None) or {})
                    res = dict(getattr(c, "response_body", None) or {})
                    if _is_request_schema(export):
                        fields.update(req)
                    elif _is_response_schema(export):
                        fields.update(res)
                    else:
                        fields.update({**req, **res})
                if fields:
                    fm.contract[export] = fields


def _move_file(manifest_files: dict[str, FileManifest], old: str, new: str) -> None:
    """Move a File Manifest from *old* to *new*, repointing every import of it."""
    fm = manifest_files.pop(old)
    fm.filepath = new
    manifest_files[new] = fm
    for other in manifest_files.values():
        if old in other.imports_from:
            names = other.imports_from.pop(old)
            merged = other.imports_from.get(new, [])
            other.imports_from[new] = merged + [n for n in names if n not in merged]


def _shallowest(paths: list[str]) -> str:
    return min(paths, key=lambda p: (p.count("/"), p))


def _adopt_skeleton_anchors(
    manifest_files: dict[str, FileManifest],
    arch: ArchData,
) -> dict[str, FileManifest]:
    """
    Move the skeleton's own backend entrypoint and connection module to their
    fixed paths, so the scaffold and persistence-seam steps take them over
    instead of planning a second one beside them.

    A skeleton using an `app/` package plans e.g. `output/backend/app/main.py`
    and `output/backend/app/db/database.py`; without this the backend would get
    two entrypoints and two connection modules.

      - entrypoint: a backend `main.py` anywhere under the backend Layer root
        → `output/backend/main.py`
      - connection module: a backend/database `.py` exporting `engine` or
        `SessionLocal` → the Persistence Contract's connection module path
        (a dependency module exporting only `get_db` is left alone)

    Nothing moves when the fixed path is already planned; among several
    candidates the shallowest wins.
    """
    def backend_owned(fm: FileManifest) -> bool:
        return fm.agent in ("backend", "database") and fm.filepath.startswith(BACKEND_ROOT)

    be_main = f"{BACKEND_ROOT}main.py"
    if be_main not in manifest_files:
        mains = [p for p, fm in manifest_files.items()
                 if backend_owned(fm) and os.path.basename(p) == "main.py"]
        if mains:
            old = _shallowest(mains)
            logger.warning("[PlannerAgent] skeleton entrypoint '%s' moved to '%s'", old, be_main)
            _move_file(manifest_files, old, be_main)

    session_path = connection_module_path(arch.persistence_contract)
    if session_path not in manifest_files:
        defining = {"engine", "SessionLocal"}
        modules = [p for p, fm in manifest_files.items()
                   if backend_owned(fm) and p.endswith(".py") and defining & set(fm.exports or [])]
        if modules:
            old = _shallowest(modules)
            logger.warning("[PlannerAgent] skeleton connection module '%s' moved to '%s'", old, session_path)
            _move_file(manifest_files, old, session_path)

    return manifest_files


def _next_task_id(prefix: str, existing_ids: set[str]) -> str:
    """Reserve and return the first free injected-task id ('<prefix>-900' upward)."""
    n = 900
    while f"{prefix}-{n:03d}" in existing_ids:
        n += 1
    tid = f"{prefix}-{n:03d}"
    existing_ids.add(tid)
    return tid


def _inject_scaffold_tasks(
    tasks: list[Task],
    manifest_files: dict[str, FileManifest],
) -> tuple[list[Task], dict[str, FileManifest]]:
    """
    Add mandatory wiring/scaffold tasks (Problem 2):
      - backend  → output/backend/main.py      (registers every backend router)
      - frontend → output/frontend/index.html  (loads every stylesheet + script)
    Both sit directly at their fixed Layer root.
    Each scaffold depends on ALL of its own agent's tasks so it runs last in that
    agent's execution order and sees the complete list of files to wire.
    """
    existing_ids = {t.id for t in tasks}
    existing_paths = set(manifest_files)
    next_tasks = list(tasks)
    next_manifest = dict(manifest_files)

    be_main = f"{BACKEND_ROOT}main.py"
    fe_index = f"{FRONTEND_ROOT}index.html"

    backend_task_ids = [t.id for t in tasks if t.agent == "backend"]
    if backend_task_ids and be_main not in existing_paths:
        tid = _next_task_id("be", existing_ids)
        next_tasks.append(Task(
            id=tid, agent="backend",
            title="Backend entrypoint (main.py)",
            description="FastAPI application entry point that registers all routers, adds CORS, and serves static files.",
            depends_on=list(backend_task_ids), priority=AGENT_PRIORITY["backend"], status="pending",
        ))
        next_manifest[be_main] = FileManifest(
            agent="backend", task_id=tid, filepath=be_main,
            purpose="FastAPI application entry point — registers all routers and middleware.",
            exports=["app"],
            imports_from={
                p: ["router"] for p, fm in manifest_files.items()
                if fm.agent == "backend" and "router" in (fm.exports or [])
            },
        )

    frontend_task_ids = [t.id for t in tasks if t.agent == "frontend"]
    if frontend_task_ids and fe_index not in existing_paths:
        tid = _next_task_id("fe", existing_ids)
        next_tasks.append(Task(
            id=tid, agent="frontend",
            title="Frontend entrypoint (index.html)",
            description="HTML entry point that loads every stylesheet and script in the correct order.",
            depends_on=list(frontend_task_ids), priority=AGENT_PRIORITY["frontend"], status="pending",
        ))
        next_manifest[fe_index] = FileManifest(
            agent="frontend", task_id=tid, filepath=fe_index,
            purpose="HTML entry point loading all stylesheets and scripts.",
            exports=["index"],
        )

    return next_tasks, next_manifest


def _inject_router_task(
    tasks: list[Task],
    manifest_files: dict[str, FileManifest],
    arch: Any,
) -> tuple[list[Task], dict[str, FileManifest]]:
    """
    Deterministically commit to an API-route module whenever the architecture
    declares API contracts but the planned file set contains no router-shaped
    file (Problem-2 guard: main.py must register routers that actually exist).

    Created only when ALL hold:
      - at least one backend task exists (a FastAPI layer),
      - arch.api_contracts is non-empty,
      - no existing backend file already looks router-shaped
        (paths containing /api/, /routers/, /routes/, or router-style basenames).

    The router task depends on every other backend task (runs last in its
    layer) and the backend entrypoint's depends_on is re-wired to include it,
    so main.py always registers this module and deviations are caught by the
    import checks downstream.
    """
    if not arch or not getattr(arch, "api_contracts", None):
        return tasks, manifest_files
    backend_task_ids = [t.id for t in tasks if t.agent == "backend"]
    if not backend_task_ids:
        return tasks, manifest_files

    be_prefix = BACKEND_ROOT

    def _router_shaped(path: str) -> bool:
        base = path.split("/")[-1]
        return (
            "/api/" in path or path.endswith("/api")
            or "/routers/" in path or "/routes/" in path
            or base in ("routes.py", "router.py", "api.py")
            or base.startswith("r_")
            or base.endswith("_routes.py") or base.endswith("_router.py")
        )

    if any(p.startswith(be_prefix) and _router_shaped(p) for p in manifest_files):
        return tasks, manifest_files

    existing_ids = {t.id for t in tasks}
    next_tasks = list(tasks)
    next_manifest = dict(manifest_files)

    # Module name from the first contract path segment (e.g. /api/todos → todos).
    module_name = ""
    for c in arch.api_contracts:
        segs = [s for s in c.path.strip("/").split("/") if s and s != "api"]
        cand = re.sub(r"[^a-zA-Z0-9_]", "", segs[0]) if segs else ""
        if cand:
            module_name = cand
            break
    module_name = module_name or "api"

    router_path = f"{BACKEND_ROOT}api/routes/{module_name}.py"
    if router_path in next_manifest:
        return tasks, manifest_files

    contracts_block = "\n".join(
        f"- {c.method.upper()} {c.path}  "
        f"request={c.request_body or None}  response={c.response_body or None}"
        for c in arch.api_contracts
    )
    tid = _next_task_id("be", existing_ids)
    next_tasks.append(Task(
        id=tid, agent="backend",
        title=f"FastAPI router ({module_name})",
        description=(
            f"Router module implementing the app's entire API surface. "
            f"Create thin HTTP handlers for EVERY contract below (verb + path + "
            f"request/response fields must match exactly) and DELEGATE data access "
            f"to the services layer — never inline business logic. "
            f"Export the APIRouter instance as 'router' (it is registered by "
            f"main.py).\nContracts:\n{contracts_block}"
        ),
        depends_on=[t.id for t in tasks if t.agent == "backend"],
        priority=AGENT_PRIORITY["backend"], status="pending",
    ))
    next_manifest[router_path] = FileManifest(
        agent="backend", task_id=tid, filepath=router_path,
        purpose="FastAPI router — implements every documented API contract, delegating to the services layer.",
        exports=["router"],
    )

    # Backend entrypoint must depend on (and thus run after) the router task.
    be_main = f"{BACKEND_ROOT}main.py"
    main_fm = next_manifest.get(be_main)
    if main_fm:
        main_fm.imports_from[router_path] = ["router"]
        for t in next_tasks:
            if t.id == main_fm.task_id and tid not in t.depends_on:
                t.depends_on = t.depends_on + [tid]

    return next_tasks, next_manifest


def _inject_project_files(
    tasks: list[Task],
    manifest_files: dict[str, FileManifest],
    arch: ArchData,
) -> tuple[list[Task], dict[str, FileManifest]]:
    """
    Commit to the deliverable's project-scaffolding file set (Phase C3).
    These files are REQUIRED for a runnable deliverable:
      - output/backend/requirements.txt  (Python dependency manifest)
      - output/README.md                 (run instructions)
      - output/backend/.env.example      (environment template — no real secrets)

    `.env.example` carries deterministic content built from the persistence
    contract (canonical DB URL variable), so no LLM writes it.
    """
    existing_ids = {t.id for t in tasks}
    next_tasks = list(tasks)
    next_manifest = dict(manifest_files)

    # (path, title, purpose, exports, planned_content — None → LLM writes it)
    specs = [
        (
            f"{BACKEND_ROOT}requirements.txt",
            "Python dependencies (requirements.txt)",
            "Locked Python dependency manifest covering every third-party import used by the backend.",
            ["requirements"],
            None,
        ),
        (
            "output/README.md",
            "Project README (README.md)",
            "Run instructions: install dependencies, start the backend, and open the frontend.",
            ["readme"],
            None,
        ),
        (
            f"{BACKEND_ROOT}.env.example",
            "Environment template (.env.example)",
            "Example environment variables (canonical DB URL) — never real secrets.",
            ["env_example"],
            build_env_example(arch.persistence_contract),
        ),
    ]
    for path, title, purpose, exports, planned_content in specs:
        if path in next_manifest:
            if planned_content is not None:
                next_manifest[path].planned_content = planned_content
            continue
        tid = _next_task_id("be", existing_ids)
        next_tasks.append(Task(
            id=tid, agent="backend", title=title, description=purpose,
            depends_on=[], priority=AGENT_PRIORITY["backend"], status="pending",
        ))
        next_manifest[path] = FileManifest(
            agent="backend", task_id=tid, filepath=path,
            purpose=purpose, exports=exports, planned_content=planned_content,
        )

    return next_tasks, next_manifest


def _inject_persistence_seam(
    tasks: list[Task],
    manifest_files: dict[str, FileManifest],
    arch: ArchData,
) -> tuple[list[Task], dict[str, FileManifest]]:
    """
    Guarantee the DB connection module from the architecture's Persistence
    Contract (every run has exactly one).

    The entry lives at the connection module's workspace path (under the
    backend Layer root, from contract.connection_module), is owned by
    the database agent, and carries the fixed CONNECTION_MODULE_EXPORTS.

    If the skeleton already planned that path, its task is taken over (same
    task id, so existing dependency edges stay valid): reassigned to the
    database agent with the connection-module title/description, and its own
    imports/deps dropped (the module is self-contained). Every file that
    imported the planned module is rewired to the fixed exports.

    The backend entrypoint imports the fixed exports and depends on the
    connection-module task, so backend code can import it race-free while the
    database agent runs in parallel.
    """
    contract = arch.persistence_contract

    session_path = connection_module_path(contract)
    exports = list(CONNECTION_MODULE_EXPORTS)
    title = "DB connection module"
    purpose = (
        f"DB connection module ({contract.orm}) — engine, session factory and "
        f"session dependency, reading the URL from {contract.db_url_env_var}."
    )

    next_tasks = list(tasks)
    next_manifest = dict(manifest_files)

    skeleton_entry = next_manifest.get(session_path)
    if skeleton_entry is not None:
        tid = skeleton_entry.task_id
        for t in next_tasks:
            if t.id == tid:
                t.agent = "database"
                t.priority = AGENT_PRIORITY["database"]
                t.title = title
                t.description = purpose
                t.depends_on = []
    else:
        tid = _next_task_id("db", {t.id for t in tasks})
        next_tasks.append(Task(
            id=tid, agent="database", title=title, description=purpose,
            depends_on=[], priority=AGENT_PRIORITY["database"], status="pending",
        ))

    next_manifest[session_path] = FileManifest(
        agent="database", task_id=tid, filepath=session_path,
        purpose=purpose, exports=exports,
    )

    for fm in next_manifest.values():
        if session_path in fm.imports_from:
            fm.imports_from[session_path] = list(exports)

    main_fm = next_manifest.get(f"{BACKEND_ROOT}main.py")
    if main_fm is not None:
        main_fm.imports_from[session_path] = list(exports)
        for t in next_tasks:
            if t.id == main_fm.task_id and tid not in t.depends_on:
                t.depends_on = t.depends_on + [tid]

    return next_tasks, next_manifest


_BARREL_NAMES = frozenset({
    "__init__.py",
    "index.ts", "index.tsx", "index.js", "index.jsx",  # barrel / re-export files
})


def is_barrel_file(filepath: str | None) -> bool:
    """True for package/barrel re-export files (`__init__.py`, `index.ts|js|tsx|jsx`)."""
    return os.path.basename(filepath or "").lower() in _BARREL_NAMES


def build_barrel_content(filepath: str, manifest: Any) -> str:
    """
    Re-export content for a barrel file (`__init__.py` / `index.*`), planned as
    its Planned Content so wiring-critical files are never guessed by the LLM.

    Re-exports come from the Planner's import graph first; when the planner gave
    no graph, falls back to re-exporting same-directory sibling exports.
    """
    if not hasattr(manifest, "files"):
        return ""
    fm = manifest.files.get(filepath)
    if fm is None:
        return ""
    imports_from = getattr(fm, "imports_from", {}) or {}

    sources: list[tuple[str, list[str]]] = []
    for src_raw, names in imports_from.items():
        src = workspace_path(src_raw)
        if src not in manifest.files or src == filepath:
            continue
        sources.append((src, [str(n) for n in names if str(n).strip()]))

    if not sources:
        sibling_dir = os.path.dirname(filepath)
        for src, sfm in sorted(manifest.files.items()):
            if src == filepath or os.path.dirname(src) != sibling_dir:
                continue
            ex = [e for e in (getattr(sfm, "exports", None) or []) if str(e).strip()]
            if ex:
                sources.append((src, list(ex)))

    if not sources:
        return ""

    if filepath.endswith("__init__.py"):
        lines: list[str] = []
        for src, names in sources:
            if names:
                lines.append(f"from {dotted_import(src)} import {', '.join(names)}")
        return "\n".join(lines) + ("\n" if lines else "")

    lines = []
    for src, names in sources:
        rel = relative_import(filepath, src)
        if names:
            lines.append(f"export {{ {', '.join(names)} }} from '{rel}'")
        else:
            lines.append(f"export * from '{rel}'")
    return "\n".join(lines) + "\n"


def _model_modules(manifest_files: dict[str, FileManifest]) -> list[str]:
    """Dotted modules of the database agent's ORM model files (under a models/ package) for create-all."""
    return sorted(
        dotted_import(path) for path, fm in manifest_files.items()
        if fm.agent == "database" and "/models/" in path
        and path.endswith(".py") and not is_barrel_file(path)
    )


def _plan_content(manifest_files: dict[str, FileManifest], arch: ArchData) -> dict[str, FileManifest]:
    """
    Fix the Planned Content of every file whose content needs only planning data,
    so no coding agent sends it to the LLM:
      - barrel files (`__init__.py`, `index.*`): re-exports from the Code Manifest
      - the connection module: synthesized from the Persistence Contract, with
        create-all importing every planned ORM model module

    Runs after every injection step, so it sees the complete Code Manifest.
    """
    manifest = CodeManifest(files=manifest_files)
    for path, fm in manifest_files.items():
        if fm.planned_content is None and is_barrel_file(path):
            fm.planned_content = build_barrel_content(path, manifest)

    contract = arch.persistence_contract
    session = manifest_files.get(connection_module_path(contract))
    if session is not None:
        session.planned_content = build_connection_module(contract, _model_modules(manifest_files))
    return manifest_files


def _generate_tasks_from_skeleton(nodes: list[Any], warnings: list[str]) -> list[Task]:
    """
    Pass 2 — deterministic task generation from the skeleton (no LLM).

    depends_on is computed by topological sort over the import graph: a task depends
    on the node(s) whose FILES it imports (per imports_from). Ordering is therefore
    driven by real data dependencies, not arbitrary priority numbers.
    """
    file_to_id: dict[str, str] = {}
    for nd in nodes:
        p = str(nd.get("file", "")).strip().lstrip("/").lstrip("\\")
        if p:
            file_to_id.setdefault(p, str(nd.get("id", "")).strip())

    tasks: list[Task] = []
    for nd in nodes:
        tid = str(nd.get("id", "")).strip()
        if not tid:
            warnings.append("skeleton entry missing 'id' — skipped")
            continue
        agent = str(nd.get("agent", "")).strip().lower()
        if not agent:
            for prefix, a in (("db-", "database"), ("be-", "backend"), ("fe-", "frontend")):
                if tid.startswith(prefix):
                    agent = a
                    break
        if agent not in VALID_AGENTS:
            warnings.append(f"skeleton task '{tid}': unknown agent '{agent}' — skipped")
            continue

        depends: set[str] = set()
        imp = nd.get("imports_from") or {}
        for src_path in imp.keys():
            src = str(src_path).strip().lstrip("/").lstrip("\\")
            dep_id = file_to_id.get(src)
            if dep_id is None:
                warnings.append(
                    f"skeleton task '{tid}' imports unknown file '{src}' (phantom import) — dependency dropped"
                )
                continue
            if dep_id != tid:
                depends.add(dep_id)

        tasks.append(Task(
            id=tid, agent=agent,
            title=str(nd.get("title", "")),
            description=str(nd.get("description", "")),
            depends_on=sorted(depends),
            priority=AGENT_PRIORITY[agent], status="pending",
        ))

    valid_ids = {t.id for t in tasks}
    for t in tasks:
        bad = [d for d in t.depends_on if d not in valid_ids]
        if bad:
            warnings.append(f"task '{t.id}': depends_on references unknown IDs {bad} — removed")
            t.depends_on = [d for d in t.depends_on if d in valid_ids]

    # frontend runs parallel to backend — strip any FE->BE dependencies
    backend_ids = {t.id for t in tasks if t.agent == "backend"}
    for t in tasks:
        if t.agent != "frontend":
            continue
        be = [d for d in t.depends_on if d in backend_ids]
        if be:
            warnings.append(f"task '{t.id}' (frontend) depends on backend task(s) {be} — removed")
            t.depends_on = [d for d in t.depends_on if d not in backend_ids]

    if _has_cycle(tasks):
        warnings.append("CYCLE DETECTED in import graph — clearing all depends_on")
        logger.error("[PlannerAgent] circular dependency detected — clearing all depends_on")
        for t in tasks:
            t.depends_on = []

    return tasks


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PlannerAgent                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class PlannerAgent(BaseAgent):
    """
    Two-pass planner: one skeleton call (global visibility), then deterministic
    task generation via topological sort on the import graph.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRIES,
        retries: int | None = None,
        backoff: float = DEFAULT_BACKOFF,
        temperature: float = 0.0,
        provider: BaseLLMProvider | str | None = None,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retries=retries,
            backoff=backoff,
            temperature=temperature,
            provider=provider,
        )

    @property
    def system_prompt(self) -> str:
        return _SKELETON_SYSTEM


    # ── Main execution ───────────────────────────────────────────────────────
    def _work(self, ps: ProjectState) -> None:
        arch = ps.architecture
        lock = Lock()

        # Build the Pass-1 skeleton prompt from architecture + SRS context
        prompt = SKELETON_USER_TEMPLATE.format(
            srs_bullets         = _srs_bullets(ps.requirements),
            tech_stack          = _fmt_tech_stack(arch.tech_stack),
            api_contracts       = _fmt_api_contracts(arch.api_contracts),
            database_schema_overview = arch.database_schema_overview or "(none provided)",
            persistence_contract     = render_persistence_contract(arch.persistence_contract),
            complexity_constraint    = _complexity_constraint(getattr(arch, "complexity_tier", "STANDARD")),
        )

        # ── PASS 1: single skeleton LLM call ────────────────────────────────
        logger.info("[PlannerAgent] ── Pass 1: single skeleton call ──")
        try:
            from agents.schemas.planner import SkeletonSchema
            raw = self._call_with_system(_SKELETON_SYSTEM, prompt, "planner-skeleton", state=ps, lock=lock)
            data = self.extract_json_and_validate(raw, SkeletonSchema, state=ps, lock=lock)
        except Exception as exc:
            logger.error("[PlannerAgent] skeleton call failed: %s", exc)
            ps.log_error(self.name, f"skeleton call failed: {exc}")
            return

        skeleton = data.get("skeleton", []) if isinstance(data, dict) else []
        logger.info("[PlannerAgent] ✓ skeleton produced (%d files)", len(skeleton))
        if not skeleton:
            msg = "No skeleton produced — check LLM response."
            logger.error("[PlannerAgent] %s", msg)
            ps.log_error(self.name, msg)
            return

        # ── PASS 2: deterministic task generation (no LLM) ─────────────────
        warnings: list[str] = []
        tasks = _generate_tasks_from_skeleton(skeleton, warnings)
        for w in warnings:
            logger.warning("[PlannerAgent] %s", w)
            ps.log_error(self.name, w)

        if not tasks:
            msg = "No valid tasks produced — check LLM response."
            logger.error("[PlannerAgent] %s", msg)
            ps.log_error(self.name, msg)
            return

        layout_errors: list[str] = []
        manifest_files = _build_manifest(tasks, skeleton, arch, layout_errors)
        for e in layout_errors:
            logger.error("[PlannerAgent] %s", e)
            ps.log_error(self.name, e)

        # The skeleton's own entrypoint/connection module (e.g. under app/) take
        # the fixed paths, so the steps below never plan a second one.
        manifest_files = _adopt_skeleton_anchors(manifest_files, arch)

        # Inject mandatory wiring/scaffold tasks (Problem 2) — run last per agent.
        tasks, manifest_files = _inject_scaffold_tasks(tasks, manifest_files)

        # Commit to an API-route module whenever API contracts exist but no
        # router shape was planned (main.py must register routers that exist).
        tasks, manifest_files = _inject_router_task(tasks, manifest_files, arch)

        # Guarantee the DB connection module (fixed exports) and wire it into
        # the backend entrypoint.
        tasks, manifest_files = _inject_persistence_seam(tasks, manifest_files, arch)

        # Inject the committed project-scaffolding file set (Phase C3).
        tasks, manifest_files = _inject_project_files(tasks, manifest_files, arch)

        # Barrels and the connection module are Planned Content (never the LLM).
        manifest_files = _plan_content(manifest_files, arch)

        manifest_task_ids = {fm.task_id for fm in manifest_files.values()}
        synced_tasks = [t for t in tasks if t.id in manifest_task_ids]

        ps.tasks = TaskList(tasks=synced_tasks)
        ps.code_manifest = CodeManifest(files=manifest_files)

        counts = {a: len(ps.tasks.by_agent(a)) for a in VALID_AGENTS}
        logger.info(
            "[PlannerAgent] %d tasks — db:%d be:%d fe:%d | manifest:%d",
            len(tasks),
            counts["database"], counts["backend"], counts["frontend"],
            len(manifest_files),
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Quick smoke-test  (run:  python -m agents.planner)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PlannerAgent — local smoke-test (no LLM call)")
    print("=" * 60)

    from core.state import Task, FileManifest, CodeManifest

    def _make_task(tid, agent="backend", deps=None):
        return Task(id=tid, agent=agent, title=tid, description="",
                    depends_on=deps or [], priority=2)

    # ── _has_cycle ──────────────────────────────────────────────────────────────
    assert not _has_cycle([
        _make_task("a"), _make_task("b", deps=["a"]), _make_task("c", deps=["b"]),
    ]), "FAIL: false positive cycle"
    print("[PASS] _has_cycle — linear chain: no cycle")

    t1, t2, t3 = _make_task("x"), _make_task("y", deps=["x"]), _make_task("z", deps=["y"])
    t1.depends_on = ["z"]
    assert _has_cycle([t1, t2, t3]), "FAIL: cycle not detected"
    print("[PASS] _has_cycle — circular deps: cycle detected")

    # ── _validate_tasks ─────────────────────────────────────────────────────────
    raw = [
        {"id": "db-001", "agent": "database", "title": "Users table",
         "description": "Create users migration.", "depends_on": [], "status": "pending"},
        {"id": "be-001", "agent": "backend",  "title": "Auth routes",
         "description": "POST /auth/login.",       "depends_on": ["db-001"], "status": "pending"},
        {"id": "fe-001", "agent": "frontend", "title": "Login page",
         "description": "React login form.",       "depends_on": ["be-001"], "status": "pending"},
        {"id": "be-001", "agent": "backend",  "title": "Dup",
         "description": "Should be skipped.",      "depends_on": [], "status": "pending"},
        {"id": "xx-001", "agent": "wizard",   "title": "Bad agent",
         "description": "Should be skipped.",      "depends_on": [], "status": "pending"},
    ]
    tasks, warnings = _validate_tasks(raw)
    assert len(tasks) == 3
    assert len(warnings) >= 2
    fe = next(t for t in tasks if t.agent == "frontend")
    assert "be-001" not in fe.depends_on
    print("[PASS] _validate_tasks — 3 tasks, FE->BE dep stripped, dup/bad-agent dropped")

    for t in tasks:
        assert t.priority == AGENT_PRIORITY[t.agent]
    print("[PASS] Priority enforcement — all tasks match AGENT_PRIORITY")

    # ── _extract_table_names ────────────────────────────────────────────────────
    schema = "users (id UUID PK)\n\nworkspaces (id UUID PK)\n\ntasks (id UUID PK)"
    result = _extract_table_names(schema)
    assert "db-001: users" in result
    assert "db-002: workspaces" in result
    assert "db-003: tasks" in result
    print("[PASS] _extract_table_names — 3 tables extracted with db-NNN IDs")

    # ── _build_manifest ────────────────────────────────────────────────────────
    test_tasks = [
        Task(id="db-001", agent="database", title="T1", description="", depends_on=[], priority=1),
        Task(id="be-001", agent="backend", title="T2", description="", depends_on=["db-001"], priority=2),
    ]
    test_raw = [
        {"id": "db-001", "file": "database/migrations/001_users.sql", "exports": [], "description": "Users table"},
        {"id": "be-001", "file": "backend/app/models/user.py", "exports": ["User"], "description": "User model"},
    ]
    mf = _build_manifest(test_tasks, test_raw)
    assert len(mf) == 2
    assert mf["output/backend/app/models/user.py"].exports == ["User"]
    print("[PASS] _build_manifest — 2 entries with correct exports")

    # ── _populate_contracts (Phase 2) ──────────────────────────────────────────
    from core.state import ArchData, APIContract
    arch_c = ArchData(
        database_schema_overview=(
            "users (id UUID PK, email VARCHAR UNIQUE)\n"
            "todos (id UUID PK, title VARCHAR, description TEXT, is_completed BOOLEAN, user_id UUID)\n"
        ),
        api_contracts=[
            APIContract(method="POST", path="/api/todos",
                        request_body={"title": "str", "description": "str"},
                        response_body={"id": "UUID", "title": "str", "description": "str", "is_completed": "bool"}),
        ],
    )
    ct_tasks = [
        Task(id="db-001", agent="database", title="T1", description="", depends_on=[], priority=1),
        Task(id="db-002", agent="database", title="T2", description="", depends_on=[], priority=1),
        Task(id="be-001", agent="backend", title="T3", description="", depends_on=["db-001"], priority=2),
    ]
    ct_raw = [
        {"id": "db-001", "file": "app/models/todo.py", "exports": ["TodoItem"], "imports_from": {}},
        {"id": "db-002", "file": "app/models/user.py", "exports": ["User"], "imports_from": {}},
        {"id": "be-001", "file": "app/schemas/todo.py", "exports": ["TodoCreate", "TodoResponse"], "imports_from": {}},
    ]
    mfc = _build_manifest(ct_tasks, ct_raw, arch_c)
    assert mfc["output/backend/app/models/todo.py"].contract["TodoItem"]["is_completed"] == "BOOLEAN"
    assert mfc["output/backend/app/models/todo.py"].contract["TodoItem"]["user_id"] == "UUID"
    assert mfc["output/backend/app/models/user.py"].contract == {
        "User": {"id": "UUID PK", "email": "VARCHAR UNIQUE"}}
    assert mfc["output/backend/app/schemas/todo.py"].contract["TodoCreate"] == {"title": "str", "description": "str"}
    assert mfc["output/backend/app/schemas/todo.py"].contract["TodoResponse"]["is_completed"] == "bool"
    assert _match_entity("Workspace", {"todos": {}, "users": {}}) is None
    print("[PASS] _populate_contracts — DB model + schema field contracts populated deterministically")

    # ── _generate_tasks_from_skeleton (Pass 2) ────────────────────────────────
    skel = [
        {"id": "db-001", "agent": "database", "title": "Todo model",
         "file": "app/models/todo.py", "exports": ["TodoItem"], "imports_from": {}},
        {"id": "be-002", "agent": "backend", "title": "Todo schemas",
         "file": "app/schemas/todo.py", "exports": ["TodoItemCreate"], "imports_from": {}},
        {"id": "be-001", "agent": "backend", "title": "Todo routes",
         "file": "app/api/routes/todo.py", "exports": ["router"],
         "imports_from": {"app/models/todo.py": ["TodoItem"], "app/schemas/todo.py": ["TodoItemCreate"]}},
        {"id": "fe-001", "agent": "frontend", "title": "Todo list",
         "file": "assets/js/todoList.js", "exports": ["TodoList"], "imports_from": {}},
        # Phantom import — source file never appears in the skeleton
        {"id": "be-003", "agent": "backend", "title": "Phantom svc",
         "file": "app/services/phantom.py", "exports": ["srv"],
         "imports_from": {"app/models/nonexistent.py": ["Missing"]}},
    ]
    warnings = []
    gen = _generate_tasks_from_skeleton(skel, warnings)
    by_id = {t.id: t for t in gen}
    assert len(gen) == 5, f"expected 5 tasks, got {len(gen)}"
    assert by_id["be-001"].depends_on == ["be-002", "db-001"], by_id["be-001"].depends_on
    assert by_id["be-003"].depends_on == [], "phantom import should be dropped"
    assert any("phantom import" in w for w in warnings), "phantom import should be warned"
    assert by_id["be-001"].priority == AGENT_PRIORITY["backend"]
    print("[PASS] _generate_tasks_from_skeleton — import-graph deps + phantom drop")

    # FE->BE deps are stripped (frontend runs parallel to backend)
    skel2 = [
        {"id": "be-001", "agent": "backend", "title": "r", "file": "app/api/r.py",
         "exports": ["router"], "imports_from": {}},
        {"id": "fe-001", "agent": "frontend", "title": "f", "file": "assets/js/f.js",
         "exports": ["App"], "imports_from": {"app/api/r.py": ["router"]}},
    ]
    gen2 = _generate_tasks_from_skeleton(skel2, [])
    fe = next(t for t in gen2 if t.agent == "frontend")
    assert fe.depends_on == [], f"FE->BE dep not stripped: {fe.depends_on}"
    print("[PASS] _generate_tasks_from_skeleton — FE->BE dep stripped")

    # SkeletonSchema validates imports_from
    from agents.schemas.planner import SkeletonSchema
    ss = SkeletonSchema.model_validate({"skeleton": skel[:2]})
    assert len(ss.skeleton) == 2 and ss.skeleton[0].imports_from == {}
    print("[PASS] SkeletonSchema — parses skeleton + imports_from")

    # ── FileManifest / CodeManifest ─────────────────────────────────────────────
    fm = FileManifest(agent="backend", task_id="be-001", filepath="backend/app/routes/auth.py",
                      purpose="JWT auth routes", exports=["router", "login"])
    assert fm.written is False and fm.exports == ["router", "login"]
    cm = CodeManifest(files={fm.filepath: fm})
    cm.mark_written("backend/app/routes/auth.py")
    assert cm.files["backend/app/routes/auth.py"].written is True
    assert cm.files["backend/app/routes/auth.py"].exports == ["router", "login"]
    print("[PASS] FileManifest/CodeManifest — planner exports preserved after mark_written")

    # ── _inject_router_task ─────────────────────────────────────────────────────
    from core.state import ArchData, APIContract
    arch_w_contracts = ArchData(api_contracts=[
        APIContract(method="GET", path="/api/todos", description="List"),
        APIContract(method="POST", path="/api/todos", description="Create"),
    ])
    tasks_m = [_make_task("be-001", agent="backend")]
    mf_m = {
        "output/backend/services/todo.py": FileManifest(
            agent="backend", task_id="be-001", filepath="output/backend/services/todo.py",
            purpose="svc", exports=["svc"]),
    }
    t2, m2 = _inject_scaffold_tasks(tasks_m, mf_m)
    t2, m2 = _inject_router_task(t2, m2, arch_w_contracts)
    router_file = next(p for p in m2 if "/api/routes/" in p)
    assert router_file == "output/backend/api/routes/todos.py", router_file
    assert m2[router_file].exports == ["router"]
    main_fm = m2["output/backend/main.py"]
    main_task = next(t for t in t2 if t.id == main_fm.task_id)
    router_tid = m2[router_file].task_id
    assert router_tid in main_task.depends_on, main_task.depends_on
    router_task = next(t for t in t2 if t.id == router_tid)
    assert "be-001" in router_task.depends_on
    assert "/api/todos" in router_task.description and "router" in m2[router_file].exports
    print("[PASS] _inject_router_task — router committed w/ contracts, main.py rewired to depend on it")

    mf_has_router = dict(mf_m)
    mf_has_router["output/backend/api/routes/auth.py"] = FileManifest(
        agent="backend", task_id="be-999", filepath="output/backend/api/routes/auth.py",
        purpose="auth routes", exports=["router"])
    t3, m3 = _inject_router_task(tasks_m, mf_has_router, arch_w_contracts)
    assert set(m3) == set(mf_has_router), "no injection when a router shape already exists"
    t4, m4 = _inject_router_task(tasks_m, mf_m, ArchData(api_contracts=[]))
    assert set(m4) == set(mf_m), "no injection without API contracts"
    print("[PASS] _inject_router_task — skipped when router exists or contracts absent")

    # ── _inject_persistence_seam + deterministic .env.example ─────────────────
    from core.state import build_persistence_contract
    arch_db = ArchData(
        tech_stack={"database": "SQLite 3"},
        persistence_contract=build_persistence_contract("SQLite 3"),
    )
    t5, m5 = _inject_scaffold_tasks(tasks_m, mf_m)
    t5, m5 = _inject_persistence_seam(t5, m5, arch_db)
    t5, m5 = _inject_project_files(t5, m5, arch_db)
    session_fm = m5["output/backend/db/session.py"]
    assert session_fm.agent == "database"
    assert session_fm.exports == list(CONNECTION_MODULE_EXPORTS)
    main_fm = m5["output/backend/main.py"]
    assert main_fm.imports_from["output/backend/db/session.py"] == list(CONNECTION_MODULE_EXPORTS)
    assert session_fm.task_id in next(t for t in t5 if t.id == main_fm.task_id).depends_on
    assert "DATABASE_URL=sqlite" in m5["output/backend/.env.example"].planned_content
    print("[PASS] _inject_persistence_seam — connection module pinned, main.py wired, .env.example deterministic")

    t6, m6 = _inject_persistence_seam(tasks_m, mf_m, ArchData())
    assert m6["output/backend/db/session.py"].agent == "database", "default contract → connection module"
    _, m6 = _inject_project_files(t6, m6, ArchData())
    assert "DATABASE_URL=postgresql" in m6["output/backend/.env.example"].planned_content
    print("[PASS] _inject_persistence_seam — no named database still gets the default contract's seam")

    print("\n" + "=" * 60)
    print("All PlannerAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
