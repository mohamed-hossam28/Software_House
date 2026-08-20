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
import time
from threading import Lock
from typing import Any

from .base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF
from providers.llm.base import BaseLLMProvider
from core.state import (
    ProjectState, Task, TaskList, FileManifest, CodeManifest, APIContract,
)
import re

logger = logging.getLogger(__name__)

VALID_AGENTS: frozenset[str] = frozenset({"database", "backend", "frontend", "devops"})
AGENT_PRIORITY: dict[str, int] = {"database": 1, "backend": 2, "frontend": 2, "devops": 3}
AGENT_PREFIX: dict[str, str] = {"database": "db", "backend": "be", "frontend": "fe", "devops": "do"}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Pass-1 system + user prompts                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_SKELETON_SYSTEM = """\
You are a senior engineering project manager. Design the COMPLETE file structure for the ENTIRE
project in a single response. You have GLOBAL visibility over every file, so you MUST list every
file — including each file that any other file imports from.

Respond with a SINGLE JSON object — no prose outside it:
{"skeleton": [
  {"id":"<ID>","agent":"<database|backend|frontend|devops>","title":"<short>",
   "description":"<what this file does>","file":"<repo-relative path>",
   "exports":["<name>", ...],
   "imports_from":{"<source file path>":["<name>", ...], ...}}
]}

Rules:
- ID prefix by owner agent: db-*, be-*, fe-*, do-* (e.g. be-001).
- exports: ONLY cross-module names — what OTHER files import from THIS file.
- imports_from: the exact source file path(s) this file imports from, and which names it imports.
  This is how the system computes build order — be precise and exhaustive.
- EVERY file referenced in imports_from MUST itself appear in the skeleton. No phantom imports.
- Include entrypoint/wiring files: backend main.py (registers routers) and frontend index.html.
- Unique, non-empty, repo-relative file paths.
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
    """Prompt-level constraint calibrating file count to user intent (Problem 4)."""
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


def _build_manifest(tasks: list[Task], raw_list: list[Any], arch: Any = None) -> dict[str, FileManifest]:
    """Build CodeManifest entries from validated tasks + raw LLM dicts."""
    raw_by_id: dict[str, dict] = {}
    for item in raw_list:
        if isinstance(item, dict) and item.get("id"):
            raw_by_id[str(item["id"]).strip()] = item

    db_model_dir = "app/models/"
    if arch is not None and getattr(arch, "tech_stack", None):
        from core.state import resolve_db_tooling
        db_model_dir = resolve_db_tooling(arch.tech_stack.get("database", "")).get("model_dir", "app/models/")

    manifest_files: dict[str, FileManifest] = {}
    for task in tasks:
        raw_item  = raw_by_id.get(task.id, {})
        file_path = str(raw_item.get("file", "")).strip().lstrip("/").lstrip("\\")

        if not file_path:
            # Fallback file path if LLM omitted 'file' field
            sanitized_title = re.sub(r'[^a-zA-Z0-9_]', '_', task.title.lower()).strip('_') or f"task_{task.id}"
            if task.agent == "database":
                folder, ext = db_model_dir, ".py"
            elif task.agent == "backend":
                folder, ext = "app/api", ".py"
            else:
                folder, ext = "src", ".js"
            file_path = f"{folder}{sanitized_title}{ext}"
            logger.info("[PlannerAgent] task '%s' missing 'file' field — created fallback path: %s", task.id, file_path)

        if not file_path.startswith("output/") and not file_path.startswith("output\\"):
            file_path = f"output/{file_path}"

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
            src = str(src_path).strip().lstrip("/").lstrip("\\")
            if not src.startswith("output/") and not src.startswith("output\\"):
                src = f"output/{src}"
            src = src.replace("\\", "/")
            if src == fm.filepath or src not in manifest_files:
                if src not in manifest_files:
                    logger.warning(
                        "[PlannerAgent] task '%s' imports unknown file '%s' (phantom import) — dropped from import contract",
                        task.id, src,
                    )
                continue
            imports_from[src] = [str(n) for n in names if str(n).strip()]
        fm.imports_from = imports_from

    return manifest_files


def _detect_backend_root(manifest_files: dict[str, FileManifest]) -> str | None:
    """Return the common parent directory of all backend files, e.g. 'output/backend/'."""
    backend_paths = [fm.filepath for fm in manifest_files.values() if fm.agent == "backend"]
    if not backend_paths:
        return None
    parts_list = [p.rstrip("/").split("/") for p in backend_paths]
    common = parts_list[0][:-1]  # exclude filename
    for p in parts_list[1:]:
        common = [a for a, b in zip(common, p[:-1]) if a == b]
    if not common:
        return None
    root = "/".join(common) + "/"
    # If all backend files live in the same directory, go up one level
    # so that main.py sits alongside the subdirectory, not inside it.
    dirs = {p.rsplit("/", 1)[0] + "/" for p in backend_paths}
    if len(dirs) <= 1:
        root_parts = root.rstrip("/").split("/")
        if len(root_parts) > 1:
            root = "/".join(root_parts[:-1]) + "/"
    return root


def _detect_frontend_root(manifest_files: dict[str, FileManifest]) -> str | None:
    """Return the common parent directory of all frontend files, e.g. 'output/frontend/'."""
    frontend_paths = [fm.filepath for fm in manifest_files.values() if fm.agent == "frontend"]
    if not frontend_paths:
        return None
    parts_list = [p.rstrip("/").split("/") for p in frontend_paths]
    common = parts_list[0][:-1]
    for p in parts_list[1:]:
        common = [a for a, b in zip(common, p[:-1]) if a == b]
    if not common:
        return None
    root = "/".join(common) + "/"
    dirs = {p.rsplit("/", 1)[0] + "/" for p in frontend_paths}
    if len(dirs) <= 1:
        root_parts = root.rstrip("/").split("/")
        if len(root_parts) > 1:
            root = "/".join(root_parts[:-1]) + "/"
    return root


def _inject_scaffold_tasks(
    tasks: list[Task],
    manifest_files: dict[str, FileManifest],
) -> tuple[list[Task], dict[str, FileManifest]]:
    """
    Add mandatory wiring/scaffold tasks (Problem 2):
      - backend  → <backend_root>/main.py  (registers every backend router)
      - frontend → <frontend_root>/index.html   (loads every stylesheet + script)
    Paths are inferred from the skeleton's existing file structure so that
    entrypoints always land in the same directory tree as the rest of the code.
    Each scaffold depends on ALL of its own agent's tasks so it runs last in that
    agent's execution order and sees the complete list of files to wire.
    """
    existing_ids = {t.id for t in tasks}
    existing_paths = set(manifest_files)
    next_tasks = list(tasks)
    next_manifest = dict(manifest_files)

    def _next_id(prefix: str) -> str:
        n = 900
        while f"{prefix}-{n:03d}" in existing_ids:
            n += 1
        return f"{prefix}-{n:03d}"

    # Infer root directories from the skeleton's actual file paths
    backend_root = _detect_backend_root(manifest_files) or "output/app/"
    frontend_root = _detect_frontend_root(manifest_files) or "output/"

    be_main = f"{backend_root}main.py"
    fe_index = f"{frontend_root}index.html"

    backend_task_ids = [t.id for t in tasks if t.agent == "backend"]
    if backend_task_ids and be_main not in existing_paths:
        tid = _next_id("be")
        existing_ids.add(tid)
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
        tid = _next_id("fe")
        existing_ids.add(tid)
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
    FILE_SET / import checks in the integration agent.
    """
    if not arch or not getattr(arch, "api_contracts", None):
        return tasks, manifest_files
    backend_task_ids = [t.id for t in tasks if t.agent == "backend"]
    if not backend_task_ids:
        return tasks, manifest_files

    backend_root = _detect_backend_root(manifest_files) or "output/backend/"
    be_prefix = backend_root.rstrip("/") + "/"

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

    def _next_id(prefix: str) -> str:
        n = 900
        while f"{prefix}-{n:03d}" in existing_ids:
            n += 1
        return f"{prefix}-{n:03d}"

    # Module name from the first contract path segment (e.g. /api/todos → todos).
    module_name = ""
    for c in arch.api_contracts:
        segs = [s for s in c.path.strip("/").split("/") if s and s != "api"]
        cand = re.sub(r"[^a-zA-Z0-9_]", "", segs[0]) if segs else ""
        if cand:
            module_name = cand
            break
    module_name = module_name or "api"

    router_path = f"{backend_root}api/routes/{module_name}.py"
    if router_path in next_manifest:
        return tasks, manifest_files

    contracts_block = "\n".join(
        f"- {c.method.upper()} {c.path}  "
        f"request={c.request_body or None}  response={c.response_body or None}"
        for c in arch.api_contracts
    )
    tid = _next_id("be")
    existing_ids.add(tid)
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
    be_main = f"{backend_root}main.py"
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
) -> tuple[list[Task], dict[str, FileManifest], list[str]]:
    """
    Commit to the deliverable's project-scaffolding file set (Phase C3).
    These files are REQUIRED for a runnable deliverable and the integration
    agent verifies they exist and are written (FILE_SET check):
      - output/requirements.txt          (Python dependency manifest)
      - output/README.md                 (run instructions)
      - <backend_root>.env.example       (environment template — no real secrets)
    """
    existing_ids = {t.id for t in tasks}
    next_tasks = list(tasks)
    next_manifest = dict(manifest_files)

    backend_root = _detect_backend_root(manifest_files) or "output/backend/"

    def _next_id(prefix: str) -> str:
        n = 900
        while f"{prefix}-{n:03d}" in existing_ids:
            n += 1
        return f"{prefix}-{n:03d}"

    required_files: list[str] = []
    specs = [
        (
            "output/requirements.txt",
            "Python dependencies (requirements.txt)",
            "Locked Python dependency manifest covering every third-party import used by the backend.",
            ["requirements"],
        ),
        (
            "output/README.md",
            "Project README (README.md)",
            "Run instructions: install dependencies, start the backend, and open the frontend.",
            ["readme"],
        ),
        (
            f"{backend_root}.env.example",
            "Environment template (.env.example)",
            "Example environment variables (DB URL, ports, secrets) — never real secrets.",
            ["env_example"],
        ),
    ]
    for path, title, purpose, exports in specs:
        if path in next_manifest:
            continue
        tid = _next_id("be")
        existing_ids.add(tid)
        next_tasks.append(Task(
            id=tid, agent="backend", title=title, description=purpose,
            depends_on=[], priority=AGENT_PRIORITY["backend"], status="pending",
        ))
        next_manifest[path] = FileManifest(
            agent="backend", task_id=tid, filepath=path,
            purpose=purpose, exports=exports,
        )
        required_files.append(path)

    return next_tasks, next_manifest, required_files


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
            for prefix, a in (("db-", "database"), ("be-", "backend"), ("fe-", "frontend"), ("do-", "devops")):
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
    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)
        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()
        arch = ps.architecture
        lock = Lock()

        # Build the Pass-1 skeleton prompt from architecture + SRS context
        prompt = SKELETON_USER_TEMPLATE.format(
            srs_bullets         = _srs_bullets(ps.requirements),
            tech_stack          = _fmt_tech_stack(arch.tech_stack),
            api_contracts       = _fmt_api_contracts(arch.api_contracts),
            database_schema_overview = arch.database_schema_overview or "(none provided)",
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
            ps.metrics.agent_durations[self.name] = round(time.perf_counter() - t0, 3)
            return self._compute_diff(before, ps)

        skeleton = data.get("skeleton", []) if isinstance(data, dict) else []
        logger.info("[PlannerAgent] ✓ skeleton produced (%d files)", len(skeleton))
        if not skeleton:
            msg = "No skeleton produced — check LLM response."
            logger.error("[PlannerAgent] %s", msg)
            ps.log_error(self.name, msg)
            ps.metrics.agent_durations[self.name] = round(time.perf_counter() - t0, 3)
            return self._compute_diff(before, ps)

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
            ps.metrics.agent_durations[self.name] = round(time.perf_counter() - t0, 3)
            return self._compute_diff(before, ps)

        manifest_files = _build_manifest(tasks, skeleton, arch)

        # Inject mandatory wiring/scaffold tasks (Problem 2) — run last per agent.
        tasks, manifest_files = _inject_scaffold_tasks(tasks, manifest_files)

        # Commit to an API-route module whenever API contracts exist but no
        # router shape was planned (main.py must register routers that exist).
        tasks, manifest_files = _inject_router_task(tasks, manifest_files, arch)

        # Inject the committed project-scaffolding file set (Phase C3) and
        # record it on state so the integration agent can enforce it (FILE_SET).
        tasks, manifest_files, required_files = _inject_project_files(tasks, manifest_files)

        manifest_task_ids = {fm.task_id for fm in manifest_files.values()}
        synced_tasks = [t for t in tasks if t.id in manifest_task_ids]

        ps.tasks = TaskList(tasks=synced_tasks)
        ps.code_manifest = CodeManifest(files=manifest_files)
        ps.required_files = required_files

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)
        counts = {a: len(ps.tasks.by_agent(a)) for a in VALID_AGENTS}
        logger.info(
            "[PlannerAgent] %d tasks in %.2fs — db:%d be:%d fe:%d do:%d | manifest:%d",
            len(tasks), duration,
            counts["database"], counts["backend"], counts["frontend"], counts["devops"],
            len(manifest_files),
        )
        return self._compute_diff(before, ps)


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

    print("\n" + "=" * 60)
    print("All PlannerAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
