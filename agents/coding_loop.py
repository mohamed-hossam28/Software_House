"""
agents/coding_loop.py
─────────────────────
The coding loop: turns a Coding agent's File Manifests into written, recorded
files. The three Coding agents (database, backend, frontend) are thin graph
nodes, each a name plus a CodingProfile; everything else lives here.

The loop's steps, per File Manifest:
  1. staging: the agent's tasks are grouped into dependency-resolved stages
     (by priority first for profiles that ask for it); a stage runs in parallel
  2. a File Manifest with Planned Content is written verbatim, without the LLM
  3. otherwise the request context is built (request header + file contract +
     the profile's own sections + Import Contract + pending contracts + tree)
  4. the LLM is called (with retries)
  5. the reply's JSON is extracted and validated against the profile's
     response schema
  6. the code is written through the workspace's write guard
  7. the file is recorded once through the state-update module, marking its
     File Manifest Written

The coding request header — the fixed first lines of every coding request,
naming the file(s) asked for — is defined here too, with its parser.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Iterable, Optional

from agents.base import BaseAgent
from agents.utils import parse_metadata, resolve_stages
from core.state import ArchData, CodeManifest, FileManifest, ProjectState, Task
from core.workspace import dotted_import, relative_import, workspace_path

logger = logging.getLogger(__name__)

#: Concurrent LLM calls per stage.
MAX_PARALLEL_CALLS = 3
#: Max files per batch LLM call.
TRIVIAL_BATCH_SIZE = 3
#: Prompt size caps — keep the request body under typical gateway payload limits.
MAX_TREE_ENTRIES = 80       # project file tree entries shown in coding prompts
MAX_MANIFEST_ENTRIES = 25   # dependency/manifest entries per prompt


# ══════════════════════════════════════════════════════════════════════════════
#  The coding request header
# ══════════════════════════════════════════════════════════════════════════════
#
#     <<<coding-request>>>
#     file: output/backend/app/api/routes/todo.py
#     <<<end-coding-request>>>
#
# One `file:` line per file asked for, in order. A reader (the scripted fake LLM)
# knows which File Manifests a request is about without reading its prose.

HEADER_OPEN = "<<<coding-request>>>"
HEADER_CLOSE = "<<<end-coding-request>>>"
FILE_PREFIX = "file: "


def request_header(filepaths: Iterable[str]) -> str:
    """The header naming *filepaths*, followed by a blank line, to prepend to a coding request."""
    paths = list(filepaths)
    if not paths:
        raise ValueError("a coding request header names at least one file")
    for path in paths:
        if not path or path != path.strip() or "\n" in path:
            raise ValueError(f"not a header-safe file path: {path!r}")
    lines = [HEADER_OPEN, *(FILE_PREFIX + path for path in paths), HEADER_CLOSE]
    return "\n".join(lines) + "\n\n"


def parse_request_header(request: str) -> list[str]:
    """
    The file paths named by the header *request* starts with, in order.

    Returns [] when *request* does not start with a header (it is not a coding
    request). Raises ValueError when it starts with a malformed header.
    """
    lines = request.split("\n")
    if lines[0] != HEADER_OPEN:
        return []
    paths: list[str] = []
    for line in lines[1:]:
        if line == HEADER_CLOSE:
            if not paths:
                raise ValueError("coding request header names no file")
            return paths
        if not line.startswith(FILE_PREFIX):
            raise ValueError(f"malformed coding request header line: {line!r}")
        paths.append(line[len(FILE_PREFIX):])
    raise ValueError("coding request header is never closed")


# ══════════════════════════════════════════════════════════════════════════════
#  The profile
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FileContext:
    """What a profile's prompt sections may read about the file being generated."""
    filepath: str
    fm: FileManifest
    task: Optional[Task]
    manifest: CodeManifest
    arch: ArchData


@dataclass(frozen=True)
class CodingProfile:
    """
    Everything that differs between the Coding agents, as plain data.

    agent            — the owning agent key: tasks, File Manifests and the
                       generated-code bucket are looked up under it
    code_key         — the reply key holding the file's source
    response_schema  — the pydantic model every reply (or batch item) must match
    system_prompt    — the system prompt, rendered from the architecture
    sections         — the agent-specific sections of a file's request context
    tech             — the technology named in batch requests
    default_return_type — return type recorded for functions that name none
    stage_by_priority   — stage by task priority before dependencies
    """
    agent: str
    code_key: str
    response_schema: type
    system_prompt: Callable[[ArchData], str]
    sections: Callable[[FileContext], list[str]]
    tech: Callable[[ArchData], str]
    default_return_type: str = "None"
    stage_by_priority: bool = False

    @property
    def reply_keys(self) -> list[str]:
        """The response schema's keys, code key first."""
        fields = list(getattr(self.response_schema, "model_fields", {}) or [self.code_key])
        return [self.code_key] + [f for f in fields if f != self.code_key]


# ══════════════════════════════════════════════════════════════════════════════
#  Contracts shared with the repair loop
# ══════════════════════════════════════════════════════════════════════════════

def build_import_contract(manifest: Any, filepath: str) -> str:
    """
    Deterministic IMPORT CONTRACT for *filepath*: the exact import lines it may
    use, computed from the Planner's import graph (imports_from) + manifest
    exports. Never guesses — every source must exist in the manifest.

    Returns an empty string when the file declares no internal imports.
    """
    if not hasattr(manifest, "files"):
        return ""
    fm = manifest.files.get(filepath)
    if fm is None:
        return ""
    imports_from = getattr(fm, "imports_from", {}) or {}

    lines: list[str] = []
    missing: list[str] = []
    for src_raw, names in imports_from.items():
        src = workspace_path(src_raw)
        if src not in manifest.files or src == filepath:
            if src not in manifest.files:
                missing.append(src)
            continue
        names = [str(n) for n in names if str(n).strip()]
        if filepath.endswith(".py"):
            mod = dotted_import(src)
            if names:
                lines.append(f"from {mod} import {', '.join(names)}")
            else:
                lines.append(f"import {mod}")
        else:
            rel = relative_import(filepath, src)
            if names:
                lines.append(f"import {{ {', '.join(names)} }} from '{rel}'")
            else:
                lines.append(f"import '{rel}'")

    if not lines:
        return ""

    header = (
        "## IMPORT CONTRACT (deterministic — use EXACTLY these imports)\n"
        "Every line below is derived from the planner's import graph. Do NOT invent "
        "other internal imports, modules, or paths.\n"
    )
    body = "\n".join(f"- `{l}`" for l in lines)
    note = "\nUse exactly these import statements in your code (same modules, same names)."
    if missing:
        note += (
            "\nNote: the following planned sources do not exist in the manifest and "
            "were omitted: " + ", ".join(missing)
        )
    return header + body + note


def filter_api_contracts_by_relevance(
    contracts: list[Any],
    filepath: str,
    purpose: str = "",
    task_title: str = "",
    top_k: int = 3,
) -> list[Any]:
    """Filter and rank API contracts by relevance score using Jaccard keyword similarity."""
    if not contracts:
        return []
    if len(contracts) <= top_k:
        return contracts

    search_words = set(re.findall(r"[a-zA-Z0-9]+", f"{filepath} {purpose} {task_title}".lower()))
    if not search_words:
        return contracts[:top_k]

    def score_contract(contract: Any) -> float:
        if isinstance(contract, dict):
            c_text = f"{contract.get('path', '')} {contract.get('description', '')}"
        else:
            c_text = f"{getattr(contract, 'path', '')} {getattr(contract, 'description', '')}"
        c_words = set(re.findall(r"[a-zA-Z0-9]+", c_text.lower()))
        if not c_words:
            return 0.0
        intersection = len(search_words & c_words)
        if not intersection:
            return 0.0
        return intersection / len(search_words | c_words)

    scored = [(score_contract(c), c) for c in contracts]
    scored.sort(key=lambda x: x[0], reverse=True)
    threshold = 0.05
    filtered = [c for score, c in scored if score >= threshold]
    return filtered[:top_k] if filtered else [c for _, c in scored[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
#  Batching (private)
# ══════════════════════════════════════════════════════════════════════════════

#: Filename patterns that are almost always trivial (< 20 lines, no bespoke logic)
_TRIVIAL_NAMES = frozenset({
    "__init__.py",
    "index.ts", "index.tsx", "index.js", "index.jsx",  # barrel / re-export files
    "types.ts", "types.tsx",                            # pure type declarations
    "constants.ts", "constants.js",                     # constant modules
    "config.py", "config.ts", "config.js",              # app config stubs
    "settings.py",
    "exceptions.py", "errors.py", "errors.ts",          # exception/error stubs
    "utils.py",                                         # small utility stubs
})

_TRIVIAL_PATTERNS = (
    "/__init__.py",
    "/index.",          # barrel files under any directory
    "/types.",
    "/constants.",
    "/exceptions.",
    "/errors.",
    "/config.",
    "/settings.",
)

#: Entrypoint/scaffold files get their wiring context in a single-file request;
#: they are NEVER batched.
_SCAFFOLD_NAMES = frozenset({"main.py", "app.py", "server.py", "index.html", "index.htm"})


def _is_scaffold_file(filepath: str | None) -> bool:
    return os.path.basename(filepath or "").lower() in _SCAFFOLD_NAMES


def _is_trivial_filepath(filepath: str, exports: list | None = None) -> bool:
    """True if a file is 'trivial' boilerplate (init, barrel, config, types, constants, errors, <3 exports)."""
    basename = os.path.basename(filepath or "").lower()
    if basename in _TRIVIAL_NAMES:
        return True
    for pattern in _TRIVIAL_PATTERNS:
        if pattern in (filepath or "").lower():
            return True
    if exports is not None and len(exports) < 3:
        return True
    return False


def _is_batchable(fm: FileManifest | None) -> bool:
    """A File Manifest goes into a batch call: trivial, not a scaffold, no Planned Content."""
    if fm is None or fm.planned_content is not None:
        return False
    return _is_trivial_filepath(fm.filepath, fm.exports or []) and not _is_scaffold_file(fm.filepath)


def _align_batch_items(items: list[dict], fms: list[FileManifest]) -> list[tuple[dict, FileManifest]] | None:
    """
    Pair batch reply items with their File Manifests.

    Prefers filepath-based matching (the LLM echoes each filepath), falls back to
    order-based pairing. Returns None on a count mismatch so the caller can fall
    back to per-file generation instead of silently misassigning code.
    """
    if not items or len(items) != len(fms):
        return None
    by_path: dict[str, dict] = {}
    for it in items:
        fp = str(it.get("filepath", "") or "").strip() if isinstance(it, dict) else ""
        if fp:
            by_path[fp] = it
    if len(by_path) == len(items):
        aligned: list[tuple[dict, FileManifest]] = []
        for fm in fms:
            it = by_path.get(fm.filepath)
            if it is None:
                return None
            aligned.append((it, fm))
        return aligned
    return list(zip(items, fms))


# ══════════════════════════════════════════════════════════════════════════════
#  Context building (private)
# ══════════════════════════════════════════════════════════════════════════════

def _filter_manifest_by_dependencies(manifest: Any, filepath: str, task: Any = None) -> dict[str, Any]:
    """The File Manifests the given file/task actually depends on."""
    if not hasattr(manifest, "files") or not manifest.files:
        return {}
    if not manifest.files.get(filepath):
        return {}

    dep_task_ids = set(task.depends_on) if (task and getattr(task, "depends_on", None)) else set()
    filtered: dict[str, Any] = {}
    if dep_task_ids:
        for path, file_m in manifest.files.items():
            if path == filepath:
                continue
            if getattr(file_m, "task_id", None) in dep_task_ids or path in dep_task_ids:
                filtered[path] = file_m

    if not filtered:
        target_token = os.path.basename(filepath).split('.')[0].lower()
        for path, file_m in manifest.files.items():
            if path == filepath:
                continue
            path_token = os.path.basename(path).split('.')[0].lower()
            if getattr(file_m, "written", False) or (
                path_token and (path_token in target_token or target_token in path_token)
            ):
                filtered[path] = file_m
    return filtered


def _build_pending_contracts(manifest: Any, filepath: str) -> str:
    """
    PENDING CONTRACTS section for *filepath*: for each not-yet-written dependency
    that carries a planner-derived field contract (DB model or schema file), its
    planned exports + fields, so the file matches them while that dependency is
    generated in parallel. Returns "" when nothing applies.
    """
    if not hasattr(manifest, "files"):
        return ""
    fm = manifest.files.get(filepath)
    if fm is None:
        return ""
    imports_from = getattr(fm, "imports_from", {}) or {}

    blocks: list[str] = []
    seen: set[str] = set()
    for src_raw in imports_from:
        src = workspace_path(src_raw)
        if src not in manifest.files or src == filepath or src in seen:
            continue
        seen.add(src)
        dep = manifest.files[src]
        if getattr(dep, "written", False):
            continue
        contract = getattr(dep, "contract", {}) or {}
        if not contract:
            continue
        lines = []
        for export in getattr(dep, "exports", None) or []:
            fields = contract.get(export)
            if fields:
                lines.append(f"  - `{export}`: " + ", ".join(f"{n}: {t}" for n, t in fields.items()))
        if lines:
            blocks.append(f"### `{src}` (planned exports + fields)\n" + "\n".join(lines))

    if not blocks:
        return ""
    return (
        "## PENDING CONTRACTS — fields of not-yet-written dependencies\n"
        "The files below are being generated in parallel. Their exports and fields "
        "are FIXED by the planner — use EXACTLY these names (no guessing, no renaming):\n\n"
        + "\n\n".join(blocks)
    )


def _shares_resource(path_a: str, path_b: str) -> bool:
    """True when two paths share a directory or an identifier token (e.g. both relate to 'todo')."""
    if not path_a or not path_b:
        return False
    if os.path.dirname(path_a).rstrip("/") == os.path.dirname(path_b).rstrip("/"):
        return True
    a_tokens = {t for t in os.path.basename(path_a).split(".")[0].lower().split("_") if t}
    b_tokens = {t for t in os.path.basename(path_b).split(".")[0].lower().split("_") if t}
    return bool(a_tokens & b_tokens)


def _build_project_file_tree(manifest: Any, highlight_path: str, task: Any, agent: str) -> str:
    """
    A CAPPED project file tree: a file is included iff it is the current file,
    a declared dependency of *task*, already written, trivial, or a same-agent
    + same-resource file. The current file is marked with '★'.
    """
    if not hasattr(manifest, "files") or not manifest.files:
        return "(no files planned yet)"

    dep_ids = set(task.depends_on) if (task and getattr(task, "depends_on", None)) else set()
    all_filepaths = sorted(manifest.files.keys(), key=lambda p: (p.rstrip("/").count("/"), p))
    entries: list[str] = []
    for path in all_filepaths:
        fm = manifest.files[path]
        written = getattr(fm, "written", False)
        included = (
            path == highlight_path
            or getattr(fm, "task_id", None) in dep_ids
            or written
            or _is_trivial_filepath(path, getattr(fm, "exports", []))
            or (fm.agent == agent and _shares_resource(path, highlight_path))
        )
        if not included:
            continue
        marker = " ★ (generating)" if path == highlight_path else ""
        status = "written" if written else "pending"
        exports = ", ".join(getattr(fm, "exports", []) or []) or "(none)"
        depth = path.rstrip("/").count("/")
        entries.append(f"{'  ' * depth}- `{path}`  [{status}] {fm.agent} — exports: {exports}{marker}")
    if not entries:
        return "(no relevant project files planned yet)"
    if len(entries) > MAX_TREE_ENTRIES:
        omitted = len(entries) - MAX_TREE_ENTRIES
        entries = entries[:MAX_TREE_ENTRIES]
        entries.append(f"  … and {omitted} more files omitted (tree capped)")
    return "\n".join(entries)


def _build_scaffold_context(manifest: Any, filepath: str) -> str:
    """
    Deterministic wiring instructions for entrypoint/scaffold files: the backend
    entrypoint gets the exact router list; index.html the exact stylesheet/script list.
    """
    if not hasattr(manifest, "files"):
        return ""
    base = os.path.basename(filepath or "").lower()
    if base in ("main.py", "app.py", "server.py"):
        routers = []
        for path, fm in manifest.files.items():
            if path == filepath or fm.agent != "backend":
                continue
            exports = getattr(fm, "exports", []) or []
            if "router" in exports or "routes" in path.lower() or "controllers" in path.lower():
                routers.append(f"- `{path}` → `from {dotted_import(path)} import router`")
        header = "## Scaffold Wiring — You MUST register these routers (their files already exist):"
        note = "\n**IMPORT RULE**: Import each router with the EXACT import line shown (layer-root-relative), NEVER with the `output.backend.` / `output.` workspace prefix."
        return header + note + ("\n" + "\n".join(routers) if routers else "\n(none found)")
    if base == "index.html":
        styles = sorted(p for p in manifest.files if p != filepath and p.endswith((".css", ".scss", ".sass")))
        scripts = sorted(
            p for p in manifest.files if p != filepath and p.endswith((".js", ".ts", ".tsx", ".jsx"))
        )
        parts = ["## Scaffold Wiring — index.html MUST load the following files in this order:"]
        if styles:
            parts.append("\n".join(f"- stylesheet: `{p}`" for p in styles))
        if scripts:
            parts.append("\n".join(f"- script: `{p}`" for p in scripts))
        if not styles and not scripts:
            parts.append("(no stylesheets/scripts planned yet)")
        if scripts:
            parts.append(
                "**NO INLINE LOGIC RULE**: All application behavior MUST live in the script "
                "files listed above and be loaded with <script src>. Do NOT duplicate "
                "fetch/render logic inside inline <script> tags in index.html — inline "
                "duplicates shadow the real script files and break the app."
            )
        return "\n".join(parts)
    return ""


def _tech_table(arch: Any) -> str:
    stack = getattr(arch, "tech_stack", None) or {}
    return "\n".join(f"  {k}: {v}" for k, v in stack.items()) if stack else "  (none)"


def _dependency_manifest(manifest: CodeManifest, filepath: str, task: Task | None) -> str:
    """The file's dependencies in the Code Manifest, with signatures of written ones."""
    entries = []
    for path, fm in _filter_manifest_by_dependencies(manifest, filepath, task).items():
        status = "written" if fm.written else "pending"
        exports = ", ".join(fm.exports) or "(none)"
        entry = [f"- `{path}` ({status})\n  Purpose: {fm.purpose}\n  Exports: {exports}"]
        if fm.written:
            for cls in fm.classes:
                args = ", ".join(f"{a.name}: {a.type}" for a in cls.arguments)
                entry.append(f"    class {cls.name}({args}): {cls.docstring}")
            for fn in fm.functions:
                args = ", ".join(f"{a.name}: {a.type}" for a in fn.arguments)
                entry.append(f"    function {fn.name}({args}) -> {fn.return_type}: {fn.docstring}")
        entries.append("\n".join(entry))
    if len(entries) > MAX_MANIFEST_ENTRIES:
        omitted = len(entries) - MAX_MANIFEST_ENTRIES
        entries = entries[:MAX_MANIFEST_ENTRIES]
        entries.append(f"- … and {omitted} more dependency files omitted")
    return "\n\n".join(entries) or "No relevant dependency files in manifest."


def _file_request(profile: CodingProfile, ctx: FileContext) -> str:
    """The request for one file: header, its contracts, the profile's sections, the tree."""
    filepath, fm = ctx.filepath, ctx.fm
    import_rule = (
        "**IMPORT RULE**: Import internal modules relative to the layer root (e.g. "
        "`from db.session import ...`), NEVER with the `output.backend.` / `output.` workspace prefix.\n"
        if filepath.endswith(".py") else ""
    )
    parts = [
        "## File to Generate\n"
        f"- **File Path**: `{filepath}`\n"
        f"- **Purpose**: {fm.purpose}\n"
        f"- **Required Exports**: {', '.join(fm.exports) or '(none)'}\n"
        "  *(CRITICAL: The generated code MUST define and export these exact names "
        "with the exact signatures other modules expect.)*",
        "## Tech Stack\n" + _tech_table(ctx.arch),
        *profile.sections(ctx),
        _build_scaffold_context(ctx.manifest, filepath),
        build_import_contract(ctx.manifest, filepath),
        _build_pending_contracts(ctx.manifest, filepath),
        "## Project File Tree (capped)\n"
        "You may ONLY import from files listed here — use the EXACT export names shown.\n"
        + import_rule
        + _build_project_file_tree(ctx.manifest, filepath, ctx.task, profile.agent),
        "## Project Code Manifest (Dependencies)\n"
        "**RULE**: Never import from a file path not in this list. Use these exact "
        "signatures when calling other modules.\n\n"
        + _dependency_manifest(ctx.manifest, filepath, ctx.task),
        "---\n"
        f"Write the complete source for `{filepath}`.\n"
        f"Respond with a single valid JSON object with keys: {', '.join(profile.reply_keys)}. "
        "No prose outside the JSON.\n```json\n",
    ]
    return request_header([filepath]) + "\n\n".join(p for p in parts if p.strip())


def _batch_request(profile: CodingProfile, fms: list[FileManifest], arch: ArchData) -> str:
    """A compact request for 2-3 trivial files, answered with a JSON array."""
    entries = [
        f"### File: {fm.filepath}\nPurpose: {fm.purpose}\nRequired Exports: {', '.join(fm.exports) or '(none)'}"
        for fm in fms
    ]
    keys = profile.reply_keys
    item = ",".join([f'"filepath":"<path>"', f'"{profile.code_key}":"<full source>"',
                     *(f'"{k}":[]' for k in keys[1:])])
    return request_header([fm.filepath for fm in fms]) + (
        f"Generate the following {len(fms)} trivial/boilerplate {profile.tech(arch)} files.\n\n"
        + "\n\n".join(entries) + "\n\n"
        "Return a JSON ARRAY (not object) — one element per file in the same order:\n"
        f"[{{{item}}}, ...]\n"
        "```json\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  The loop
# ══════════════════════════════════════════════════════════════════════════════

class CodingAgent(BaseAgent):
    """
    A Coding agent: a name plus a CodingProfile. Subclasses set `profile`; the
    loop (staging, Planned Content, context, LLM call, validation, write,
    record) is the same for all of them.
    """

    profile: CodingProfile

    def __init__(self, profile: CodingProfile | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        if profile is not None:
            self.profile = profile
        if getattr(self, "profile", None) is None:
            raise TypeError(f"{type(self).__name__} needs a CodingProfile")

    @property
    def system_prompt(self) -> str:
        return f"You are an expert {self.profile.agent} engineer."

    # ── staging ───────────────────────────────────────────────────────────
    def _work(self, ps: ProjectState) -> None:
        profile = self.profile
        tasks = sorted(ps.tasks.by_agent(profile.agent), key=lambda t: t.priority)
        if not tasks:
            logger.info("[%s] No %s tasks — nothing to do.", self.name, profile.agent)
            return

        stages = resolve_stages(tasks, use_priority=profile.stage_by_priority)
        logger.info("[%s] %d tasks → %d stage(s).", self.name, len(tasks), len(stages))
        lock = Lock()

        for idx, stage in enumerate(stages):
            batched = [t for t in stage if _is_batchable(ps.code_manifest.by_task(t.id))]
            singles = [t for t in stage if t not in batched]
            batches = [batched[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(batched), TRIVIAL_BATCH_SIZE)]
            logger.info(
                "[%s] Stage %d/%d — %d batched (→ %d call(s)) + %d single",
                self.name, idx + 1, len(stages), len(batched), len(batches), len(singles),
            )
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
                futures: list[Future] = [pool.submit(self._code_batch, b, ps, lock) for b in batches]
                futures += [pool.submit(self._code_file, t, ps, lock) for t in singles]
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("[%s] Unhandled task exception: %s", self.name, exc)
                        with lock:
                            ps.log_error(self.name, f"Task exception: {exc}")

        unwritten = [m for m in ps.code_manifest.unwritten() if m.agent == profile.agent]
        logger.info("[%s] Done. Written: %d  Unwritten: %d.",
                    self.name, len(tasks) - len(unwritten), len(unwritten))

    # ── one file ──────────────────────────────────────────────────────────
    def _code_file(self, task: Task, ps: ProjectState, lock: Lock) -> None:
        with lock:
            fm = ps.code_manifest.by_task(task.id)
        if fm is None:
            logger.error("[%s] Task %s has no manifest entry", self.name, task.id)
            with lock:
                ps.log_error(self.name, f"Task {task.id} has no manifest entry")
            return

        if fm.planned_content is not None:
            self._commit(ps, lock, fm.filepath, fm.planned_content, {})
            return

        with lock:
            arch = ps.architecture
            ctx = FileContext(fm.filepath, fm, task, ps.code_manifest, arch)
            system = self.profile.system_prompt(arch)
            request = _file_request(self.profile, ctx)

        raw = self._ask(system, request, fm.filepath, ps, lock)
        if raw is None:
            return
        self._accept(raw, fm.filepath, ps, lock)

    # ── a batch of trivial files ──────────────────────────────────────────
    def _code_batch(self, tasks: list[Task], ps: ProjectState, lock: Lock) -> None:
        with lock:
            fms = [fm for fm in (ps.code_manifest.by_task(t.id) for t in tasks) if fm is not None]
            arch = ps.architecture
            system = self.profile.system_prompt(arch)
            request = _batch_request(self.profile, fms, arch)
        if not fms:
            return
        paths = [fm.filepath for fm in fms]
        logger.info("[%s] Batch call: %s", self.name, paths)

        raw = self._ask(system, request, ", ".join(paths), ps, lock)
        if raw is None:
            return
        try:
            items = self._parse_batch_response(raw)
        except Exception as exc:
            logger.error("[%s] Batch JSON parse failed: %s — raw: %s", self.name, exc, raw[:300])
            with lock:
                ps.log_error(self.name, f"Batch JSON parse failed: {exc}")
            return

        aligned = _align_batch_items(items, fms)
        if aligned is None:
            logger.warning(
                "[%s] Batch reply count mismatch (%d items for %d files) — falling back to per-file calls",
                self.name, len(items), len(fms),
            )
            for t in tasks:
                self._code_file(t, ps, lock)
            return

        for item, fm in aligned:
            code = str(item.get(self.profile.code_key, "") or "").strip()
            if not code:
                logger.warning("[%s] Batch: empty %s for %s — skipping", self.name, self.profile.code_key, fm.filepath)
                continue
            self._commit(ps, lock, fm.filepath, code, item)

    # ── LLM call, validation, write + record ──────────────────────────────
    def _ask(self, system: str, request: str, what: str, ps: ProjectState, lock: Lock) -> str | None:
        """Call the LLM (with retries); log an error and return None on failure."""
        try:
            return self._call_with_system(system, request, agent_name=f"{self.profile.agent}", state=ps, lock=lock)
        except Exception as exc:
            logger.error("[%s] LLM call failed for %s: %s", self.name, what, exc)
            with lock:
                ps.log_error(self.name, f"LLM call failed for {what}: {exc}")
            return None

    def _accept(self, raw: str, filepath: str, ps: ProjectState, lock: Lock) -> None:
        """Extract and validate one file's reply against the profile's schema, then commit it."""
        try:
            data = self.extract_json_and_validate(raw, self.profile.response_schema, state=ps, lock=lock)
        except Exception as exc:
            logger.error("[%s] JSON parse/validation failed for %s: %s", self.name, filepath, exc)
            with lock:
                ps.log_error(self.name, f"JSON parse/validation failed for {filepath}: {exc}")
            return
        code = str(data.get(self.profile.code_key, "") or "").strip()
        if not code:
            logger.error("[%s] '%s' missing in reply for %s", self.name, self.profile.code_key, filepath)
            with lock:
                ps.log_error(self.name, f"'{self.profile.code_key}' missing in LLM reply for {filepath}")
            return
        self._commit(ps, lock, filepath, code, data)

    def _commit(self, ps: ProjectState, lock: Lock, filepath: str, code: str, reply: dict) -> None:
        """
        The one place a Coding agent's file is written, recorded and marked
        Written: write through the workspace's guard, then record through the
        state-update module.
        """
        def _refuse(path: str, conflict: dict) -> None:
            logger.warning("[safe-write] REFUSED corrupted content for %s: %s", path, conflict.get("reason"))

        try:
            written = self.workspace.write(filepath, code, on_corrupt=_refuse)
        except Exception as exc:
            logger.error("[%s] Disk write failed for %s: %s", self.name, filepath, exc)
            with lock:
                ps.log_error(self.name, f"Disk write failed for {filepath}: {exc}")
            return
        if written is None:
            logger.error("[%s] Corrupted content refused for %s — file not written", self.name, filepath)
            return

        classes, functions = parse_metadata(reply, default_return_type=self.profile.default_return_type)
        with lock:
            ps.record_written(self.profile.agent, filepath, written, classes, functions)
        logger.info("[%s] ✓ %s", self.name, filepath)
