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

import json
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
    refusal: str = ""   # why the safe-write guard refused this file's previous write


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


def refusal_recorder(ps: ProjectState, lock: Lock | None = None) -> Callable[[str, dict], None]:
    """
    The safe-write guard's `on_corrupt` callback: logs the refused write and
    records its reason in ProjectState as a Write refusal (under *lock*, when given).
    """
    def refuse(path: str, conflict: dict) -> None:
        reason = str(conflict.get("reason") or "")
        logger.warning("[safe-write] REFUSED corrupted content for %s: %s", path, reason)
        if lock is None:
            ps.record_refusal(path, reason)
        else:
            with lock:
                ps.record_refusal(path, reason)
    return refuse


# ══════════════════════════════════════════════════════════════════════════════
#  Batching (private)
# ══════════════════════════════════════════════════════════════════════════════

#: File name stems that are trivial by name: type declarations, constants,
#: configuration, settings and error/exception modules. Only these are batched;
#: routers, ORM models and schema modules always get a single call, however
#: few exports they have.
_TRIVIAL_STEMS = frozenset({"types", "constants", "config", "settings", "errors", "exceptions"})

#: Barrel / re-export files. Their content is Planned Content, so they never
#: reach the LLM; the file tree always lists them.
_BARREL_NAMES = frozenset({"__init__.py", "index.ts", "index.tsx", "index.js", "index.jsx"})


def _is_trivial_by_name(filepath: str) -> bool:
    """True for a file trivial by its name (e.g. `types.ts`, `app/core/config.py`, `errors.js`)."""
    stem = os.path.basename(filepath or "").lower().split(".")[0]
    return stem in _TRIVIAL_STEMS


def _is_batchable(fm: FileManifest | None) -> bool:
    """A File Manifest goes into a batch call: trivial by name and without Planned Content."""
    return fm is not None and fm.planned_content is None and _is_trivial_by_name(fm.filepath)


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


def _build_project_file_tree(manifest: Any, ctxs: list[FileContext], agent: str) -> str:
    """
    A CAPPED project file tree for the file(s) being generated: a file is
    included iff it is being generated, a declared dependency of one of their
    tasks, already written, trivial by name, a barrel, small (fewer than three
    exports), or a same-agent + same-resource file. Generated files are marked '★'.
    """
    if not hasattr(manifest, "files") or not manifest.files:
        return "(no files planned yet)"

    highlights = {c.filepath for c in ctxs}
    dep_ids = {d for c in ctxs if c.task for d in (c.task.depends_on or [])}
    all_filepaths = sorted(manifest.files.keys(), key=lambda p: (p.rstrip("/").count("/"), p))
    entries: list[str] = []
    for path in all_filepaths:
        fm = manifest.files[path]
        written = getattr(fm, "written", False)
        included = (
            path in highlights
            or getattr(fm, "task_id", None) in dep_ids
            or written
            or _is_trivial_by_name(path)
            or os.path.basename(path).lower() in _BARREL_NAMES
            or len(getattr(fm, "exports", None) or []) < 3
            or (fm.agent == agent and any(_shares_resource(path, h) for h in highlights))
        )
        if not included:
            continue
        marker = " ★ (generating)" if path in highlights else ""
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


def _refusal_section(reason: str) -> str:
    """The previous write's refusal reason, so the retry avoids the same mistake."""
    if not reason:
        return ""
    return (
        "## PREVIOUS ATTEMPT REFUSED\n"
        "The last version of this file was refused by the safe-write guard and NOT written:\n"
        f"- {reason}\n"
        "Return the complete, valid file this time."
    )


def _file_block(profile: CodingProfile, ctx: FileContext) -> list[str]:
    """One file's part of a request: its File Manifest, the profile's sections and its contracts."""
    fm = ctx.fm
    return [
        "## File to Generate\n"
        f"- **File Path**: `{ctx.filepath}`\n"
        f"- **Purpose**: {fm.purpose}\n"
        f"- **Required Exports**: {', '.join(fm.exports) or '(none)'}\n"
        "  *(CRITICAL: The generated code MUST define and export these exact names "
        "with the exact signatures other modules expect.)*",
        *profile.sections(ctx),
        _refusal_section(ctx.refusal),
        _build_scaffold_context(ctx.manifest, ctx.filepath),
        build_import_contract(ctx.manifest, ctx.filepath),
        _build_pending_contracts(ctx.manifest, ctx.filepath),
        "## Project Code Manifest (Dependencies)\n"
        "**RULE**: Never import from a file path not in this list. Use these exact "
        "signatures when calling other modules.\n\n"
        + _dependency_manifest(ctx.manifest, ctx.filepath, ctx.task),
    ]


def _build_request(profile: CodingProfile, ctxs: list[FileContext]) -> str:
    """
    The one context builder, for a single file or a batch: the request header,
    the tech stack, then for EACH file its File Manifest, the profile's
    sections, its Import Contract, its pending contracts and its dependencies,
    then the capped file tree and the reply format (an object for a single
    file, an array with one item per file for a batch).
    """
    paths = [c.filepath for c in ctxs]
    arch, manifest = ctxs[0].arch, ctxs[0].manifest
    batch = len(ctxs) > 1
    keys = profile.reply_keys

    parts: list[str] = []
    if batch:
        parts.append(
            f"Generate the following {len(ctxs)} {profile.tech(arch)} files. Each file's "
            "contracts follow its heading and apply to that file only."
        )
    parts.append("## Tech Stack\n" + _tech_table(arch))
    for n, ctx in enumerate(ctxs, start=1):
        if batch:
            parts.append(f"# File {n} of {len(ctxs)}: `{ctx.filepath}`")
        parts.extend(_file_block(profile, ctx))

    import_rule = (
        "**IMPORT RULE**: Import internal modules relative to the layer root (e.g. "
        "`from db.session import ...`), NEVER with the `output.backend.` / `output.` workspace prefix.\n"
        if any(p.endswith(".py") for p in paths) else ""
    )
    parts.append(
        "## Project File Tree (capped)\n"
        "You may ONLY import from files listed here — use the EXACT export names shown.\n"
        + import_rule
        + _build_project_file_tree(manifest, ctxs, profile.agent)
    )
    if batch:
        parts.append(
            "---\n"
            "Write the complete source for each file above.\n"
            "Return a JSON ARRAY (not object) — one element per file, in the order above:\n"
            f'[{{"filepath":"<path>","{keys[0]}":"<full source>", ...the other keys of the single-file '
            f"JSON object ({', '.join(keys[1:]) or 'none'})}}, ...]\n"
            "No prose outside the JSON.\n```json\n"
        )
    else:
        parts.append(
            "---\n"
            f"Write the complete source for `{paths[0]}`.\n"
            f"Respond with a single valid JSON object with keys: {', '.join(keys)}. "
            "No prose outside the JSON.\n```json\n"
        )
    return request_header(paths) + "\n\n".join(p for p in parts if p.strip())


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

    @classmethod
    def sharing(cls, agent: BaseAgent, profile: CodingProfile) -> "CodingAgent":
        """A Coding agent for *profile* sharing *agent*'s provider, model settings and workspace."""
        return cls(
            profile=profile,
            model=agent.model,
            max_tokens=agent.max_tokens,
            retries=agent.max_retries,
            backoff=agent.backoff,
            temperature=agent.temperature,
            provider=agent.provider,
            workspace=agent.workspace,
        )

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
            singles = [[t] for t in stage if t not in batched]
            batches = [batched[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(batched), TRIVIAL_BATCH_SIZE)]
            logger.info(
                "[%s] Stage %d/%d — %d batched (→ %d call(s)) + %d single",
                self.name, idx + 1, len(stages), len(batched), len(batches), len(singles),
            )
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
                futures: list[Future] = [pool.submit(self._code, group, ps, lock) for group in batches + singles]
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

    # ── one call: a single file, or a batch of files trivial by name ──────
    def _code(self, tasks: list[Task], ps: ProjectState, lock: Lock) -> None:
        """Code *tasks*' files in one LLM call (Planned Content is written verbatim instead)."""
        ctxs: list[FileContext] = []
        planned: list[FileManifest] = []
        with lock:
            for task in tasks:
                fm = ps.code_manifest.by_task(task.id)
                if fm is None:
                    logger.error("[%s] Task %s has no manifest entry", self.name, task.id)
                    ps.log_error(self.name, f"Task {task.id} has no manifest entry")
                elif fm.planned_content is not None:
                    planned.append(fm)
                else:
                    ctxs.append(self._context(fm, task, ps))

        for fm in planned:
            self._commit(ps, lock, fm.filepath, fm.planned_content, {})
        if not ctxs:
            return
        if len(ctxs) == 1:
            self._code_single(ctxs[0], ps, lock)
            return

        with lock:
            system = self.profile.system_prompt(ctxs[0].arch)
            request = _build_request(self.profile, ctxs)
        paths = [c.filepath for c in ctxs]
        raw = self._ask(system, request, ", ".join(paths), ps, lock)
        if raw is None:
            return

        try:
            items = self._parse_batch_response(raw)
        except Exception as exc:
            logger.error("[%s] Batch JSON parse failed: %s — raw: %s", self.name, exc, raw[:300])
            with lock:
                ps.log_error(self.name, f"Batch JSON parse failed for {', '.join(paths)}: {exc}")
            return

        aligned = _align_batch_items(items, [c.fm for c in ctxs])
        if aligned is None:
            logger.warning(
                "[%s] Batch reply count mismatch (%d items for %d files) — falling back to per-file calls",
                self.name, len(items), len(ctxs),
            )
            for ctx in ctxs:
                self._code([ctx.task], ps, lock)
            return

        # Every item is validated against the profile's schema, exactly like a single reply.
        for item, fm in aligned:
            self._accept(json.dumps(item), fm.filepath, ps, lock)

    # ── the single-file path ──────────────────────────────────────────────
    def regenerate(self, ps: ProjectState, filepath: str) -> bool:
        """
        Code one of this agent's planned files again through the single-file
        path (its contracts, task and any previous refusal reason in the
        request). Returns True once the file is written.
        """
        fm = ps.code_manifest.files.get(filepath)
        if fm is None or fm.agent != self.profile.agent:
            raise ValueError(f"{filepath!r} is not a planned {self.profile.agent} file")
        lock = Lock()
        if fm.planned_content is not None:
            self._commit(ps, lock, filepath, fm.planned_content, {})
            return filepath in (getattr(ps.generated_code, self.profile.agent) or {})
        task = next((t for t in ps.tasks.tasks if t.id == fm.task_id), None)
        ctx = self._context(fm, task, ps)
        return self.code_request(ps, filepath, _build_request(self.profile, [ctx]), lock)

    @staticmethod
    def _context(fm: FileManifest, task: Task | None, ps: ProjectState) -> FileContext:
        return FileContext(
            fm.filepath, fm, task, ps.code_manifest, ps.architecture,
            refusal=ps.write_refusals.get(fm.filepath, ""),
        )

    def _code_single(self, ctx: FileContext, ps: ProjectState, lock: Lock) -> None:
        """One planned file: its request context, then the single-file path."""
        with lock:
            request = _build_request(self.profile, [ctx])
        self.code_request(ps, ctx.filepath, request, lock)

    def code_request(self, ps: ProjectState, filepath: str, request: str, lock: Lock | None = None) -> bool:
        """
        The single-file path for a ready coding *request*: the profile's system
        prompt → LLM → validate → write → record. Returns True once *filepath*
        is in this agent's generated-code bucket.
        """
        lock = lock or Lock()
        with lock:
            system = self.profile.system_prompt(ps.architecture)
        raw = self._ask(system, request, filepath, ps, lock)
        if raw is not None:
            self._accept(raw, filepath, ps, lock)
        return filepath in (getattr(ps.generated_code, self.profile.agent) or {})

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
        try:
            written = self.workspace.write(filepath, code, on_corrupt=refusal_recorder(ps, lock))
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
