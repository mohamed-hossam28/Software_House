"""
agents/frontend.py
──────────────────
Frontend Agent — runs after the Planner Agent (parallel to BackendAgent).

Input:  ProjectState.tasks          (TaskList)
        ProjectState.code_manifest  (CodeManifest)
        ProjectState.architecture   (ArchData — tech stack + API contracts)
Output: ProjectState.generated_code.frontend (dict path → code)
        ProjectState.code_manifest  (marked written)

Execution model:
  - Groups frontend tasks into dependency-resolved stages.
  - Tasks in the same stage run in parallel (ThreadPoolExecutor, cap=MAX_PARALLEL_CALLS).
  - 2-second inter-stage delay to respect API rate limits.
  - Every file is written to disk under output/ AND stored in state.
  - API contracts are passed verbatim (method, path, request_body, response_body).
  - Tech stack drives the framework/language used in generated code.
"""

import os
import sys
import re
import json as _json
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent, agent_refuse_corrupt, filter_manifest_by_dependencies, filter_api_contracts_by_relevance, is_trivial_file, TRIVIAL_BATCH_SIZE, MAX_TREE_ENTRIES, MAX_MANIFEST_ENTRIES, build_project_file_tree, build_scaffold_context, is_scaffold_file, build_import_contract, align_batch_items
except ImportError:
    # Fallback for direct execution
    from base import BaseAgent, agent_refuse_corrupt, filter_manifest_by_dependencies, filter_api_contracts_by_relevance, is_trivial_file, TRIVIAL_BATCH_SIZE, MAX_TREE_ENTRIES, MAX_MANIFEST_ENTRIES, build_project_file_tree, build_scaffold_context, is_scaffold_file, build_import_contract, align_batch_items
from agents.coding_request import request_header
from agents.utils import parse_metadata, resolve_stages

from core.state import (
    ProjectState,
    Task,
    TaskList,
    FileManifest,
    CodeManifest,
    APIContract,
    Argument,
    ClassName,
    FunctionName,
)


from agents.schemas.frontend import FrontendSchema


logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 3

# ── Output directory (all generated files land here) ─────────────────────────


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  System prompt builder                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _build_system_prompt(frontend_tech: str) -> str:
    return f"""\
You are an expert frontend engineer specialising in {frontend_tech}.
Generate the COMPLETE source code for a single frontend file plus its manifest metadata.

## Non-negotiable rules
1. Return ONE valid JSON object — no prose outside the JSON.
2. The "code" field must be production-ready, complete, and syntax-valid. No placeholders, no TODOs.
3. Export every name listed in "Required Exports" with the exact signature implied by the manifest.
4. Only import from file paths listed in the Project Code Manifest.
5. Use the exact API endpoint paths, HTTP methods, request_body, and response_body fields from
   the API Contracts section — do NOT invent or rename fields.
6. Generate real, complete CSS/styling (not inline placeholders). If the file is a stylesheet,
   write a full stylesheet with variables, reset, typography, layout, and component classes.
   If it is a component, embed a <style> block or import the stylesheet — never leave styles empty.
7. Use {frontend_tech} idioms and best practices throughout.
8. Do not hardcode secrets; read from environment variables or config files.

Return JSON: {{"code":"<full source>","classes":[{{"name":"<class name>","arguments":[{{"name":"<param>","type":"<type>","default":""}}],"docstring":"<desc>"}}],"functions":[{{"name":"<func name>","arguments":[{{"name":"<param>","type":"<type>","default":""}}],"return_type":"<type>","docstring":"<desc>"}}]}}
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage resolver                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def resolve_frontend_stages(tasks: list[Task]) -> list[list[Task]]:
    """Groups frontend tasks into dependency-resolved execution stages."""
    return resolve_stages(tasks, use_priority=False)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Prompt builder                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _fmt_contracts(
    contracts: list[APIContract],
    filepath: str = "",
    purpose: str = "",
    task_title: str = "",
) -> str:
    rel_contracts = filter_api_contracts_by_relevance(
        contracts,
        filepath=filepath,
        purpose=purpose,
        task_title=task_title,
    )
    if not rel_contracts:
        return "  (none)"
    lines = []
    for c in rel_contracts:
        lock = "🔒" if c.auth_required else "🔓"
        lines.append(
            f"  {lock} {c.method:7} {c.path}\n"
            f"     Description : {c.description}\n"
            f"     request_body : {c.request_body}\n"
            f"     response_body: {c.response_body}"
        )
    return "\n".join(lines)


def _fmt_manifest(manifest: CodeManifest, skip_path: str, task: Task | None = None) -> str:
    dep_files = filter_manifest_by_dependencies(manifest, skip_path, task)
    entries = []
    for path, fm in dep_files.items():
        status = "written" if fm.written else "pending"
        exports = ", ".join(fm.exports) or "(none)"
        entry = [f"- `{path}` ({status})\n  Purpose: {fm.purpose}\n  Exports: {exports}"]
        if fm.written:
            for cls in fm.classes:
                args = ", ".join(f"{a.name}: {a.type}" for a in cls.arguments)
                entry.append(f"    class {cls.name}({args}): {cls.docstring}")
            for fn in fm.functions:
                args = ", ".join(f"{a.name}: {a.type}" for a in fn.arguments)
                entry.append(f"    function {fn.name}({args}): {fn.docstring}")
        entries.append("\n".join(entry))
    if len(entries) > MAX_MANIFEST_ENTRIES:
        omitted = len(entries) - MAX_MANIFEST_ENTRIES
        entries = entries[:MAX_MANIFEST_ENTRIES]
        entries.append(f"- … and {omitted} more dependency files omitted")
    return "\n\n".join(entries) or "No relevant dependency files in manifest."


def _build_file_prompt(
    filepath: str,
    manifest: CodeManifest,
    arch: Any,
    frontend_tech: str,
    task: Task | None = None,
) -> str:
    fm = manifest.files.get(filepath)
    if not fm:
        raise ValueError(f"File '{filepath}' not found in CodeManifest.")

    tech_table = "\n".join(f"  {k}: {v}" for k, v in arch.tech_stack.items()) if arch.tech_stack else "  (none)"
    contracts  = _fmt_contracts(
        arch.api_contracts or [],
        filepath=filepath,
        purpose=fm.purpose,
        task_title=task.title if task else "",
    )
    manifest_s = _fmt_manifest(manifest, filepath, task)
    is_style   = any(filepath.endswith(ext) for ext in (".css", ".scss", ".sass", ".less"))

    style_note = ""
    if is_style:
        style_note = (
            "\n⚠️  This is a STYLESHEET file. Write a complete, production-quality stylesheet:\n"
            "   - CSS custom properties (variables) for colours, spacing, typography\n"
            "   - Normalise / reset block\n"
            "   - Typography scale\n"
            "   - Layout helpers (flex/grid utilities)\n"
            "   - Component classes that match the component files in the manifest\n"
            "   - Responsive breakpoints\n"
            "   No empty rule-sets, no placeholder comments.\n"
        )

    tree_str = build_project_file_tree(
        manifest, highlight_path=filepath, task=task, agent="frontend"
    )

    scaffold_note = build_scaffold_context(manifest, filepath)

    import_contract = build_import_contract(manifest, filepath)

    return request_header([filepath]) + f"""\
## File to Generate
- **Path**     : `{filepath}`
- **Purpose**  : {fm.purpose}
- **Framework**: {frontend_tech}
- **Exports**  : {', '.join(fm.exports) or '(none)'}
  *(You MUST export each name listed above with the exact signature other modules expect.)*
{style_note}
## Tech Stack
{tech_table}

## API Contracts  ← USE THESE EXACT PATHS / FIELD NAMES
{contracts}

{scaffold_note}

{import_contract}

## Project File Tree (capped)
You may ONLY import from files listed here — use the EXACT export names shown.
{tree_str}

## Project Code Manifest
{manifest_s}

---

Write the complete source for `{filepath}`.
Respond with a single JSON object — "code", "classes", "functions" — no prose outside.
```json
"""



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FrontendAgent                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class FrontendAgent(BaseAgent):
    """
    Staged, parallel frontend code generator.

    Ordering: priority-based stages (same priority → parallel).
    Within each stage, intra-frontend dependency order is also respected.
    """

    # ── BaseAgent interface ───────────────────────────────────────────────────
    @property
    def system_prompt(self) -> str:
        return "You are an expert frontend engineer."


    # ── Per-task worker (runs in thread pool) ─────────────────────────────────
    def _build_batch_prompt(
        self, tasks: list[Task], manifest: Any, arch: Any, frontend_tech: str
    ) -> str:
        """Compact batch prompt for 2-3 trivial frontend files — returns a JSON array."""
        entries: list[str] = []
        for task in tasks:
            fm = manifest.by_task(task.id)
            if not fm:
                continue
            exports = ", ".join(fm.exports) or "(none)"
            entries.append(
                f"### File: {fm.filepath}\n"
                f"Purpose: {fm.purpose}\n"
                f"Required Exports: {exports}"
            )
        batch_desc = "\n\n".join(entries)
        return request_header([manifest.by_task(t.id).filepath for t in tasks if manifest.by_task(t.id)]) + (
            f"Generate the following {len(tasks)} trivial/boilerplate {frontend_tech} files.\n\n"
            f"{batch_desc}\n\n"
            "Return a JSON ARRAY (not object) — one element per file in the same order:\n"
            '[{"filepath":"<path>","code":"<full source>","classes":[],"functions":[]}, ...]\n'
            "```json\n"
        )

    def _process_batch(self, tasks: list[Task], state: ProjectState, lock: Lock) -> None:
        """Single LLM call for a batch of trivial frontend files."""
        with lock:
            manifest = state.code_manifest
            arch = state.architecture
            frontend_tech = arch.tech_stack.get("frontend", "React 18 + TypeScript") if arch.tech_stack else "React 18 + TypeScript"
            sys_prompt  = _build_system_prompt(frontend_tech)
            user_prompt = self._build_batch_prompt(tasks, manifest, arch, frontend_tech)

        filepaths = [manifest.by_task(t.id).filepath for t in tasks if manifest.by_task(t.id)]
        logger.info("[FrontendAgent] Batch trivial call: %s", filepaths)

        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name="fe-batch", state=state, lock=lock)
        except Exception as exc:
            logger.error("[FrontendAgent] Batch LLM call failed: %s", exc)
            with lock:
                state.log_error(self.name, f"Batch LLM call failed: {exc}")
            return

        try:
            items = self._parse_batch_response(raw)
        except Exception as exc:
            logger.error("[FrontendAgent] Batch JSON parse failed: %s — raw: %s", exc, raw[:300])
            with lock:
                state.log_error(self.name, f"Batch JSON parse failed: {exc}")
            return

        aligned = align_batch_items(items, tasks, manifest)
        if aligned is None:
            logger.warning(
                "[FrontendAgent] Batch response count mismatch (%d items for %d tasks) — falling back to per-file generation",
                len(items), len(tasks),
            )
            for t in tasks:
                self._process_task(t, state, lock)
            return

        for item, task in aligned:
            with lock:
                fm = state.code_manifest.by_task(task.id)
            if not fm:
                continue
            code = str(item.get("code", "")).strip()
            if not code:
                logger.warning("[FrontendAgent] Batch: empty code for %s", fm.filepath)
                continue
            classes, functions = parse_metadata(item, default_return_type="void")
            try:
                written_code = self.workspace.write(fm.filepath, code, on_corrupt=agent_refuse_corrupt(state, lock))
            except Exception as exc:
                logger.error("[FrontendAgent] Batch disk write failed for %s: %s", fm.filepath, exc)
                written_code = None
            if written_code is None:
                logger.warning("[FrontendAgent] Batch: corrupted content refused for %s", fm.filepath)
                continue
            code = written_code
            with lock:
                code_bucket = getattr(state.generated_code, self.code_bucket)
                code_bucket[fm.filepath] = code
                state.code_manifest.mark_written(fm.filepath, classes, functions)
            logger.info("[FrontendAgent] ✓ batch trivial: %s", fm.filepath)

    def _process_task(self, task: Task, state: ProjectState, lock: Lock) -> None:
        """Generate code for one frontend file, write to disk, update manifest."""
        with lock:
            fm = state.code_manifest.by_task(task.id)
            arch = state.architecture
            manifest = state.code_manifest
        if not fm:
            logger.error("[FrontendAgent] Task %s has no manifest entry", task.id)
            with lock:
                state.log_error(self.name, f"Task {task.id} has no manifest entry")
            return

        filepath = fm.filepath
        logger.info("[FrontendAgent] Generating %s (task %s) …", filepath, task.id)

        # Planned Content (barrels) is written verbatim.
        if fm.planned_content is not None:
            self._write_deterministic(task.id, filepath, fm.planned_content, "planned content", state, lock)
            return

        frontend_tech = arch.tech_stack.get("frontend", "React 18 + TypeScript") if arch and arch.tech_stack else "React 18 + TypeScript"
        user_prompt = _build_file_prompt(filepath, manifest, arch, frontend_tech, task)
        sys_prompt  = _build_system_prompt(frontend_tech)

        # LLM call (outside lock — blocking I/O)
        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name=f"fe-{task.id[:6]}", state=state, lock=lock)
        except Exception as exc:
            logger.error("[FrontendAgent] LLM call failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"LLM call failed for {filepath}: {exc}")
            return

        # Parse JSON
        try:
            data = self.extract_json_and_validate(raw, FrontendSchema, state=state, lock=lock)
        except Exception:
            snippet = raw[:500] if raw else "(empty)"
            logger.error("[FrontendAgent] JSON parse/validation failed for %s | raw: %s", filepath, snippet)
            with lock:
                state.log_error(self.name, f"JSON parse/validation failed for {filepath}")
            return

        code = data.get("code", "").strip()
        if not code:
            logger.error("[FrontendAgent] 'code' missing in response for %s", filepath)
            with lock:
                state.log_error(self.name, f"'code' missing in LLM response for {filepath}")
            return

        classes, functions = parse_metadata(data, default_return_type="void")

        # Write to disk (safe-write guard refuses corrupted content)
        try:
            written_code = self.workspace.write(filepath, code, on_corrupt=agent_refuse_corrupt(state, lock))
        except Exception as exc:
            logger.error("[FrontendAgent] Disk write failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"Disk write failed for {filepath}: {exc}")
            written_code = None
        if written_code is None:
            logger.error("[FrontendAgent] Corrupted content refused for %s — file not written", filepath)
            return
        code = written_code
        # Update state (thread-safe)
        with lock:
            state.generated_code.frontend[filepath] = code
            state.code_manifest.mark_written(filepath, classes, functions)

        logger.info("[FrontendAgent] ✓ %s", filepath)

    # ── Main entry point ──────────────────────────────────────────────────────
    def _work(self, ps: ProjectState) -> None:

        fe_tasks = ps.tasks.by_agent("frontend")
        if not fe_tasks:
            logger.info("[FrontendAgent] No frontend tasks — nothing to do.")
            return

        fe_tasks.sort(key=lambda t: t.priority)
        stages = resolve_frontend_stages(fe_tasks)

        logger.info(
            "[FrontendAgent] %d tasks → %d stages.", len(fe_tasks), len(stages)
        )

        state_lock = Lock()

        for idx, stage in enumerate(stages):
            # Split into trivial batches + non-trivial singles.
            # Scaffold/entrypoint files are NEVER batched — they need their wiring prompt.
            def _is_batchable(t: Task) -> bool:
                fm = ps.code_manifest.by_task(t.id)
                return (
                    is_trivial_file(t, ps.code_manifest)
                    and not is_scaffold_file(getattr(fm, "filepath", ""))
                    and getattr(fm, "planned_content", None) is None
                )

            trivial_batch = [t for t in stage if _is_batchable(t)]
            non_trivial   = [t for t in stage if not _is_batchable(t)]
            batches    = [trivial_batch[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(trivial_batch), TRIVIAL_BATCH_SIZE)]

            logger.info(
                "[FrontendAgent] Stage %d/%d — %d trivial (→ %d batch call(s)) + %d non-trivial",
                idx + 1, len(stages), len(trivial_batch), len(batches), len(non_trivial),
            )

            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
                futures: list[Future] = []
                for batch in batches:
                    futures.append(pool.submit(self._process_batch, batch, ps, state_lock))
                for task in non_trivial:
                    futures.append(pool.submit(self._process_task, task, ps, state_lock))

                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("[FrontendAgent] Unhandled task exception: %s", exc)
                        ps.log_error(self.name, f"Task exception: {exc}")

        unwritten = [m for m in ps.code_manifest.unwritten() if m.agent == "frontend"]
        logger.info(
            "[FrontendAgent] Done. Written: %d  Unwritten: %d.",
            len(fe_tasks) - len(unwritten),
            len(unwritten),
        )
