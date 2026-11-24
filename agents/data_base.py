"""
agents/db.py
────────────
Database Agent — runs after the Planner Agent.

Input:  ProjectState.tasks          (TaskList)
        ProjectState.code_manifest  (CodeManifest)
        ProjectState.architecture   (ArchData — tech stack + schema overview)
Output: ProjectState.generated_code.database (dict path -> code)
        ProjectState.code_manifest  (marked written)

Execution model:
  - Tasks are sorted by priority (lower number = higher priority).
  - Tasks sharing the same priority level run in parallel via ThreadPoolExecutor.
  - A single Lock guards all writes to ProjectState.
  - Each task produces one ORM model file + a <manifest> block.

The system prompt is rendered from the Persistence Contract: it names the
contract's database and ORM, the model directory and the connection module's
import line, and describes only ORM model files (the ORM creates the schema).
"""

import os
import sys
import json
import json as _json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent, agent_refuse_corrupt, filter_manifest_by_dependencies, is_trivial_file, TRIVIAL_BATCH_SIZE, MAX_TREE_ENTRIES, MAX_MANIFEST_ENTRIES, build_project_file_tree, build_import_contract, align_batch_items
except ImportError:
    # Fallback for direct execution
    from base import BaseAgent, agent_refuse_corrupt, filter_manifest_by_dependencies, is_trivial_file, TRIVIAL_BATCH_SIZE, MAX_TREE_ENTRIES, MAX_MANIFEST_ENTRIES, build_project_file_tree, build_import_contract, align_batch_items
from agents.coding_request import request_header
from agents.utils import parse_metadata, resolve_stages

from core.state import (
    ProjectState,
    Task,
    TaskList,
    FileManifest,
    CodeManifest,
    Argument,
    ClassName,
    FunctionName,
    PersistenceContract,
    connection_import_line,
    render_persistence_contract,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 3


# ── System prompt ──────────────────────────────────────────────────────────────

def _db_tech(contract: PersistenceContract) -> str:
    """The contract's database and ORM, as named in prompts."""
    return f"{contract.database} ({contract.orm})"


def _orm_constraint_block(contract: PersistenceContract) -> str:
    """Instruction forcing the contract's single ORM paradigm."""
    orm = contract.orm
    return (
        f"## ORM: {orm}\n"
        f"IMPORTANT: Use {orm} as the single data-access layer. "
        "Do NOT use Prisma, Drizzle, TypeORM, raw DDL in parallel, or any other ORM. "
        "All models must use the same paradigm as the backend models."
    )


def _system_prompt_for(arch: Any) -> str:
    """System prompt for this project, rendered from its Persistence Contract."""
    return _build_system_prompt(arch.persistence_contract)


def _build_system_prompt(contract: PersistenceContract) -> str:
    """System prompt rendered from the Persistence Contract: ORM model files only."""
    orm = contract.orm
    return f"""\
You are an expert database engineer specialising in {_db_tech(contract)}.
Generate one complete {orm} ORM model file AND a structured manifest block.
Model files live under `{contract.model_dir}` (relative to the backend Layer root).
{orm} creates the schema from these models when the init command runs, so model
files are the only database files you write.

{_orm_constraint_block(contract)}

{render_persistence_contract(contract)}

## Model Rules
1.  Model and field names MUST match the Architecture database_schema_overview EXACTLY
    (snake_case throughout, no renaming, no pluralisation changes, no casing changes).
2.  Relationships reference only models planned in the Code Manifest.
    Respect priority/depends_on ordering.
3.  Primary keys use UUIDs by default unless the Architecture explicitly specifies otherwise,
    consistently across ALL models.
4.  Every model has `created_at` and `updated_at` timestamp fields unless the task
    explicitly says otherwise.
5.  Never invent models or fields not present in the task or schema overview.
    Suggested additions go in the manifest `suggested_additions` field only.
6.  Make every field's optionality explicit.
7.  Model files only declare models: they never create engines, sessions or clients and
    never read the database URL. Code that needs the database imports it with
    `{connection_import_line(contract)}`.
8.  The manifest lists the exact model name and exact field names + types — the backend
    reads it to use these models.

Return JSON: {{"migration_code":"<full source of the {orm} model file>","manifest":{{"table_name":"<tbl>","columns":[{{"name":"<col>","type":"<type>","nullable":true|false,"primary_key":true|false,"foreign_key":"<table.col or null>"}}],"indexes":["<col>"],"exports":["<tbl>"],"suggested_additions":[]}},"functions":[{{"name":"<func>","arguments":[{{"name":"<p>","type":"<t>","default":""}}],"return_type":"<t>","docstring":"<desc>"}}]}}
No prose outside the JSON object.
"""


# ── Priority-based stage resolver ──────────────────────────────────────────────

def resolve_db_stages(tasks: list[Task]) -> list[list[Task]]:
    """Groups DB tasks into execution stages based on priority then depends_on."""
    return resolve_stages(tasks, use_priority=True)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _fmt_manifest_context(manifest: CodeManifest, skip_path: str, task: Task | None = None) -> str:
    dep_files = filter_manifest_by_dependencies(manifest, skip_path, task)
    entries = []
    for path, fm in dep_files.items():
        status = "written" if fm.written else "pending"
        exports = ", ".join(fm.exports) or "(none)"
        entries.append(f"- `{path}` ({status})\n  Purpose: {fm.purpose}\n  Exports: {exports}")
    if len(entries) > MAX_MANIFEST_ENTRIES:
        omitted = len(entries) - MAX_MANIFEST_ENTRIES
        entries = entries[:MAX_MANIFEST_ENTRIES]
        entries.append(f"- … and {omitted} more dependency files omitted")
    return "\n\n".join(entries) or "No relevant dependency files in manifest."


def _build_file_prompt(
    filepath: str,
    task: Task,
    manifest: CodeManifest,
    arch: Any,
) -> str:
    fm = manifest.files.get(filepath)
    orm_block = _orm_constraint_block(arch.persistence_contract)
    purpose = fm.purpose if fm else task.description
    exports = ", ".join(fm.exports) if fm and fm.exports else "(none)"

    tech_table = (
        "\n".join(f"  {k}: {v}" for k, v in arch.tech_stack.items())
        if arch.tech_stack else "  (none)"
    )

    # Already-written DB files for FK context
    written_tables = []
    for path, m in manifest.files.items():
        if m.written and m.agent == "database":
            written_tables.append(f"- `{path}` exports: {', '.join(m.exports) or '(none)'}")
    if len(written_tables) > MAX_MANIFEST_ENTRIES:
        omitted = len(written_tables) - MAX_MANIFEST_ENTRIES
        written_tables = written_tables[:MAX_MANIFEST_ENTRIES]
        written_tables.append(f"- … and {omitted} more written DB files omitted")
    written_str = "\n".join(written_tables) or "None yet."

    manifest_ctx = _fmt_manifest_context(manifest, filepath, task)

    depends_str = ", ".join(task.depends_on) if task.depends_on else "(none)"

    tree_str = build_project_file_tree(
        manifest, highlight_path=filepath, task=task, agent="database"
    )

    import_contract = build_import_contract(manifest, filepath)

    return request_header([filepath]) + f"""\
## Model File to Generate
- **File Path** : `{filepath}`
- **Purpose**   : {purpose}
- **Exports**   : {exports}
- **Depends On**: {depends_str}
- **Priority**  : {task.priority}

## Tech Stack
{tech_table}

{orm_block}

{import_contract}

## Project File Tree (capped)
**IMPORT RULE**: Import internal modules relative to the layer root (e.g. `from db.session import ...`), NEVER with the `output.backend.` / `output.` workspace prefix.
{tree_str}

## Architecture — Database Schema Overview
{getattr(arch, 'database_schema_overview', 'No schema overview provided.')}

## Already Written Database Files (available for FK references)
{written_str}

## Project Code Manifest (Dependencies)
{manifest_ctx}

---
Generate the complete {arch.persistence_contract.orm} model file `{filepath}`.
Follow ALL rules in the system prompt.
Return a single JSON object with keys: migration_code, manifest, functions.
```json
"""



# ── Manifest XML serialiser ────────────────────────────────────────────────────

def _render_manifest_xml(manifest_data: dict) -> str:
    """Serialise the manifest dict returned by the LLM into the canonical <manifest> block."""
    return f"<manifest>\n{json.dumps(manifest_data, indent=2)}\n</manifest>"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DatabaseAgent                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DatabaseAgent(BaseAgent):
    """
    Priority-staged, parallel ORM model file generator.

    Execution order:
      - Tasks are sorted by priority (ascending).
      - Tasks sharing the same priority form a stage and run in parallel.
      - Within a priority group, depends_on is still respected via topological sort.
      - A threading Lock protects all writes to ProjectState.
    """

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return "You are an expert database engineer."


    # ── Trivial batch worker ──────────────────────────────────────────────────

    def _build_batch_prompt(self, tasks: list[Task], manifest: Any, arch: Any, db_tech: str) -> str:
        """Compact batch prompt for 2-3 trivial ORM model files — returns a JSON array."""
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
            f"Generate the following {len(tasks)} trivial/boilerplate {db_tech} model files.\n\n"
            f"{batch_desc}\n\n"
            "Return a JSON ARRAY (not object) — one element per file in the same order:\n"
            '[{"filepath":"<path>","migration_code":"<full source of the model file>","manifest":{},"functions":[]}, ...]\n'
            "```json\n"
        )

    def _process_batch(self, tasks: list[Task], state: ProjectState, lock: Lock) -> None:
        """Single LLM call for a batch of trivial ORM model files."""
        with lock:
            manifest = state.code_manifest
            arch = state.architecture
            db_tech = _db_tech(arch.persistence_contract)
            sys_prompt  = _system_prompt_for(arch)
            user_prompt = self._build_batch_prompt(tasks, manifest, arch, db_tech)

        filepaths = [manifest.by_task(t.id).filepath for t in tasks if manifest.by_task(t.id)]
        logger.info("[DBAgent] Batch trivial call: %s", filepaths)

        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name="db-batch", state=state, lock=lock)
        except Exception as exc:
            logger.error("[DBAgent] Batch LLM call failed: %s", exc)
            with lock:
                state.log_error(self.name, f"Batch LLM call failed: {exc}")
            return

        try:
            items = self._parse_batch_response(raw)
        except Exception as exc:
            logger.error("[DBAgent] Batch JSON parse failed: %s — raw: %s", exc, raw[:300])
            with lock:
                state.log_error(self.name, f"Batch JSON parse failed: {exc}")
            return

        aligned = align_batch_items(items, tasks, manifest)
        if aligned is None:
            logger.warning(
                "[DBAgent] Batch response count mismatch (%d items for %d tasks) — falling back to per-file generation",
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
            code = str(item.get("migration_code", "")).strip()
            if not code:
                logger.warning("[DBAgent] Batch: empty migration_code for %s", fm.filepath)
                continue
            classes, functions = parse_metadata(item)
            try:
                written_code = self.workspace.write(fm.filepath, code, on_corrupt=agent_refuse_corrupt(state, lock))
            except Exception as exc:
                logger.error("[DBAgent] Batch disk write failed for %s: %s", fm.filepath, exc)
                written_code = None
            if written_code is None:
                logger.warning("[DBAgent] Batch: corrupted content refused for %s", fm.filepath)
                continue
            code = written_code
            with lock:
                code_bucket = getattr(state.generated_code, self.code_bucket)
                code_bucket[fm.filepath] = code
                state.code_manifest.mark_written(fm.filepath, classes, functions)
            logger.info("[DBAgent] ✓ batch trivial: %s", fm.filepath)

    def _process_task(self, task: Task, state: ProjectState, lock: Lock) -> None:
        """Worker thread: call LLM, parse response, write migration + manifest."""

        # Fetch file manifest entry under lock
        with lock:
            fm = state.code_manifest.by_task(task.id)

        if not fm:
            logger.error("[DBAgent] Task %s has no manifest entry.", task.id)
            with lock:
                state.log_error(self.name, f"Task {task.id} has no manifest entry")
            return

        filepath = fm.filepath
        logger.info("[DBAgent] Generating migration for %s (task %s) …", filepath, task.id)

        # Planned Content (barrels, the connection module) is written verbatim.
        if fm.planned_content is not None:
            self._write_deterministic(task.id, filepath, fm.planned_content, "planned content", state, lock)
            return

        # Build prompts while holding the lock (read-only snapshot of shared state)
        with lock:
            arch = state.architecture
            sys_prompt = _system_prompt_for(arch)
            user_prompt = _build_file_prompt(filepath, task, state.code_manifest, arch)

        # LLM call (outside lock — blocking I/O)
        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name=f"db-{task.id}", state=state, lock=lock)
        except Exception as exc:
            logger.error("[DBAgent] LLM call failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"LLM call failed for {filepath}: {exc}")
            return

        # Parse JSON response
        try:
            from agents.schemas.database import DatabaseSchema
            data = self.extract_json_and_validate(raw, DatabaseSchema, state=state, lock=lock)
        except Exception as exc:
            snippet = raw[:500] if raw else "(empty)"
            logger.error(
                "[DBAgent] JSON parse/validation failed for %s: %s | raw: %s", filepath, exc, snippet
            )
            with lock:
                state.log_error(self.name, f"JSON parse/validation failed for {filepath}: {exc}")
            return

        migration_code = data.get("migration_code", "").strip()
        if not migration_code:
            logger.error("[DBAgent] 'migration_code' missing in response for %s", filepath)
            with lock:
                state.log_error(self.name, f"'migration_code' missing for {filepath}")
            return

        classes, functions = parse_metadata(data)

        # Write migration file to disk (safe-write guard refuses corrupted content)
        try:
            written_code = self.workspace.write(filepath, migration_code, on_corrupt=agent_refuse_corrupt(state, lock))
        except Exception as exc:
            logger.error("[DBAgent] Disk write failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"Disk write failed for {filepath}: {exc}")
            written_code = None
        if written_code is None:
            logger.error("[DBAgent] Corrupted content refused for %s — file not written", filepath)
            return
        migration_code = written_code

        # Thread-safe state update
        with lock:
            state.generated_code.database[filepath] = migration_code
            state.code_manifest.mark_written(filepath, classes=classes, functions=functions)

        logger.info("[DBAgent] ✓ %s", filepath)

    # ── Main entry point ──────────────────────────────────────────────────────

    # ── Main entry point ──────────────────────────────────────────────────────

    def _work(self, ps: ProjectState) -> None:

        db_tasks = ps.tasks.by_agent("database")
        if not db_tasks:
            logger.info("[DBAgent] No database tasks — nothing to do.")
            return

        logger.info("[DBAgent] Resolving stages for %d database tasks …", len(db_tasks))
        stages = resolve_db_stages(db_tasks)
        logger.info("[DBAgent] Scheduled into %d priority-ordered stage(s).", len(stages))

        state_lock = Lock()

        # Files with Planned Content are never batched.
        def _is_batchable(t: Task) -> bool:
            return (
                is_trivial_file(t, ps.code_manifest)
                and getattr(ps.code_manifest.by_task(t.id), "planned_content", None) is None
            )

        for idx, stage in enumerate(stages):
            trivial_batch = [t for t in stage if _is_batchable(t)]
            non_trivial   = [t for t in stage if not _is_batchable(t)]
            batches    = [trivial_batch[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(trivial_batch), TRIVIAL_BATCH_SIZE)]

            logger.info(
                "[DBAgent] Stage %d/%d — %d trivial (→ %d batch call(s)) + %d non-trivial",
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
                        logger.error("[DBAgent] Unhandled task exception: %s", exc)
                        with state_lock:
                            ps.log_error(self.name, f"Task exception: {exc}")

        unwritten = [m for m in ps.code_manifest.unwritten() if m.agent == "database"]
        logger.info(
            "[DBAgent] Done. Written: %d  Unwritten: %d.",
            len(db_tasks) - len(unwritten),
            len(unwritten),
        )
