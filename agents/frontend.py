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
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent, filter_manifest_by_dependencies, filter_api_contracts_by_relevance
except ImportError:
    from base import BaseAgent, filter_manifest_by_dependencies, filter_api_contracts_by_relevance

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

logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 3

# ── Output directory (all generated files land here) ─────────────────────────
OUTPUT_DIR = "output"


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

## Response schema
{{
  "code": "<complete file source>",
  "classes": [
    {{
      "name": "<class / component name>",
      "arguments": [{{"name": "<prop>", "type": "<type>", "default": ""}}],
      "docstring": "<what it does>"
    }}
  ],
  "functions": [
    {{
      "name": "<function / hook name>",
      "arguments": [{{"name": "<param>", "type": "<type>", "default": ""}}],
      "return_type": "<return type>",
      "docstring": "<what it does>"
    }}
  ]
}}
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Stage resolver                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def resolve_frontend_stages(tasks: list[Task]) -> list[list[Task]]:
    """
    Groups frontend tasks into dependency-resolved execution stages.
    Tasks within a stage share no intra-frontend dependencies → safe to parallelise.
    """
    fe_ids = {t.id for t in tasks}
    visited: set[str] = set()
    remaining = list(tasks)
    stages: list[list[Task]] = []

    while remaining:
        ready: list[Task] = []
        for task in remaining:
            fe_deps = [d for d in task.depends_on if d in fe_ids]
            if all(d in visited for d in fe_deps):
                ready.append(task)

        if not ready:
            # Cycle / unreachable — bundle remainder to avoid deadlock
            stages.append(remaining)
            break

        stages.append(ready)
        for t in ready:
            visited.add(t.id)
            remaining.remove(t)

    return stages


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

    return f"""\
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

## Project Code Manifest
{manifest_s}

---

Write the complete source for `{filepath}`.
Respond with a single JSON object — "code", "classes", "functions" — no prose outside.
```json
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Disk writer                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _write_to_disk(filepath: str, code: str) -> None:
    """Write generated code under OUTPUT_DIR, preserving sub-path structure."""
    # Strip any leading "output/" the planner may have already added
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith(OUTPUT_DIR):
        rel = os.path.join(OUTPUT_DIR, rel)
    abs_path = os.path.join(_root_dir, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(code)


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

    def _build_prompt(self, state: ProjectState) -> str:
        raise NotImplementedError("_build_prompt unused; run() is overridden.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        raise NotImplementedError("_parse_response unused; run() is overridden.")

    # ── LLM wrapper with per-call system prompt + retries ────────────────────
    def _call_with_system(self, system: str, user: str, agent_name: str = "frontend") -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.llm_call(
                    prompt=user,
                    system_prompt=system,
                    agent_name=agent_name,
                )
                return resp.text
            except Exception as exc:
                logger.warning(
                    "[FrontendAgent] LLM attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)
        raise RuntimeError("Unreachable")

    # ── Per-task worker (runs in thread pool) ─────────────────────────────────
    def _process_task(self, task: Task, state: ProjectState, lock: Lock) -> None:
        """Generate code for one frontend file, write to disk, update manifest."""
        with lock:
            fm = state.code_manifest.by_task(task.id)
        if not fm:
            logger.error("[FrontendAgent] Task %s has no manifest entry", task.id)
            with lock:
                state.log_error(self.name, f"Task {task.id} has no manifest entry")
            return

        filepath = fm.filepath
        logger.info("[FrontendAgent] Generating %s (task %s) …", filepath, task.id)

        # Build prompts inside the lock (reads shared manifest snapshot)
        with lock:
            frontend_tech = (
                state.architecture.tech_stack.get("frontend", "React 18 + TypeScript")
                if state.architecture.tech_stack else "React 18 + TypeScript"
            )
            sys_prompt  = _build_system_prompt(frontend_tech)
            user_prompt = _build_file_prompt(
                filepath, state.code_manifest, state.architecture, frontend_tech, task=task
            )

        # LLM call (outside lock — blocking I/O)
        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name=f"fe-{task.id}")
        except Exception as exc:
            logger.error("[FrontendAgent] LLM call failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"LLM call failed for {filepath}: {exc}")
            return

        # Parse JSON
        try:
            from agents.schemas.frontend import FrontendSchema
            data = self.extract_json_and_validate(raw, FrontendSchema)
        except Exception as exc:
            snippet = raw[:500] if raw else "(empty)"
            logger.error("[FrontendAgent] JSON parse/validation failed for %s: %s | raw: %s", filepath, exc, snippet)
            with lock:
                state.log_error(self.name, f"JSON parse/validation failed for {filepath}: {exc}")
            return

        code = data.get("code", "").strip()
        if not code:
            logger.error("[FrontendAgent] 'code' missing in response for %s", filepath)
            with lock:
                state.log_error(self.name, f"'code' missing in LLM response for {filepath}")
            return

        # Parse manifest metadata
        classes: list[ClassName] = []
        for c in data.get("classes", []):
            try:
                args = [Argument(**a) for a in c.get("arguments", [])]
                classes.append(ClassName(name=c["name"], arguments=args, docstring=c.get("docstring", "")))
            except Exception as e:
                logger.warning("[FrontendAgent] class parse error: %s", e)

        functions: list[FunctionName] = []
        for f in data.get("functions", []):
            try:
                args = [Argument(**a) for a in f.get("arguments", [])]
                functions.append(FunctionName(
                    name=f["name"], arguments=args,
                    return_type=f.get("return_type", "void"),
                    docstring=f.get("docstring", ""),
                ))
            except Exception as e:
                logger.warning("[FrontendAgent] function parse error: %s", e)

        # Write to disk
        try:
            _write_to_disk(filepath, code)
        except Exception as exc:
            logger.error("[FrontendAgent] Disk write failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"Disk write failed for {filepath}: {exc}")

        # Update state (thread-safe)
        with lock:
            state.generated_code.frontend[filepath] = code
            state.code_manifest.mark_written(filepath, classes, functions)
            state.completed_task_ids.add(task.id)

        logger.info("[FrontendAgent] ✓ %s", filepath)

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)
        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        fe_tasks = ps.tasks.by_agent("frontend")
        if not fe_tasks:
            logger.info("[FrontendAgent] No frontend tasks — nothing to do.")
            return self._compute_diff(before, ps)

        fe_tasks.sort(key=lambda t: t.priority)
        stages = resolve_frontend_stages(fe_tasks)

        logger.info(
            "[FrontendAgent] %d tasks → %d stages.", len(fe_tasks), len(stages)
        )

        state_lock = Lock()

        for idx, stage in enumerate(stages):
            if idx > 0:
                logger.info("[FrontendAgent] Sleeping 2s between stages (rate limit) …")
                time.sleep(2)

            logger.info(
                "[FrontendAgent] Stage %d/%d — %d task(s) in parallel …",
                idx + 1, len(stages), len(stage),
            )

            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
                futures: list[Future] = []
                for task in stage:
                    task.status = "in_progress"
                    futures.append(pool.submit(self._process_task, task, ps, state_lock))

                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("[FrontendAgent] Unhandled task exception: %s", exc)
                        ps.log_error(self.name, f"Task exception: {exc}")

            ps.metrics.total_llm_calls += len(stage)

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)

        unwritten = [m for m in ps.code_manifest.unwritten() if m.agent == "frontend"]
        logger.info(
            "[FrontendAgent] Done in %.2fs. Written: %d  Unwritten: %d.",
            duration,
            len(fe_tasks) - len(unwritten),
            len(unwritten),
        )
        return self._compute_diff(before, ps)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Smoke-tests  (python agents/frontend.py)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    import os, sys, logging
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    if "open_router_api" in os.environ and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["open_router_api"]
    if "gimini_api" in os.environ:
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["gimini_api"]
        if not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = os.environ["gimini_api"]
    if "qrok_api" in os.environ and not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = os.environ["qrok_api"]
    if "nvidia_api" in os.environ and not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = os.environ["nvidia_api"]

    from core.state import (
        ProjectState, Task, TaskList, FileManifest, CodeManifest,
        ArchData, APIContract,
    )
    from providers.llm.base import BaseLLMProvider, LLMResponse

    print("\n" + "=" * 60)
    print("FrontendAgent — unit tests")
    print("=" * 60)

    # ── 1. Stage resolver ─────────────────────────────────────────────────────
    tasks = [
        Task(id="fe-001", agent="frontend", title="Global CSS", description="", depends_on=[], priority=1),
        Task(id="fe-002", agent="frontend", title="Button",     description="", depends_on=["fe-001"], priority=2),
        Task(id="fe-003", agent="frontend", title="Input",      description="", depends_on=["fe-001"], priority=2),
        Task(id="fe-004", agent="frontend", title="LoginPage",  description="", depends_on=["fe-002", "fe-003"], priority=3),
    ]
    stages = resolve_frontend_stages(tasks)
    assert len(stages) == 3, f"Expected 3 stages, got {len(stages)}"
    assert {t.id for t in stages[0]} == {"fe-001"}
    assert {t.id for t in stages[1]} == {"fe-002", "fe-003"}
    assert {t.id for t in stages[2]} == {"fe-004"}
    print("[PASS] resolve_frontend_stages — 3 stages, correct membership")

    # ── 2. System prompt contains tech stack name ──────────────────────────────
    sp = _build_system_prompt("Vue 3 + TypeScript")
    assert "Vue 3 + TypeScript" in sp
    print("[PASS] _build_system_prompt — tech stack injected")

    # ── 3. Mock integration run ────────────────────────────────────────────────
    import json as _json

    class MockLLM(BaseLLMProvider):
        @property
        def provider_name(self): return "mock"
        def complete(self, system_prompt, user_message, model, max_tokens=8192, temperature=0.2):
            assert "React 18 + TypeScript" in system_prompt, "tech stack missing from system prompt"
            # Determine which file is being generated from the "File to Generate" header
            is_css = user_message.startswith("## File to Generate\n- **Path**     : `output/frontend/global.css`")
            if is_css:
                code = ":root { --color-primary: #3b82f6; }\n* { box-sizing: border-box; margin: 0; }"
                payload = {"code": code, "classes": [], "functions": []}
            else:
                code = "export function LoginPage() { return <div className='login'>Login</div>; }"
                payload = {
                    "code": code, "classes": [],
                    "functions": [{"name": "LoginPage", "arguments": [], "return_type": "JSX.Element", "docstring": "Login page"}]
                }
            return LLMResponse(text=f"```json\n{_json.dumps(payload)}\n```",
                               input_tokens=10, output_tokens=10, model="mock", provider="mock")

    state = ProjectState(project_id="fe-test", project_name="FE Test")
    state.architecture = ArchData(
        tech_stack={"frontend": "React 18 + TypeScript", "backend": "FastAPI"},
        api_contracts=[
            APIContract(method="POST", path="/auth/login", auth_required=False,
                        description="Login", request_body={"email": "string", "password": "string"},
                        response_body={"token": "string"}),
        ],
    )
    state.tasks = TaskList(tasks=[
        Task(id="fe-001", agent="frontend", title="Global CSS",  description="", depends_on=[], priority=1),
        Task(id="fe-002", agent="frontend", title="Login Page",  description="", depends_on=["fe-001"], priority=2),
    ])
    state.code_manifest = CodeManifest(files={
        "output/frontend/global.css": FileManifest(
            agent="frontend", task_id="fe-001",
            filepath="output/frontend/global.css",
            purpose="Global stylesheet", exports=["styles"],
        ),
        "output/frontend/pages/LoginPage.tsx": FileManifest(
            agent="frontend", task_id="fe-002",
            filepath="output/frontend/pages/LoginPage.tsx",
            purpose="Login page component", exports=["LoginPage"],
        ),
    })

    agent = FrontendAgent(provider=MockLLM())
    result = agent.run(state)

    assert result.code_manifest.files["output/frontend/global.css"].written
    assert result.code_manifest.files["output/frontend/pages/LoginPage.tsx"].written
    assert "LoginPage" in result.code_manifest.files["output/frontend/pages/LoginPage.tsx"].functions[0].name
    assert ":root" in result.generated_code.frontend["output/frontend/global.css"]
    assert not result.errors, f"Unexpected errors: {result.errors}"
    print("[PASS] FrontendAgent mock run — both files written, CSS present, no errors")

    # Cleanup test output dirs
    import shutil
    test_out = os.path.join(_root, "output", "frontend")
    if os.path.exists(test_out):
        shutil.rmtree(test_out, ignore_errors=True)

    print("\n" + "=" * 60)
    print("All FrontendAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
