"""
tests/planner.py
────────────────
Integration + unit tests for the two-pass PlannerAgent.

PASS 1  — single skeleton LLM call (global visibility, imports_from graph).
PASS 2  — deterministic task generation via topological sort on the import graph.

Uses a MockLLM provider so the suite runs offline — no API keys, no network.
Run:  python tests/planner.py          (plain script)
      pytest tests/planner.py          (collects test_* functions)
"""

import os
import sys
import json

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_root, ".env"))

from core.state import (
    ProjectState,
    ArchData,
    Task,
    FileManifest,
    TaskList,
    CodeManifest,
    classify_complexity,
    build_persistence_contract,
)
from core.config import settings
from agents.planner import (
    PlannerAgent,
    _generate_tasks_from_skeleton,
    _inject_scaffold_tasks,
    _build_manifest,
    _has_cycle,
    _complexity_constraint,
)
from agents.schemas.planner import SkeletonSchema, SkeletonNodeSchema
from providers.llm.base import BaseLLMProvider, LLMResponse


class RecordingProvider(BaseLLMProvider):
    """Wraps a real provider and captures the last raw LLM response."""
    def __init__(self, inner: BaseLLMProvider):
        self._inner = inner
        self.last_raw = ""

    @property
    def provider_name(self) -> str:
        return f"recording({self._inner.provider_name})"

    def complete(self, system_prompt, user_message, model, max_tokens=8192,
                 temperature=0.2, **kwargs):
        response = self._inner.complete(system_prompt, user_message, model,
                                         max_tokens, temperature, **kwargs)
        self.last_raw = response.text
        return response


# ── Mock provider ──────────────────────────────────────────────────────────────

class MockSkeletonLLM(BaseLLMProvider):
    """Returns a fixed, self-consistent project skeleton on every call.

    Includes:
      - real import edges (be routes imports models + schemas)
      - a frontend→frontend edge (entry imports component)
      - ONE phantom import (be-svc imports a file not in the skeleton)
    """

    SKELETON = {
        "skeleton": [
            {
                "id": "db-001", "agent": "database", "title": "Todo model",
                "description": "SQLAlchemy model for todo_items table.",
                "file": "app/models/todo.py",
                "exports": ["TodoItem", "Base"], "imports_from": {},
            },
            {
                "id": "be-001", "agent": "backend", "title": "Todo schemas",
                "description": "Pydantic request/response schemas.",
                "file": "app/schemas/todo.py",
                "exports": ["TodoItemCreate", "TodoItemResponse"], "imports_from": {},
            },
            {
                "id": "be-002", "agent": "backend", "title": "Todo routes",
                "description": "FastAPI CRUD router.",
                "file": "app/api/routes/todo.py",
                "exports": ["router"],
                "imports_from": {
                    "app/models/todo.py": ["TodoItem"],
                    "app/schemas/todo.py": ["TodoItemCreate", "TodoItemResponse"],
                },
            },
            {
                "id": "be-003", "agent": "backend", "title": "Todo service",
                "description": "Business logic (imports a file that does NOT exist).",
                "file": "app/services/todo.py",
                "exports": ["TodoService"],
                "imports_from": {"app/models/nonexistent.py": ["Missing"]},
            },
            {
                "id": "fe-001", "agent": "frontend", "title": "Todo list component",
                "description": "Renders the todo list.",
                "file": "assets/js/todoList.js",
                "exports": ["TodoList"], "imports_from": {},
            },
            {
                "id": "fe-002", "agent": "frontend", "title": "App entry",
                "description": "Frontend entry point.",
                "file": "assets/js/main.js",
                "exports": ["App"],
                "imports_from": {"assets/js/todoList.js": ["TodoList"]},
            },
        ]
    }

    def __init__(self):
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "mock-skeleton"

    def complete(self, system_prompt, user_message, model, max_tokens=8192,
                 temperature=0.2, **kwargs):
        self.calls += 1
        self.last_user_message = user_message
        # The planner must use the skeleton system prompt (Pass 1).
        assert "imports_from" in system_prompt, "Pass-1 skeleton system prompt not used"
        return LLMResponse(
            text=json.dumps(self.SKELETON),
            input_tokens=100,
            output_tokens=50,
            model="mock",
            provider=self.provider_name,
        )


def _make_state() -> ProjectState:
    state = ProjectState(project_id="plan-test", project_name="Todo")
    state.requirements.raw_prompt = "Build a very simple todo app."
    state.requirements.functional_requirements = ["Manage todos"]
    state.architecture = ArchData(
        tech_stack={
            "backend": "FastAPI (Python 3.12)",
            "frontend": "HTML + Vanilla JS",
            "database": "SQLite 3",
        },
        api_contracts=[],
        database_schema_overview="todo_items (id, title, done)",
        complexity_tier="MINIMAL",
        persistence_contract=build_persistence_contract("SQLite 3"),
    )
    return state


# ── Pass-2 deterministic tests ────────────────────────────────────────────────

def test_pass2_import_graph_ordering():
    """depends_on must be derived from imports_from (topological order)."""
    warnings: list[str] = []
    tasks = _generate_tasks_from_skeleton(MockSkeletonLLM.SKELETON["skeleton"], warnings)
    by_id = {t.id: t for t in tasks}

    print("\n  PASS 2 (unit) — generated dependencies:")
    for t in sorted(tasks, key=lambda x: x.id):
        print(f"    {t.id:8} [{t.agent:8}] depends_on={t.depends_on or '[]'}")
    print(f"    warnings: {warnings}")

    assert by_id["db-001"].depends_on == []
    assert by_id["be-001"].depends_on == []
    assert by_id["be-002"].depends_on == ["be-001", "db-001"], by_id["be-002"].depends_on
    # Phantom import → dependency dropped + warning recorded
    assert by_id["be-003"].depends_on == []
    assert any("phantom import" in w for w in warnings), warnings
    # Frontend→frontend edge is preserved; no backend deps leaked into frontend
    assert by_id["fe-002"].depends_on == ["fe-001"], by_id["fe-002"].depends_on
    for t in tasks:
        if t.agent == "frontend":
            assert all(d.startswith("fe-") for d in t.depends_on)
    print("  [PASS] Pass-2 import-graph ordering (incl. phantom drop + FE edge)")


def test_pass2_priorities():
    warnings: list[str] = []
    tasks = _generate_tasks_from_skeleton(MockSkeletonLLM.SKELETON["skeleton"], warnings)
    for t in tasks:
        expected = {"database": 1, "backend": 2, "frontend": 2, "devops": 3}[t.agent]
        assert t.priority == expected, (t.id, t.priority)
    print("  [PASS] Pass-2 canonical priorities assigned")


def test_pass2_fe_backend_stripped():
    """A frontend file must never depend on a backend file."""
    skel = [
        {"id": "be-001", "agent": "backend", "title": "r", "file": "app/api/r.py",
         "exports": ["router"], "imports_from": {}},
        {"id": "fe-001", "agent": "frontend", "title": "f", "file": "assets/js/f.js",
         "exports": ["App"], "imports_from": {"app/api/r.py": ["router"]}},
    ]
    tasks = _generate_tasks_from_skeleton(skel, [])
    fe = next(t for t in tasks if t.agent == "frontend")
    assert fe.depends_on == [], fe.depends_on
    print("  [PASS] Pass-2 FE->BE dependency stripped")


def test_pass2_cycle_cleared():
    """A cycle in the import graph must be recovered by clearing deps."""
    skel = [
        {"id": "be-001", "agent": "backend", "title": "a", "file": "app/a.py",
         "exports": ["a"], "imports_from": {"app/b.py": ["b"]}},
        {"id": "be-002", "agent": "backend", "title": "b", "file": "app/b.py",
         "exports": ["b"], "imports_from": {"app/a.py": ["a"]}},
    ]
    tasks = _generate_tasks_from_skeleton(skel, [])
    assert not _has_cycle(tasks), "cycle not cleared"
    assert all(t.depends_on == [] for t in tasks)
    print("  [PASS] Pass-2 cycle detected + deps cleared")


# ── Scaffold / manifest tests ─────────────────────────────────────────────────

def test_scaffold_injection():
    tasks = [
        Task(id="be-001", agent="backend", title="s", description="", depends_on=[], priority=2),
        Task(id="fe-001", agent="frontend", title="f", description="", depends_on=[], priority=2),
        Task(id="db-001", agent="database", title="d", description="", depends_on=[], priority=1),
    ]
    manifest = {
        "output/backend/app/schemas/todo.py": FileManifest(agent="backend", task_id="be-001",
                                                   filepath="output/backend/app/schemas/todo.py", purpose="", exports=["TodoItemCreate"]),
        "output/frontend/assets/js/todoList.js": FileManifest(agent="frontend", task_id="fe-001",
                                                     filepath="output/frontend/assets/js/todoList.js", purpose="", exports=["TodoList"]),
        "output/backend/app/models/todo.py": FileManifest(agent="database", task_id="db-001",
                                                  filepath="output/backend/app/models/todo.py", purpose="", exports=["TodoItem"]),
    }
    next_tasks, next_manifest = _inject_scaffold_tasks(tasks, manifest)

    assert "output/backend/main.py" in next_manifest, "backend scaffold missing"
    assert "output/frontend/index.html" in next_manifest, "frontend scaffold missing"

    main_task = next(t for t in next_tasks if next_manifest["output/backend/main.py"].task_id == t.id)
    assert main_task.agent == "backend"
    assert main_task.depends_on == ["be-001"], main_task.depends_on  # runs after all backend

    index_task = next(t for t in next_tasks if next_manifest["output/frontend/index.html"].task_id == t.id)
    assert index_task.depends_on == ["fe-001"], index_task.depends_on
    print("  [PASS] Scaffold tasks injected (main.py + index.html), ordered last")


def test_build_manifest_db_model_dir():
    """Database fallback paths use the Persistence Contract's model directory (app/models/)."""
    tasks = [
        Task(id="db-001", agent="database", title="Todo model", description="", depends_on=[], priority=1),
        Task(id="be-001", agent="backend", title="Routes", description="", depends_on=["db-001"], priority=2),
    ]
    raw = [
        {"id": "db-001", "exports": ["TodoItem"], "description": "m"},
        {"id": "be-001", "exports": ["router"], "description": "r"},
    ]
    arch = ArchData(
        tech_stack={"database": "SQLite 3"},
        persistence_contract=build_persistence_contract("SQLite 3"),
    )
    mf = _build_manifest(tasks, raw, arch)

    db_paths = [p for p in mf if "models" in p]
    assert db_paths, "no db model path generated"
    assert all(p.endswith(".py") for p in db_paths), db_paths
    assert "prisma" not in " ".join(mf), "prisma fallback must be gone"
    assert mf[next(iter(db_paths))].agent == "database"
    print("  [PASS] _build_manifest DB fallback uses ORM model_dir (no prisma)")


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_skeleton_schema():
    nodes = MockSkeletonLLM.SKELETON["skeleton"][:2]
    ss = SkeletonSchema.model_validate({"skeleton": nodes})
    assert len(ss.skeleton) == 2
    assert isinstance(ss.skeleton[0], SkeletonNodeSchema)
    assert ss.skeleton[0].imports_from == {}
    print("  [PASS] SkeletonSchema parses skeleton + imports_from")


def test_complexity_constraint():
    assert "MINIMAL" in _complexity_constraint("minimal")
    assert "ENTERPRISE" in _complexity_constraint("enterprise")
    assert "STANDARD" in _complexity_constraint("STANDARD")
    assert "STANDARD" in _complexity_constraint("")  # unknown → default
    assert classify_complexity("build a very simple todo app") == "MINIMAL"
    print("  [PASS] complexity constraint renders per tier + classify_complexity")


# ── Persistence seam (ticket #13) ─────────────────────────────────────────────

SESSION_MODULE = "output/backend/db/session.py"


def _plan_with_database(database_tech: str = "SQLite 3") -> ProjectState:
    state = _make_state()
    state.architecture.tech_stack["database"] = database_tech
    state.architecture.persistence_contract = build_persistence_contract(database_tech)
    PlannerAgent(provider=MockSkeletonLLM()).run(state)
    return state


def test_stack_with_database_gets_connection_module_with_fixed_exports():
    manifest = _plan_with_database().code_manifest.files

    session = manifest[SESSION_MODULE]
    assert session.agent == "database"
    assert session.exports == ["engine", "SessionLocal", "get_db"]


def test_backend_entrypoint_imports_the_connection_module():
    manifest = _plan_with_database().code_manifest.files

    assert manifest["output/backend/main.py"].imports_from[SESSION_MODULE] == [
        "engine", "SessionLocal", "get_db",
    ]


def test_skeleton_planned_connection_module_is_taken_over_by_the_database_agent():
    state = _make_state()
    state.architecture.persistence_contract = build_persistence_contract("SQLite 3")
    skeleton = {"skeleton": MockSkeletonLLM.SKELETON["skeleton"] + [{
        "id": "be-009", "agent": "backend", "title": "DB session",
        "description": "Session helper.", "file": "db/session.py",
        "exports": ["get_session"], "imports_from": {},
    }]}
    mock = MockSkeletonLLM()
    mock.SKELETON = skeleton
    PlannerAgent(provider=mock).run(state)

    session = state.code_manifest.files[SESSION_MODULE]
    assert (session.agent, session.task_id) == ("database", "be-009")
    assert session.exports == ["engine", "SessionLocal", "get_db"]
    task = next(t for t in state.tasks.tasks if t.id == "be-009")
    assert task.agent == "database"
    assert task.title == "DB connection module"


def test_files_importing_a_taken_over_connection_module_get_the_fixed_exports():
    state = _make_state()
    state.architecture.persistence_contract = build_persistence_contract("SQLite 3")
    skeleton = {"skeleton": MockSkeletonLLM.SKELETON["skeleton"] + [
        {"id": "be-009", "agent": "backend", "title": "DB session", "description": "",
         "file": "db/session.py", "exports": ["get_session"], "imports_from": {}},
        {"id": "be-010", "agent": "backend", "title": "Repo", "description": "",
         "file": "app/repo.py", "exports": ["Repo"],
         "imports_from": {"db/session.py": ["get_session"]}},
    ]}
    mock = MockSkeletonLLM()
    mock.SKELETON = skeleton
    PlannerAgent(provider=mock).run(state)

    repo = state.code_manifest.files["output/backend/app/repo.py"]
    assert repo.imports_from[SESSION_MODULE] == ["engine", "SessionLocal", "get_db"]


def test_backend_entrypoint_depends_on_the_connection_module_task():
    state = _plan_with_database()
    manifest = state.code_manifest.files
    main_task = next(t for t in state.tasks.tasks if t.id == manifest["output/backend/main.py"].task_id)

    assert manifest[SESSION_MODULE].task_id in main_task.depends_on


def test_env_example_is_deterministic_content_from_the_contract():
    env = _plan_with_database("PostgreSQL 16").code_manifest.files["output/backend/.env.example"]

    assert env.planned_content is not None, ".env.example must not be left to the LLM"
    assert "DATABASE_URL=postgresql://" in env.planned_content


def test_skeleton_prompt_carries_the_persistence_contract():
    from core.state import render_persistence_contract

    state = _make_state()
    state.architecture.persistence_contract = build_persistence_contract("SQLite 3")
    mock = MockSkeletonLLM()
    PlannerAgent(provider=mock).run(state)

    assert render_persistence_contract(state.architecture.persistence_contract) in mock.last_user_message


# ── End-to-end two-pass run ───────────────────────────────────────────────────

def test_full_two_pass_run():
    state = _make_state()
    mock = MockSkeletonLLM()
    agent = PlannerAgent(provider=mock)
    agent.run(state)  # mutates `state` directly (run returns a diff dict)

    assert mock.calls == 1, f"two-pass planner must make exactly 1 LLM call, got {mock.calls}"

    # ── PASS 1 result: the skeleton the planner received ───────────────────
    print("\n  PASS 1 — skeleton (from the single LLM call):")
    for node in mock.SKELETON["skeleton"]:
        imports = ", ".join(f"{f}->{n}" for f, names in node["imports_from"].items()
                            for n in names) or "(none)"
        print(f"    {node['id']:8} [{node['agent']:8}] {node['file']:<30} exports={node['exports']}  imports={imports}")

    # ── PASS 2 result: generated tasks + their dependencies ────────────────
    tasks = state.tasks.tasks
    manifest = state.code_manifest.files
    print(f"\n  PASS 2 — {len(tasks)} tasks with dependencies (import-graph ordering):")
    for t in sorted(tasks, key=lambda x: x.id):
        print(f"    {t.id:8} [{t.agent:8}] depends_on={t.depends_on or '[]'}")

    # All 6 skeleton files + 2 scaffolds + 3 pinned project files
    # (requirements.txt, README.md, backend .env.example) from Phase C
    # + the DB connection module every run's Persistence Contract implies.
    assert len(tasks) == 12, f"expected 12 tasks (6 skeleton + 2 scaffold + 3 project files + connection module), got {len(tasks)}"
    assert len(manifest) == 12, len(manifest)
    assert "output/backend/main.py" in manifest
    assert "output/frontend/index.html" in manifest
    assert "output/backend/requirements.txt" in manifest
    assert "output/README.md" in manifest
    assert "output/backend/.env.example" in manifest

    by_id = {t.id: t for t in tasks}
    assert by_id["be-002"].depends_on == ["be-001", "db-001"]
    assert by_id["be-003"].depends_on == []  # phantom import dropped

    # Scaffold ordering
    main_tid = manifest["output/backend/main.py"].task_id
    session_tid = manifest["output/backend/db/session.py"].task_id
    assert by_id[main_tid].depends_on == ["be-001", "be-002", "be-003", session_tid]

    # No FE->BE deps, no cycles, all deps valid
    be_ids = {t.id for t in tasks if t.agent == "backend"}
    for t in tasks:
        if t.agent == "frontend":
            assert not (set(t.depends_on) & be_ids)
    assert not _has_cycle(tasks)
    valid = {t.id for t in tasks}
    for t in tasks:
        assert all(d in valid for d in t.depends_on)

    # The only recorded error is the (intentional) phantom-import warning.
    phantom_msgs = [e["message"] for e in state.errors if "phantom import" in e["message"]]
    unexpected = [e for e in state.errors if "phantom import" not in e["message"]]
    assert len(phantom_msgs) == 1, state.errors
    assert not unexpected, unexpected
    print(f"\n  [PASS] Full two-pass run: 1 LLM call, {len(tasks)} tasks, imports + scaffolds OK")


def _extract_skeleton_from_raw(raw: str) -> dict | None:
    """Try to extract the skeleton dict from the LLM's raw response text."""
    try:
        import json_repair
        stripped = raw.strip()
        if stripped.startswith("```"):
            start = stripped.find("{")
            if start >= 0:
                stripped = stripped[start:]
            end = stripped.rfind("}")
            if end >= 0:
                stripped = stripped[:end+1]
        return json_repair.loads(stripped)
    except Exception:
        return None


def test_real_llm_skeleton():
    """Makes a REAL LLM call and prints the skeleton + file dependencies."""
    print("\n" + "=" * 60)
    print("REAL LLM TWO-PASS TEST (requires API keys in .env)")
    print("=" * 60)

    # Build a compact but representative state
    state = _make_state()
    state.requirements.raw_prompt = (
        "Build a simple todo list with FastAPI backend and vanilla JS frontend. "
        "SQLite database. Just CRUD operations on todos."
    )
    state.requirements.functional_requirements = [
        "List all todos",
        "Create a new todo",
        "Mark todo complete/incomplete",
        "Delete a todo",
    ]

    # Wrap the real provider to capture the raw response
    from providers.llm.llmFactory import LLMFactory as llmFactory
    inner = llmFactory.get(settings.default_provider)
    recorder = RecordingProvider(inner)

    agent = PlannerAgent(provider=recorder, model=settings.model_name)
    agent.run(state)

    # ── PASS 1: Print the actual skeleton the LLM produced ──────────────
    print("\n  PASS 1 — skeleton returned by the LLM:")
    skel = _extract_skeleton_from_raw(recorder.last_raw)
    if skel and "skeleton" in skel:
        for node in skel["skeleton"]:
            imports = ", ".join(
                f"{src}->{names}" for src, names in node.get("imports_from", {}).items()
            ) or "(none)"
            print(f"    {node['id']:8} [{node['agent']:8}] {node['file']:<35} exports={node['exports']}  imports={imports}")
    else:
        print("    (could not parse skeleton from raw response)")

    # ── PASS 2: Print the resulting file skeleton + dependencies ────────
    tasks = state.tasks.tasks
    manifest = state.code_manifest.files
    print(f"\n  PASS 2 — {len(tasks)} tasks with dependencies:")
    for t in sorted(tasks, key=lambda x: x.id):
        print(f"    {t.id:8} [{t.agent:8}] depends_on={t.depends_on or '[]'}")

    print("\n  File skeleton (from manifest):")
    for path, fm in sorted(manifest.items()):
        print(f"    {path:<40} [{fm.agent:8}] exports={fm.exports}")

    # Basic sanity checks
    assert len(tasks) > 0
    assert any("main.py" in p for p in manifest), "backend entrypoint missing"
    assert any("index.html" in p for p in manifest), "frontend entrypoint missing"
    assert not _has_cycle(tasks)

    print("\n  [PASS] Real LLM run complete — skeleton + deps printed above")


def test_run_inference_works():
    """The planner should fall back to prefix-based agent inference when agent is blank."""
    skel = {"skeleton": [{"id": "be-001", "file": "app/x.py", "title": "x",
                          "exports": [], "imports_from": {}}]}
    warnings: list[str] = []
    tasks = _generate_tasks_from_skeleton(skel["skeleton"], warnings)
    assert tasks and tasks[0].agent == "backend", "agent not inferred from be- prefix"
    print("  [PASS] Pass-2 infers agent from ID prefix")


def main() -> None:
    print("=" * 60)
    print("PlannerAgent — two-pass architecture tests")
    print("=" * 60)
    test_pass2_import_graph_ordering()
    test_pass2_priorities()
    test_pass2_fe_backend_stripped()
    test_pass2_cycle_cleared()
    test_scaffold_injection()
    test_build_manifest_db_model_dir()
    test_skeleton_schema()
    test_complexity_constraint()
    test_full_two_pass_run()
    test_real_llm_skeleton()  # REAL LLM call — requires API keys
    test_run_inference_works()
    print("\n" + "=" * 60)
    print("ALL PlannerAgent TESTS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
