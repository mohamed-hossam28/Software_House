"""
The two-pass Planner: one skeleton LLM call (Pass 1), then deterministic task
generation from the skeleton's import graph (Pass 2). The LLM is the scripted
fake answering with the todo skeleton fixture.
"""
from core.state import (
    ArchData,
    FileManifest,
    ProjectState,
    Task,
    build_persistence_contract,
    classify_complexity,
    render_persistence_contract,
)
from agents.planner import (
    PlannerAgent,
    _build_manifest,
    _complexity_constraint,
    _generate_tasks_from_skeleton,
    _has_cycle,
    _inject_scaffold_tasks,
)
from agents.schemas.planner import SkeletonNodeSchema, SkeletonSchema
from tests.fixtures.scripted_llm import TODO_SKELETON, scripted_planner

SESSION_MODULE = "output/backend/db/session.py"


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


def _plan_with_database(database_tech: str = "SQLite 3") -> ProjectState:
    state = _make_state()
    state.architecture.tech_stack["database"] = database_tech
    state.architecture.persistence_contract = build_persistence_contract(database_tech)
    PlannerAgent(provider=scripted_planner()).run(state)
    return state


# ── Pass 2: tasks from the import graph ───────────────────────────────────────

def test_pass2_derives_depends_on_from_the_import_graph():
    warnings: list[str] = []
    by_id = {t.id: t for t in _generate_tasks_from_skeleton(TODO_SKELETON["skeleton"], warnings)}

    assert by_id["db-001"].depends_on == []
    assert by_id["be-001"].depends_on == []
    assert by_id["be-002"].depends_on == ["be-001", "db-001"]
    assert by_id["fe-002"].depends_on == ["fe-001"]


def test_pass2_drops_a_phantom_import_and_warns():
    warnings: list[str] = []
    by_id = {t.id: t for t in _generate_tasks_from_skeleton(TODO_SKELETON["skeleton"], warnings)}

    assert by_id["be-003"].depends_on == []
    assert any("phantom import" in w for w in warnings), warnings


def test_pass2_assigns_canonical_priorities():
    tasks = _generate_tasks_from_skeleton(TODO_SKELETON["skeleton"], [])

    for t in tasks:
        expected = {"database": 1, "backend": 2, "frontend": 2, "devops": 3}[t.agent]
        assert t.priority == expected, (t.id, t.priority)


def test_pass2_strips_a_frontend_dependency_on_the_backend():
    skel = [
        {"id": "be-001", "agent": "backend", "title": "r", "file": "app/api/r.py",
         "exports": ["router"], "imports_from": {}},
        {"id": "fe-001", "agent": "frontend", "title": "f", "file": "assets/js/f.js",
         "exports": ["App"], "imports_from": {"app/api/r.py": ["router"]}},
    ]
    fe = next(t for t in _generate_tasks_from_skeleton(skel, []) if t.agent == "frontend")

    assert fe.depends_on == []


def test_pass2_clears_the_dependencies_of_a_cycle():
    skel = [
        {"id": "be-001", "agent": "backend", "title": "a", "file": "app/a.py",
         "exports": ["a"], "imports_from": {"app/b.py": ["b"]}},
        {"id": "be-002", "agent": "backend", "title": "b", "file": "app/b.py",
         "exports": ["b"], "imports_from": {"app/a.py": ["a"]}},
    ]
    tasks = _generate_tasks_from_skeleton(skel, [])

    assert not _has_cycle(tasks)
    assert all(t.depends_on == [] for t in tasks)


def test_pass2_infers_the_agent_from_the_id_prefix():
    skel = [{"id": "be-001", "file": "app/x.py", "title": "x", "exports": [], "imports_from": {}}]

    tasks = _generate_tasks_from_skeleton(skel, [])

    assert tasks and tasks[0].agent == "backend"


# ── Scaffolds and manifest ────────────────────────────────────────────────────

def test_scaffold_tasks_are_injected_and_ordered_last():
    tasks = [
        Task(id="be-001", agent="backend", title="s", description="", depends_on=[], priority=2),
        Task(id="fe-001", agent="frontend", title="f", description="", depends_on=[], priority=2),
        Task(id="db-001", agent="database", title="d", description="", depends_on=[], priority=1),
    ]
    manifest = {
        "output/backend/app/schemas/todo.py": FileManifest(
            agent="backend", task_id="be-001", filepath="output/backend/app/schemas/todo.py",
            purpose="", exports=["TodoItemCreate"]),
        "output/frontend/assets/js/todoList.js": FileManifest(
            agent="frontend", task_id="fe-001", filepath="output/frontend/assets/js/todoList.js",
            purpose="", exports=["TodoList"]),
        "output/backend/app/models/todo.py": FileManifest(
            agent="database", task_id="db-001", filepath="output/backend/app/models/todo.py",
            purpose="", exports=["TodoItem"]),
    }

    next_tasks, next_manifest = _inject_scaffold_tasks(tasks, manifest)

    main_task = next(t for t in next_tasks if next_manifest["output/backend/main.py"].task_id == t.id)
    assert main_task.agent == "backend"
    assert main_task.depends_on == ["be-001"]
    index_task = next(t for t in next_tasks if next_manifest["output/frontend/index.html"].task_id == t.id)
    assert index_task.depends_on == ["fe-001"]


def test_database_fallback_paths_use_the_persistence_contract_model_dir():
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
    assert db_paths
    assert all(p.endswith(".py") for p in db_paths), db_paths
    assert "prisma" not in " ".join(mf)
    assert mf[db_paths[0]].agent == "database"


# ── Schema and complexity ─────────────────────────────────────────────────────

def test_skeleton_schema_parses_nodes_and_their_imports():
    ss = SkeletonSchema.model_validate({"skeleton": TODO_SKELETON["skeleton"][:2]})

    assert len(ss.skeleton) == 2
    assert isinstance(ss.skeleton[0], SkeletonNodeSchema)
    assert ss.skeleton[0].imports_from == {}


def test_complexity_constraint_renders_per_tier():
    assert "MINIMAL" in _complexity_constraint("minimal")
    assert "ENTERPRISE" in _complexity_constraint("enterprise")
    assert "STANDARD" in _complexity_constraint("STANDARD")
    assert "STANDARD" in _complexity_constraint("")  # unknown → default
    assert classify_complexity("build a very simple todo app") == "MINIMAL"


# ── The persistence seam ──────────────────────────────────────────────────────

def test_stack_with_database_gets_connection_module_with_fixed_exports():
    session = _plan_with_database().code_manifest.files[SESSION_MODULE]

    assert session.agent == "database"
    assert session.exports == ["engine", "SessionLocal", "get_db"]


def test_backend_entrypoint_imports_the_connection_module():
    manifest = _plan_with_database().code_manifest.files

    assert manifest["output/backend/main.py"].imports_from[SESSION_MODULE] == [
        "engine", "SessionLocal", "get_db",
    ]


def test_skeleton_planned_connection_module_is_taken_over_by_the_database_agent():
    state = _make_state()
    skeleton = {"skeleton": TODO_SKELETON["skeleton"] + [{
        "id": "be-009", "agent": "backend", "title": "DB session",
        "description": "Session helper.", "file": "db/session.py",
        "exports": ["get_session"], "imports_from": {},
    }]}
    PlannerAgent(provider=scripted_planner(skeleton)).run(state)

    session = state.code_manifest.files[SESSION_MODULE]
    assert (session.agent, session.task_id) == ("database", "be-009")
    assert session.exports == ["engine", "SessionLocal", "get_db"]
    task = next(t for t in state.tasks.tasks if t.id == "be-009")
    assert task.agent == "database"
    assert task.title == "DB connection module"


def test_files_importing_a_taken_over_connection_module_get_the_fixed_exports():
    state = _make_state()
    skeleton = {"skeleton": TODO_SKELETON["skeleton"] + [
        {"id": "be-009", "agent": "backend", "title": "DB session", "description": "",
         "file": "db/session.py", "exports": ["get_session"], "imports_from": {}},
        {"id": "be-010", "agent": "backend", "title": "Repo", "description": "",
         "file": "app/repo.py", "exports": ["Repo"],
         "imports_from": {"db/session.py": ["get_session"]}},
    ]}
    PlannerAgent(provider=scripted_planner(skeleton)).run(state)

    repo = state.code_manifest.files["output/backend/app/repo.py"]
    assert repo.imports_from[SESSION_MODULE] == ["engine", "SessionLocal", "get_db"]


def _plan_skeleton_extra(*extra: dict) -> ProjectState:
    state = _make_state()
    PlannerAgent(provider=scripted_planner({"skeleton": TODO_SKELETON["skeleton"] + list(extra)})).run(state)
    return state


def _backend_files_named(manifest: dict, name: str) -> list[str]:
    return sorted(p for p in manifest if p.startswith("output/backend/") and p.endswith("/" + name))


def test_skeleton_entrypoint_under_app_becomes_the_one_backend_entrypoint():
    state = _plan_skeleton_extra({
        "id": "be-009", "agent": "backend", "title": "App", "description": "FastAPI app.",
        "file": "app/main.py", "exports": ["app"],
        "imports_from": {"app/api/routes/todo.py": ["router"]},
    })
    manifest = state.code_manifest.files

    assert _backend_files_named(manifest, "main.py") == ["output/backend/main.py"]
    main = manifest["output/backend/main.py"]
    assert main.task_id == "be-009"
    assert main.imports_from["output/backend/app/api/routes/todo.py"] == ["router"]
    assert main.imports_from[SESSION_MODULE] == ["engine", "SessionLocal", "get_db"]
    assert not any(t.title == "Backend entrypoint (main.py)" for t in state.tasks.tasks)


def test_skeleton_connection_module_under_app_becomes_the_one_connection_module():
    state = _plan_skeleton_extra(
        {"id": "db-009", "agent": "database", "title": "DB", "description": "",
         "file": "app/db/database.py", "exports": ["engine", "SessionLocal", "Base"],
         "imports_from": {}},
        {"id": "be-010", "agent": "backend", "title": "Repo", "description": "",
         "file": "app/repo.py", "exports": ["Repo"],
         "imports_from": {"app/db/database.py": ["SessionLocal"]}},
    )
    manifest = state.code_manifest.files

    assert "output/backend/app/db/database.py" not in manifest
    session = manifest[SESSION_MODULE]
    assert (session.agent, session.task_id) == ("database", "db-009")
    repo = manifest["output/backend/app/repo.py"]
    assert repo.imports_from == {SESSION_MODULE: ["engine", "SessionLocal", "get_db"]}
    assert session.planned_content is not None


def test_a_dependency_module_exporting_only_get_db_is_not_taken_as_the_connection_module():
    state = _plan_skeleton_extra({
        "id": "be-009", "agent": "backend", "title": "Deps", "description": "",
        "file": "app/api/deps.py", "exports": ["get_db", "get_current_user"], "imports_from": {},
    })
    manifest = state.code_manifest.files

    assert manifest["output/backend/app/api/deps.py"].exports == ["get_db", "get_current_user"]
    assert manifest[SESSION_MODULE].task_id != "be-009"


def test_backend_entrypoint_depends_on_the_connection_module_task():
    state = _plan_with_database()
    manifest = state.code_manifest.files
    main_task = next(t for t in state.tasks.tasks if t.id == manifest["output/backend/main.py"].task_id)

    assert manifest[SESSION_MODULE].task_id in main_task.depends_on


def test_env_example_is_planned_content_from_the_contract():
    env = _plan_with_database("PostgreSQL 16").code_manifest.files["output/backend/.env.example"]

    assert env.planned_content is not None, ".env.example must not be left to the LLM"
    assert "DATABASE_URL=postgresql://" in env.planned_content


def test_skeleton_request_carries_the_persistence_contract():
    state = _make_state()
    fake = scripted_planner()
    PlannerAgent(provider=fake).run(state)

    assert render_persistence_contract(state.architecture.persistence_contract) in fake.requests[0].user_message


# ── A full two-pass run ───────────────────────────────────────────────────────

def test_full_two_pass_run_makes_one_call_and_plans_every_file():
    state = _make_state()
    fake = scripted_planner()

    PlannerAgent(provider=fake).run(state)

    assert len(fake.requests) == 1
    tasks = state.tasks.tasks
    manifest = state.code_manifest.files
    # 6 skeleton files + 2 scaffolds + 3 pinned project files + the connection module.
    assert len(tasks) == 12
    assert len(manifest) == 12
    for path in ("output/backend/main.py", "output/frontend/index.html",
                 "output/backend/requirements.txt", "output/README.md", "output/backend/.env.example"):
        assert path in manifest

    by_id = {t.id: t for t in tasks}
    assert by_id["be-002"].depends_on == ["be-001", "db-001"]
    assert by_id["be-003"].depends_on == []
    main_tid = manifest["output/backend/main.py"].task_id
    session_tid = manifest[SESSION_MODULE].task_id
    assert by_id[main_tid].depends_on == ["be-001", "be-002", "be-003", session_tid]

    be_ids = {t.id for t in tasks if t.agent == "backend"}
    assert not any(set(t.depends_on) & be_ids for t in tasks if t.agent == "frontend")
    assert not _has_cycle(tasks)
    valid = {t.id for t in tasks}
    assert all(d in valid for t in tasks for d in t.depends_on)

    # The only recorded error is the skeleton's intentional phantom import.
    assert [e for e in state.errors if "phantom import" in e["message"]]
    assert not [e for e in state.errors if "phantom import" not in e["message"]], state.errors
