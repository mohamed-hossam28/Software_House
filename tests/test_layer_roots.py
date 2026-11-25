"""
Fixed Layer roots: whatever layout the skeleton LLM picks, the Planner re-roots
every path under its owning coding agent's Layer root (`output/backend/`,
`output/frontend/`), and the Import Contract agrees with the Persistence Contract.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from agents.coding_loop import build_import_contract
from agents.planner import PlannerAgent
from core.state import ArchData, ProjectState, build_persistence_contract, connection_import_line
from core.workspace import connection_module_path
from tests.fixtures.scripted_llm import scripted_planner


def skeleton(backend_prefix: str, frontend_prefix: str) -> dict:
    """A small skeleton with every path written under the given prefixes."""
    be, fe = backend_prefix, frontend_prefix
    return {"skeleton": [
        {"id": "db-001", "agent": "database", "title": "Todo model", "description": "",
         "file": f"{be}app/models/todo.py", "exports": ["TodoItem"], "imports_from": {}},
        {"id": "be-001", "agent": "backend", "title": "Todo service", "description": "",
         "file": f"{be}app/services/todo.py", "exports": ["list_todos"],
         "imports_from": {f"{be}app/models/todo.py": ["TodoItem"]}},
        {"id": "be-002", "agent": "backend", "title": "Todo routes", "description": "",
         "file": f"{be}app/api/routes/todo.py", "exports": ["router"],
         "imports_from": {f"{be}app/services/todo.py": ["list_todos"]}},
        {"id": "fe-001", "agent": "frontend", "title": "Todo list", "description": "",
         "file": f"{fe}js/todoList.js", "exports": ["TodoList"], "imports_from": {}},
        {"id": "fe-002", "agent": "frontend", "title": "App", "description": "",
         "file": f"{fe}js/main.js", "exports": ["App"],
         "imports_from": {f"{fe}js/todoList.js": ["TodoList"]}},
    ]}


def plan(skel: dict, database_tech: str = "SQLite 3") -> ProjectState:
    state = ProjectState(project_id="layer-roots", project_name="Todo")
    state.requirements.raw_prompt = "Build a very simple todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "frontend": "HTML + Vanilla JS",
                    "database": database_tech},
        persistence_contract=build_persistence_contract(database_tech),
    )
    PlannerAgent(provider=scripted_planner(skel)).run(state)
    return state


#: Where the skeleton above must land, whatever prefix the LLM used.
EXPECTED_SKELETON_PATHS = {
    "output/backend/app/models/todo.py",
    "output/backend/app/services/todo.py",
    "output/backend/app/api/routes/todo.py",
    "output/frontend/js/todoList.js",
    "output/frontend/js/main.js",
}


@pytest.mark.parametrize(
    ("backend_prefix", "frontend_prefix"),
    [
        ("output/", "output/"),              # output/app/…
        ("backend/", "frontend/"),           # backend/app/…
        ("", ""),                            # bare app/…
        ("output/backend/", "output/frontend/"),
    ],
    ids=["output-prefixed", "layer-prefixed", "bare", "already-rooted"],
)
def test_every_skeleton_layout_lands_under_the_fixed_layer_roots(backend_prefix, frontend_prefix):
    state = plan(skeleton(backend_prefix, frontend_prefix))
    manifest = state.code_manifest.files

    assert EXPECTED_SKELETON_PATHS <= set(manifest)
    for path, fm in manifest.items():
        root = "output/frontend/" if fm.agent == "frontend" else "output/backend/"
        assert path.startswith(root), (path, fm.agent)
    assert manifest["output/backend/app/services/todo.py"].imports_from == {
        "output/backend/app/models/todo.py": ["TodoItem"],
    }
    assert manifest["output/frontend/js/main.js"].imports_from == {
        "output/frontend/js/todoList.js": ["TodoList"],
    }
    assert not [e for e in state.errors if "Layer root" in e["message"]]


def test_path_prefixed_with_the_other_layer_is_re_rooted_and_logged():
    skel = skeleton("backend/", "frontend/")
    skel["skeleton"][1]["file"] = "frontend/app/services/todo.py"   # backend file, wrong layer
    skel["skeleton"][2]["imports_from"] = {"frontend/app/services/todo.py": ["list_todos"]}

    state = plan(skel)
    manifest = state.code_manifest.files

    assert manifest["output/backend/app/services/todo.py"].agent == "backend"
    assert manifest["output/backend/app/api/routes/todo.py"].imports_from == {
        "output/backend/app/services/todo.py": ["list_todos"],
    }
    assert not any(p.startswith("output/frontend/app/") for p in manifest)
    errors = [e["message"] for e in state.errors if "frontend/app/services/todo.py" in e["message"]]
    assert len(errors) == 1, state.errors


def test_entrypoints_project_files_and_connection_module_sit_at_fixed_places():
    manifest = plan(skeleton("", "")).code_manifest.files

    assert manifest["output/backend/main.py"].agent == "backend"
    assert manifest["output/frontend/index.html"].agent == "frontend"
    assert manifest["output/backend/requirements.txt"].agent == "backend"
    assert manifest["output/backend/.env.example"].agent == "backend"
    assert manifest["output/backend/db/session.py"].agent == "database"


@pytest.mark.parametrize("backend_prefix", ["output/", "backend/", ""])
def test_import_contract_for_the_connection_module_matches_the_persistence_contract(backend_prefix):
    skel = skeleton(backend_prefix, "frontend/")
    skel["skeleton"].append({   # the skeleton plans the connection module itself
        "id": "be-009", "agent": "backend", "title": "DB session", "description": "",
        "file": f"{backend_prefix}db/session.py", "exports": ["get_session"], "imports_from": {},
    })
    skel["skeleton"][1]["imports_from"][f"{backend_prefix}db/session.py"] = ["get_session"]
    state = plan(skel)
    contract = state.architecture.persistence_contract
    manifest = state.code_manifest
    session = connection_module_path(contract)

    importers = [p for p, fm in manifest.files.items() if session in fm.imports_from]
    assert "output/backend/main.py" in importers
    assert "output/backend/app/services/todo.py" in importers
    for path in importers:
        assert f"- `{connection_import_line(contract)}`" in build_import_contract(manifest, path), path


def test_connection_module_workspace_path_comes_from_the_persistence_contract():
    contract = build_persistence_contract("PostgreSQL 16")

    assert connection_module_path(contract) == "output/backend/db/session.py"
    assert connection_module_path(contract.model_copy(update={"connection_module": "core/db.py"})) == (
        "output/backend/core/db.py"
    )


def test_single_file_model_module_feeds_create_schema_and_base(tmp_path, monkeypatch):
    """A single-module layout (`db/models.py`) is still a model module: the
    connection module creates its tables and re-exports its Base."""
    skel = {"skeleton": [
        {"id": "db-001", "agent": "database", "title": "Models", "description": "",
         "file": "backend/db/models.py", "exports": ["Base", "Todo"], "imports_from": {}},
        {"id": "be-001", "agent": "backend", "title": "Todo routes", "description": "",
         "file": "backend/api/todo.py", "exports": ["router"],
         "imports_from": {"backend/db/models.py": ["Todo"]}},
    ]}
    session = plan(skel).code_manifest.files["output/backend/db/session.py"].planned_content

    db = tmp_path / "db"
    db.mkdir()
    (db / "__init__.py").write_text("", encoding="utf-8")
    (db / "session.py").write_text(session, encoding="utf-8")
    (db / "models.py").write_text(
        "from sqlalchemy import Column, Integer\n"
        "from sqlalchemy.orm import declarative_base\n"
        "Base = declarative_base()\n"
        "class Todo(Base):\n"
        "    __tablename__ = 'todos'\n"
        "    id = Column(Integer, primary_key=True)\n",
        encoding="utf-8",
    )
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == "db" or m.startswith("db.")]:
        monkeypatch.delitem(sys.modules, name)

    import importlib
    db_session = importlib.import_module("db.session")
    db_session.create_schema()

    from sqlalchemy import inspect
    assert db_session.Base is importlib.import_module("db.models").Base
    assert inspect(db_session.engine).get_table_names() == ["todos"]
    db_session.engine.dispose()
