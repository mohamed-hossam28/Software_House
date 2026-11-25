"""
Pipeline graph builder: the real pipeline graph, run end to end with a scripted
fake LLM for every agent and a temporary workspace. The main end-to-end test seam.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import ProjectState
from core.workspace import PROJECT_DIR, layer_root
from graph.graph import AgentProviders, build_graph
from providers.llm.scripted import ScriptedLLMProvider
from tests.test_persistence_contract import TODO_MODEL_SOURCE
from tests.fixtures.scripted_llm import (
    STUB_REPLIES,
    TODO_LLM_FILES,
    scripted_coding_agent,
    scripted_planner,
)

#: Requirements script: a complete SRS that asks no clarifying questions.
COMPLETE_SRS = {
    "functional_requirements": ["User can create, list and delete todo items"],
    "non_functional_requirements": ["Responses under 200ms"],
    "constraints": ["Store todos in SQLite"],
    "assumptions": ["Single user"],
    "out_of_scope": ["Mobile app"],
    "clarifying_questions": [],
}

#: Architecture script: one reply that satisfies each of its three sub-calls.
ARCHITECTURE = {
    "tech_stack": {"backend": "FastAPI (Python 3.12)", "frontend": "HTML + Vanilla JS"},
    "system_components": ["API", "Web UI", "Database"],
    "security_requirements": ["Validate input"],
    "database_schema_overview": "todo_items(id INTEGER PK, title TEXT, done BOOLEAN)",
    "api_contracts": [{
        "method": "GET", "path": "/api/todos", "auth_required": False,
        "description": "List todos", "request_body": {}, "response_body": {"items": "list"},
    }],
}

ROUTES = "output/backend/app/api/routes/todo.py"
REQUIREMENTS_TXT = "output/backend/requirements.txt"
README = "output/README.md"

#: The docs node's reply: the README, written after integration.
README_REPLY = {**STUB_REPLIES["backend"], "code": "# TodoGraph\n\nRun: uvicorn main:app\n"}

#: Backend replies that pass the Integration checks, so Integration needs no repair call.
BACKEND_REPLIES = {
    ROUTES: {**STUB_REPLIES["backend"], "code": (
        "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
        "@router.get(\"/api/todos\")\ndef list_todos():\n    return {\"items\": []}\n"
    )},
    REQUIREMENTS_TXT: {**STUB_REPLIES["backend"], "code": "fastapi\nsqlalchemy\nuvicorn\n"},
}


#: The database model reply defines the Base and TodoItem the skeleton promises.
DATABASE_REPLIES = {
    "output/backend/app/models/todo.py": {"migration_code": TODO_MODEL_SOURCE, "functions": []},
}


def scripted_providers(backend_replies: dict | None = None, integration_calls: list | None = None) -> AgentProviders:
    backend = scripted_coding_agent("backend", TODO_LLM_FILES["backend"])
    backend.files.update(BACKEND_REPLIES if backend_replies is None else backend_replies)
    database = scripted_coding_agent("database", TODO_LLM_FILES["database"])
    database.files.update(DATABASE_REPLIES)
    return AgentProviders(
        requirements=ScriptedLLMProvider("requirements", calls=[COMPLETE_SRS]),
        architecture=ScriptedLLMProvider("architecture", calls=[ARCHITECTURE] * 3),
        planner=scripted_planner(),
        database=database,
        backend=backend,
        frontend=scripted_coding_agent("frontend", TODO_LLM_FILES["frontend"]),
        integration=ScriptedLLMProvider("integration", calls=integration_calls or []),
        docs=ScriptedLLMProvider("docs", files={README: README_REPLY}),
    )


def _real_output_snapshot() -> dict[str, float]:
    out = PROJECT_DIR / "output"
    if not out.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in out.rglob("*") if p.is_file()}


def run_pipeline(graph, state: ProjectState):
    config = {"configurable": {"thread_id": state.project_id}}
    visited = [node for event in graph.stream(state, config=config) for node in event]
    final = graph.get_state(config).values
    final = final if isinstance(final, ProjectState) else ProjectState(**final)
    return visited, final


def test_graph_runs_end_to_end_on_scripted_fakes_inside_the_temporary_workspace(workspace, tmp_path):
    before = _real_output_snapshot()
    providers = scripted_providers()
    graph = build_graph(providers, workspace)
    state = ProjectState(project_id="graph-e2e", project_name="TodoGraph")
    state.requirements.raw_prompt = "Build a todo app that stores todos in SQLite."

    visited, final = run_pipeline(graph, state)

    assert "human" not in visited
    assert visited[-3:] == ["integration", "docs", "finalize"]
    assert {"requirements", "architecture", "planner", "database", "backend",
            "frontend"} <= set(visited)
    assert not [e for e in final.errors if "scripted fake" in e["message"]], final.errors

    manifest = final.code_manifest.files
    assert manifest
    unwritten = [path for path, fm in manifest.items() if not fm.written]
    assert not unwritten, unwritten
    for path, fm in manifest.items():
        assert (tmp_path / path).is_file(), path
        if path != README:  # the one project-level file, written by the docs node
            assert path.startswith(layer_root(fm.agent)), (fm.agent, path)

    # Every Coding agent was asked for exactly its scripted files, nothing else.
    for agent in ("database", "backend", "frontend"):
        asked = {p for request in getattr(providers, agent).requests for p in request.files}
        assert asked == set(TODO_LLM_FILES[agent]), agent
    assert providers.integration.requests == []
    assert [r.files for r in providers.docs.requests] == [[README]]
    assert manifest[README].written

    assert (tmp_path / "output" / "backend" / "db" / "session.py").is_file()
    assert (tmp_path / "output" / "run_summary.json").is_file()
    assert _real_output_snapshot() == before


#: Routes the backend writes (valid Python, so the write guard keeps them) without the
#: contract's GET /api/todos; Integration repairs them before the docs node runs.
BROKEN_ROUTES = (
    "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
    "@router.get(\"/todos\")\ndef list_todos():\n    return {\"items\": []}\n"
)
REPAIRED_ROUTES = (
    "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
    "@router.get(\"/api/todos\")\ndef list_todos():\n    return {\"status\": \"repaired\", \"items\": []}"
)


def test_readme_is_written_after_integration_from_the_repaired_files(workspace, tmp_path):
    providers = scripted_providers(
        backend_replies={
            **BACKEND_REPLIES,
            ROUTES: {**STUB_REPLIES["backend"], "code": BROKEN_ROUTES},
        },
        integration_calls=[json.dumps({"filepath": ROUTES, "code": REPAIRED_ROUTES})],
    )
    graph = build_graph(providers, workspace)
    state = ProjectState(project_id="graph-docs", project_name="TodoGraph")
    state.requirements.raw_prompt = "Build a todo app that stores todos in SQLite."

    visited, final = run_pipeline(graph, state)

    assert visited[-3:] == ["integration", "docs", "finalize"]
    assert len(providers.integration.requests) == 1, "Integration must have repaired the routes"
    [request] = providers.docs.requests
    assert request.files == [README]
    assert REPAIRED_ROUTES in request.user_message
    assert BROKEN_ROUTES not in request.user_message
    # The run convention: uvicorn serves the frontend; never http.server or opening index.html.
    assert "uvicorn main:app" in request.user_message
    assert "http.server" in request.user_message
    assert (tmp_path / README).read_text(encoding="utf-8") == README_REPLY["code"].strip()
    # The coding agents ran before integration: none of them asked for the README.
    assert not [r for r in providers.backend.requests if README in r.files]
