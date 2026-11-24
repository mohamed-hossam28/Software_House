"""
Pipeline graph builder: the real pipeline graph, run end to end with a scripted
fake LLM for every agent and a temporary workspace. The main end-to-end test seam.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import ProjectState
from core.workspace import PROJECT_DIR, layer_root
from graph.graph import AgentProviders, build_graph
from providers.llm.scripted import ScriptedLLMProvider
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

#: Backend replies that pass the Integration checks, so Integration needs no repair call.
BACKEND_REPLIES = {
    ROUTES: {**STUB_REPLIES["backend"], "code": (
        "from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n"
        "@router.get(\"/api/todos\")\ndef list_todos():\n    return {\"items\": []}\n"
    )},
    REQUIREMENTS_TXT: {**STUB_REPLIES["backend"], "code": "fastapi\nsqlalchemy\nuvicorn\n"},
}


def scripted_providers() -> AgentProviders:
    backend = scripted_coding_agent("backend", TODO_LLM_FILES["backend"])
    backend.files.update(BACKEND_REPLIES)
    return AgentProviders(
        requirements=ScriptedLLMProvider("requirements", calls=[COMPLETE_SRS]),
        architecture=ScriptedLLMProvider("architecture", calls=[ARCHITECTURE] * 3),
        planner=scripted_planner(),
        database=scripted_coding_agent("database", TODO_LLM_FILES["database"]),
        backend=backend,
        frontend=scripted_coding_agent("frontend", TODO_LLM_FILES["frontend"]),
        integration=ScriptedLLMProvider("integration"),
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
    assert visited[-1] == "finalize"
    assert {"requirements", "architecture", "planner", "database", "backend",
            "frontend", "integration"} <= set(visited)
    assert not [e for e in final.errors if "scripted fake" in e["message"]], final.errors

    manifest = final.code_manifest.files
    assert manifest
    unwritten = [path for path, fm in manifest.items() if not fm.written]
    assert not unwritten, unwritten
    for path, fm in manifest.items():
        assert (tmp_path / path).is_file(), path
        if path != "output/README.md":  # the one project-level file
            assert path.startswith(layer_root(fm.agent)), (fm.agent, path)

    # Every Coding agent was asked for exactly its scripted files, nothing else.
    for agent in ("database", "backend", "frontend"):
        asked = {p for request in getattr(providers, agent).requests for p in request.files}
        assert asked == set(TODO_LLM_FILES[agent]), agent
    assert providers.integration.requests == []

    assert (tmp_path / "output" / "backend" / "db" / "session.py").is_file()
    assert (tmp_path / "output" / "run_summary.json").is_file()
    assert _real_output_snapshot() == before
