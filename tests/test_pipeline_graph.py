"""
Pipeline graph builder: the real pipeline graph, run end to end with stub
providers and a temporary workspace. The main end-to-end test seam.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import ProjectState
from core.workspace import PROJECT_DIR
from graph.graph import AgentProviders, build_graph
from providers.llm.base import BaseLLMProvider, LLMResponse
from tests.planner import MockSkeletonLLM
from tests.test_end_to_end_verification import AnyCodingFileProvider


class CannedProvider(BaseLLMProvider):
    """Answers every call with the same JSON reply."""

    def __init__(self, reply: dict) -> None:
        self.text = json.dumps(reply)
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "canned"

    def complete(self, system_prompt, user_message, model="", max_tokens=8192,
                 temperature=0.2, stream=False, chunk_callback=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text, input_tokens=10, output_tokens=10)


#: Requirements stub: a complete SRS that asks no clarifying questions.
COMPLETE_SRS = {
    "functional_requirements": ["User can create, list and delete todo items"],
    "non_functional_requirements": ["Responses under 200ms"],
    "constraints": ["Store todos in SQLite"],
    "assumptions": ["Single user"],
    "out_of_scope": ["Mobile app"],
    "clarifying_questions": [],
}

#: Architecture stub: one reply that satisfies every architecture sub-call.
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


def stub_providers() -> AgentProviders:
    coding = AnyCodingFileProvider()
    return AgentProviders(
        requirements=CannedProvider(COMPLETE_SRS),
        architecture=CannedProvider(ARCHITECTURE),
        planner=MockSkeletonLLM(),
        database=coding,
        backend=coding,
        frontend=coding,
        integration=CannedProvider({"code": ""}),
    )


def _real_output_snapshot() -> dict[str, float]:
    out = PROJECT_DIR / "output"
    if not out.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in out.rglob("*") if p.is_file()}


def run_pipeline(graph, state: ProjectState) -> ProjectState:
    config = {"configurable": {"thread_id": state.project_id}}
    visited = [node for event in graph.stream(state, config=config) for node in event]
    final = graph.get_state(config).values
    final = final if isinstance(final, ProjectState) else ProjectState(**final)
    return visited, final


def test_graph_runs_end_to_end_inside_the_temporary_workspace(workspace, tmp_path):
    before = _real_output_snapshot()
    graph = build_graph(stub_providers(), workspace)
    state = ProjectState(project_id="graph-e2e", project_name="TodoGraph")
    state.requirements.raw_prompt = "Build a todo app that stores todos in SQLite."

    visited, final = run_pipeline(graph, state)

    assert "human" not in visited
    assert visited[-1] == "finalize"
    assert {"requirements", "architecture", "planner", "database", "backend",
            "frontend", "integration"} <= set(visited)

    written = [p for p, fm in final.code_manifest.files.items() if fm.written]
    assert written
    for path in written:
        assert (tmp_path / path).is_file(), path
        assert path == "output/README.md" or path.startswith(("output/backend/", "output/frontend/")), path
    assert (tmp_path / "output" / "backend" / "db" / "session.py").is_file()
    assert (tmp_path / "output" / "frontend" / "index.html").is_file()
    assert (tmp_path / "output" / "run_summary.json").is_file()
    assert _real_output_snapshot() == before
