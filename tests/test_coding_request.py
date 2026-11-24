"""
The coding request header: the fixed first lines of every request a Coding
agent sends, naming the file (or, for a batch, each file) it asks for.
"""
import pytest

from agents.backend import BackendAgent
from agents.coding_request import parse_request_header, request_header
from agents.data_base import DatabaseAgent
from agents.frontend import FrontendAgent
from core.state import ArchData, FileManifest, ProjectState, Task
from providers.llm.scripted import ScriptedLLMProvider


def test_header_for_one_file_round_trips():
    path = "output/backend/app/api/routes/todo.py"

    assert parse_request_header(request_header([path]) + "## File to Generate\n...") == [path]


def test_header_for_a_batch_names_every_file_in_order():
    paths = ["output/frontend/src/types.ts", "output/frontend/src/constants.ts", "output/frontend/src/config.ts"]

    header = request_header(paths)

    assert header.count("\n") >= len(paths)
    assert parse_request_header(header + "Generate the following 3 files.") == paths


def test_a_request_that_does_not_start_with_the_header_names_no_file():
    body = "## Resolved Tech Stack\n" + request_header(["output/backend/main.py"])

    assert parse_request_header(body) == []


def test_an_unclosed_header_is_rejected():
    lines = request_header(["output/backend/main.py"]).strip().splitlines()
    truncated = "\n".join(lines[:-1]) + "\n## File to Generate\n"

    with pytest.raises(ValueError):
        parse_request_header(truncated)


def test_a_header_needs_at_least_one_file():
    with pytest.raises(ValueError):
        request_header([])


# ── Every request a Coding agent sends starts with the header ────────────────

PYTHON_REPLY = {"code": "VALUE = 1\n", "classes": [], "functions": []}
JS_REPLY = {"code": "export const VALUE = 1;\n", "classes": [], "functions": []}
DB_REPLY = {"migration_code": "VALUE = 1\n", "functions": []}

#: Per Coding agent: its class, its reply, one file it generates alone (three
#: exports, an ordinary name) and two trivially named files it batches.
CODING_AGENTS = {
    "database": (DatabaseAgent, DB_REPLY, "output/backend/app/models/todo.py",
                 ["output/backend/app/models/types.py", "output/backend/app/models/errors.py"]),
    "backend": (BackendAgent, PYTHON_REPLY, "output/backend/app/services/todo.py",
                ["output/backend/app/types.py", "output/backend/app/errors.py"]),
    "frontend": (FrontendAgent, JS_REPLY, "output/frontend/assets/js/todoList.js",
                 ["output/frontend/assets/js/types.js", "output/frontend/assets/js/constants.js"]),
}


def _plan(agent: str, single: str, batched: list[str]) -> ProjectState:
    state = ProjectState(project_id=f"header-{agent}", project_name="Header")
    state.architecture = ArchData(tech_stack={
        "backend": "FastAPI (Python 3.12)", "frontend": "HTML + Vanilla JS", "database": "SQLite 3",
    })
    files = {single: ["TodoA", "TodoB", "TodoC"], **{path: ["VALUE"] for path in batched}}
    for n, (path, exports) in enumerate(files.items(), start=1):
        task_id = f"{agent[:2]}-{n:03d}"
        state.tasks.tasks.append(Task(id=task_id, agent=agent, title=path, description=path, priority=1))
        state.code_manifest.files[path] = FileManifest(
            agent=agent, task_id=task_id, filepath=path, purpose=f"Purpose of {path}", exports=exports,
        )
    return state


@pytest.mark.parametrize("agent", sorted(CODING_AGENTS))
def test_every_single_and_batch_request_starts_with_the_header(agent, workspace):
    agent_cls, reply, single, batched = CODING_AGENTS[agent]
    state = _plan(agent, single, batched)
    fake = ScriptedLLMProvider(agent, files={path: reply for path in [single, *batched]})

    agent_cls(provider=fake, retries=1, workspace=workspace).run(state)

    named = [request.files for request in fake.requests]
    assert sorted(named) == sorted([[single], batched]), named
    assert all(fm.written for fm in state.code_manifest.files.values()), state.errors
