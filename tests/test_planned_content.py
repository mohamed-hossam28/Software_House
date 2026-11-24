"""
Planned Content is the only way to skip the LLM (#33).

The Planner fixes the content of barrel files and of the connection module as
Planned Content on their File Manifests; every coding agent writes Planned
Content verbatim and never sends those files to the LLM.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from agents.backend import BackendAgent
from agents.data_base import DatabaseAgent
from agents.frontend import FrontendAgent
from agents.planner import PlannerAgent
from core.state import (
    ArchData,
    CodeManifest,
    FileManifest,
    ProjectState,
    Task,
    TaskList,
    build_persistence_contract,
)
from providers.llm.base import BaseLLMProvider
from tests.planner import MockSkeletonLLM


class NoLLM(BaseLLMProvider):
    """A provider that fails the test if any request reaches it."""

    def __init__(self):
        self.calls: list[str] = []

    @property
    def provider_name(self) -> str:
        return "no-llm"

    def complete(self, system_prompt, user_message, *args, **kwargs):
        self.calls.append(user_message)
        raise AssertionError("Planned Content must never reach the LLM")


def _arch(database_tech: str = "SQLite 3") -> ArchData:
    return ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "frontend": "React 18 + TypeScript",
                    "database": database_tech},
        persistence_contract=build_persistence_contract(database_tech),
    )


# ── Planner: barrels and the connection module get Planned Content ───────────

SKELETON = {"skeleton": [
    {"id": "db-001", "agent": "database", "title": "Todo model", "description": "",
     "file": "app/models/todo.py", "exports": ["TodoItem"], "imports_from": {}},
    {"id": "db-002", "agent": "database", "title": "Models package", "description": "",
     "file": "app/models/__init__.py", "exports": ["TodoItem"],
     "imports_from": {"app/models/todo.py": ["TodoItem"]}},
    {"id": "be-001", "agent": "backend", "title": "Todo service", "description": "",
     "file": "app/services/todo.py", "exports": ["list_todos"],
     "imports_from": {"app/models/__init__.py": ["TodoItem"]}},
    {"id": "fe-001", "agent": "frontend", "title": "Todo list", "description": "",
     "file": "frontend/src/components/TodoList.tsx", "exports": ["TodoList"], "imports_from": {}},
    {"id": "fe-002", "agent": "frontend", "title": "Components barrel", "description": "",
     "file": "frontend/src/components/index.ts", "exports": ["TodoList"],
     "imports_from": {"frontend/src/components/TodoList.tsx": ["TodoList"]}},
]}


def _plan() -> ProjectState:
    state = ProjectState(project_id="planned-content", project_name="Todo")
    state.requirements.raw_prompt = "Build a very simple todo app."
    state.architecture = _arch()
    mock = MockSkeletonLLM()
    mock.SKELETON = SKELETON
    PlannerAgent(provider=mock).run(state)
    return state


def test_planning_gives_barrel_files_their_re_exports_as_planned_content():
    manifest = _plan().code_manifest.files

    assert manifest["output/backend/app/models/__init__.py"].planned_content == (
        "from app.models.todo import TodoItem\n"
    )
    assert manifest["output/frontend/src/components/index.ts"].planned_content == (
        "export { TodoList } from './TodoList'\n"
    )


def test_planning_gives_the_connection_module_planned_content_from_the_persistence_contract():
    manifest = _plan().code_manifest.files

    content = manifest["output/backend/db/session.py"].planned_content
    assert content is not None
    assert 'os.environ["DATABASE_URL"]' in content
    assert "['app.models.todo']" in content, "create-all imports the planned model modules"


def test_planning_leaves_ordinary_files_to_the_llm():
    manifest = _plan().code_manifest.files

    assert manifest["output/backend/app/models/todo.py"].planned_content is None
    assert manifest["output/backend/app/services/todo.py"].planned_content is None
    assert manifest["output/frontend/src/components/TodoList.tsx"].planned_content is None


# ── Coding agents: Planned Content is written verbatim, without the LLM ──────

CASES = {
    "database": (DatabaseAgent, {
        "output/backend/db/session.py": "# planned connection module\n",
        "output/backend/app/models/__init__.py": "# planned models barrel\n",
        "output/backend/app/models/types.py": "# planned trivial file\n",
    }),
    "backend": (BackendAgent, {
        "output/backend/app/__init__.py": "# planned app barrel\n",
        "output/backend/app/config.py": "# planned trivial file\n",
    }),
    "frontend": (FrontendAgent, {
        "output/frontend/src/components/index.ts": "// planned components barrel\n",
        "output/frontend/src/types.ts": "// planned trivial file\n",
    }),
}


@pytest.mark.parametrize("agent", list(CASES))
def test_coding_agent_writes_planned_content_verbatim_without_the_llm(agent, workspace):
    agent_cls, planned = CASES[agent]
    state = ProjectState(project_id="planned-content", project_name="Todo")
    state.architecture = _arch()
    tasks, files = [], {}
    for i, (path, content) in enumerate(planned.items()):
        tid = f"{agent[:2]}-{i:03d}"
        tasks.append(Task(id=tid, agent=agent, title=path, description="", depends_on=[], priority=1))
        files[path] = FileManifest(agent=agent, task_id=tid, filepath=path, purpose="planned",
                                   exports=["X"], planned_content=content)
    state.tasks = TaskList(tasks=tasks)
    state.code_manifest = CodeManifest(files=files)
    provider = NoLLM()

    agent_cls(provider=provider, workspace=workspace).run(state)

    assert provider.calls == []
    assert not state.errors, state.errors
    for path, content in planned.items():
        on_disk = workspace.abs_path(path).read_text(encoding="utf-8")
        assert on_disk == content, path
        assert state.code_manifest.files[path].written, path
        assert getattr(state.generated_code, agent)[path] == content, path
