"""
Coding-loop replies are validated against the profile's response schema: a
valid reply is written as is, an invalid one goes through the repair pipeline,
and one that cannot be repaired is an error that writes nothing.
"""
import pytest

from agents.backend import BACKEND_PROFILE
from agents.coding_loop import CodingAgent
from agents.data_base import DATABASE_PROFILE
from agents.frontend import FRONTEND_PROFILE
from core.state import ArchData, CodeManifest, FileManifest, ProjectState, Task, TaskList, build_persistence_contract
from providers.llm.scripted import ScriptedLLMProvider

#: Per profile: a file it generates in a single call and a valid reply for it.
CASES = {
    "backend": (BACKEND_PROFILE, "output/backend/app/services/todo.py",
                {"code": "generated_code = 1", "classes": [], "functions": []}),
    "frontend": (FRONTEND_PROFILE, "output/frontend/src/components/Todo.tsx",
                 {"code": "export const generated = 1;", "classes": [], "functions": []}),
    "database": (DATABASE_PROFILE, "output/backend/app/models/todo.py",
                 {"migration_code": "generated_code = 1", "functions": []}),
}
#: Valid JSON, but missing every profile's code key.
INVALID_REPLY = {"classes": [], "functions": []}


def _state(agent: str, filepath: str) -> ProjectState:
    state = ProjectState(project_id="schema-test", project_name="Schema Test")
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI", "frontend": "React + TypeScript", "database": "SQLite 3"},
        persistence_contract=build_persistence_contract("SQLite 3"),
    )
    state.tasks = TaskList(tasks=[
        Task(id="t-001", agent=agent, title="Generate todo implementation", description="", priority=1),
    ])
    state.code_manifest = CodeManifest(files={
        filepath: FileManifest(agent=agent, task_id="t-001", filepath=filepath, purpose="Todo implementation",
                               exports=["Todo", "createTodo", "listTodos"]),
    })
    return state


def _code(profile, reply: dict) -> str:
    return reply[profile.code_key]


@pytest.mark.parametrize("agent", sorted(CASES))
def test_a_valid_reply_is_written_without_repair(agent, workspace):
    profile, filepath, reply = CASES[agent]
    state = _state(agent, filepath)
    fake = ScriptedLLMProvider(agent, files={filepath: reply})

    CodingAgent(profile, provider=fake, retries=1, workspace=workspace).run(state)

    assert [r.files for r in fake.requests] == [[filepath]]
    assert workspace.abs_path(filepath).read_text(encoding="utf-8") == _code(profile, reply)
    assert getattr(state.generated_code, agent)[filepath] == _code(profile, reply)
    assert state.code_manifest.files[filepath].written


@pytest.mark.parametrize("agent", sorted(CASES))
def test_a_schema_invalid_reply_goes_through_the_repair_pipeline(agent, workspace):
    profile, filepath, reply = CASES[agent]
    state = _state(agent, filepath)
    fake = ScriptedLLMProvider(agent, files={filepath: INVALID_REPLY}, calls=[reply])

    CodingAgent(profile, provider=fake, retries=1, workspace=workspace).run(state)

    file_request, repair_request = fake.requests
    assert file_request.files == [filepath]
    assert repair_request.files == []
    assert "repairs JSON syntax" in repair_request.system_prompt
    assert "Expected JSON schema" in repair_request.user_message
    assert workspace.abs_path(filepath).read_text(encoding="utf-8") == _code(profile, reply)
    assert state.code_manifest.files[filepath].written


@pytest.mark.parametrize("agent", sorted(CASES))
def test_an_unrepairable_reply_is_an_error_and_writes_nothing(agent, workspace):
    profile, filepath, _ = CASES[agent]
    state = _state(agent, filepath)
    fake = ScriptedLLMProvider(agent, files={filepath: INVALID_REPLY}, calls=[INVALID_REPLY])

    CodingAgent(profile, provider=fake, retries=1, workspace=workspace).run(state)

    assert not workspace.abs_path(filepath).exists()
    assert not state.code_manifest.files[filepath].written
    assert filepath not in getattr(state.generated_code, agent)
    assert any(filepath in e["message"] for e in state.errors), state.errors
