"""
The coding loop, through its interface: a planned Code Manifest, a profile, a
scripted fake LLM and a temporary output root in; files on disk, Written flags,
recorded code and the requests the fake received out.
"""
from pydantic import BaseModel

from agents.coding_loop import CodingAgent, CodingProfile
from core.state import ArchData, CodeManifest, FileManifest, ProjectState, Task, TaskList, build_persistence_contract
from providers.llm.scripted import ScriptedLLMProvider

SERVICE = "output/backend/app/services/todo.py"
MODELS = "output/backend/app/models/todo.py"
BARREL = "output/backend/app/__init__.py"


def _arch() -> ArchData:
    return ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "frontend": "HTML + Vanilla JS", "database": "SQLite 3"},
        persistence_contract=build_persistence_contract("SQLite 3"),
    )


def _state(*manifests: FileManifest) -> ProjectState:
    """A planned state: one task per File Manifest, in the given order."""
    state = ProjectState(project_id="coding-loop", project_name="CodingLoop")
    state.architecture = _arch()
    state.tasks = TaskList(tasks=[
        Task(id=fm.task_id, agent=fm.agent, title=fm.filepath, description=fm.purpose, priority=1)
        for fm in manifests
    ])
    state.code_manifest = CodeManifest(files={fm.filepath: fm for fm in manifests})
    return state


def _fm(path: str, agent: str, task_id: str, **kw) -> FileManifest:
    kw.setdefault("purpose", f"Purpose of {path}")
    kw.setdefault("exports", ["One", "Two", "Three"])
    return FileManifest(agent=agent, task_id=task_id, filepath=path, **kw)


class SourceReply(BaseModel):
    source: str


#: A test profile: everything the loop needs from a coding agent, as plain data.
TEST_PROFILE = CodingProfile(
    agent="backend",
    code_key="source",
    response_schema=SourceReply,
    system_prompt=lambda arch: "TEST SYSTEM PROMPT",
    sections=lambda ctx: [f"## Test Section for {ctx.filepath}"],
    tech=lambda arch: "TestTech",
)


def test_a_profile_drives_the_system_prompt_the_sections_and_the_code_key(workspace):
    state = _state(_fm(SERVICE, "backend", "be-001"))
    fake = ScriptedLLMProvider("backend", files={SERVICE: {"source": "VALUE = 1"}})

    CodingAgent(TEST_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)

    [request] = fake.requests
    assert request.files == [SERVICE]
    assert request.system_prompt == "TEST SYSTEM PROMPT"
    assert f"## Test Section for {SERVICE}" in request.user_message
    assert workspace.abs_path(SERVICE).read_text(encoding="utf-8") == "VALUE = 1"
    assert state.code_manifest.files[SERVICE].written
    assert state.generated_code.backend[SERVICE] == "VALUE = 1"


def test_the_run_returns_the_written_file_as_a_diff(workspace):
    state = _state(_fm(SERVICE, "backend", "be-001"))
    fake = ScriptedLLMProvider("backend", files={SERVICE: {"source": "VALUE = 1"}})

    diff = CodingAgent(TEST_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)

    assert diff["generated_code"]["backend"] == {SERVICE: "VALUE = 1"}
    assert diff["code_manifest"]["files"][SERVICE]["written"] is True


def test_planned_content_is_written_verbatim_and_never_reaches_the_llm(workspace):
    state = _state(
        _fm(BARREL, "backend", "be-001", planned_content="# planned barrel\n"),
        _fm(SERVICE, "backend", "be-002"),
    )
    fake = ScriptedLLMProvider("backend", files={SERVICE: {"source": "VALUE = 1"}})

    CodingAgent(TEST_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)

    assert [r.files for r in fake.requests] == [[SERVICE]]
    assert all("# planned barrel" not in r.user_message for r in fake.requests)
    assert workspace.abs_path(BARREL).read_text(encoding="utf-8") == "# planned barrel\n"
    assert state.code_manifest.files[BARREL].written
    assert state.generated_code.backend[BARREL] == "# planned barrel\n"


def test_a_single_request_carries_the_files_import_contract_and_pending_contracts(workspace):
    state = _state(
        _fm(MODELS, "database", "db-001", exports=["TodoItem"],
            contract={"TodoItem": {"id": "UUID", "is_completed": "BOOLEAN"}}),
        _fm(SERVICE, "backend", "be-001", imports_from={MODELS: ["TodoItem"]}),
    )
    fake = ScriptedLLMProvider("backend", files={SERVICE: {"source": "VALUE = 1"}})

    CodingAgent(TEST_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)

    [request] = fake.requests
    assert "from app.models.todo import TodoItem" in request.user_message
    assert "is_completed: BOOLEAN" in request.user_message


def test_only_the_profiles_own_files_are_coded(workspace):
    state = _state(_fm(MODELS, "database", "db-001"), _fm(SERVICE, "backend", "be-001"))
    fake = ScriptedLLMProvider("backend", files={SERVICE: {"source": "VALUE = 1"}})

    CodingAgent(TEST_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)

    assert [r.files for r in fake.requests] == [[SERVICE]]
    assert not state.code_manifest.files[MODELS].written


def test_a_reply_missing_the_code_key_is_an_error_and_writes_nothing(workspace):
    state = _state(_fm(SERVICE, "backend", "be-001"))
    fake = ScriptedLLMProvider("backend", files={SERVICE: {"source": "   "}})

    CodingAgent(TEST_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)

    assert not state.code_manifest.files[SERVICE].written
    assert not workspace.abs_path(SERVICE).exists()
    assert any(SERVICE in e["message"] for e in state.errors)

