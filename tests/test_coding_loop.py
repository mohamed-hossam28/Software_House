"""
The coding loop, through its interface: a planned Code Manifest, a profile, a
scripted fake LLM and a temporary output root in; files on disk, Written flags,
recorded code and the requests the fake received out.
"""
from pydantic import BaseModel

from agents.backend import BACKEND_PROFILE
from agents.coding_loop import CodingAgent, CodingProfile
from core.state import ArchData, CodeManifest, FileManifest, ProjectState, Task, TaskList, build_persistence_contract
from providers.llm.scripted import ScriptedLLMProvider
from tests.fixtures.scripted_llm import STUB_REPLIES

SERVICE = "output/backend/app/services/todo.py"
MODELS = "output/backend/app/models/todo.py"
BARREL = "output/backend/app/__init__.py"
ROUTER = "output/backend/app/api/routes/todo.py"
SCHEMAS = "output/backend/app/schemas/todo.py"
TYPES = "output/backend/app/core/types.py"
CONSTANTS = "output/backend/app/core/constants.py"
CONFIG = "output/backend/app/core/config.py"
INVALID_REPLY = {"classes": [], "functions": []}  # valid JSON, no code key


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


# ── Batching: by name only, with full contracts and per-item validation ──────

def _models_and_schemas() -> list[FileManifest]:
    """An unwritten ORM model and schema module, each with a planned field contract."""
    return [
        _fm(MODELS, "database", "db-001", exports=["TodoItem", "Base"],
            contract={"TodoItem": {"id": "UUID", "is_completed": "BOOLEAN"}}),
        _fm(SCHEMAS, "database", "db-002", exports=["TodoCreate"],
            contract={"TodoCreate": {"title": "str", "due_on": "date"}}),
    ]


def _backend(fake, workspace, state) -> None:
    CodingAgent(BACKEND_PROFILE, provider=fake, retries=1, workspace=workspace).run(state)


def test_a_router_with_few_exports_goes_through_a_single_call_with_its_full_contracts(workspace):
    state = _state(*_models_and_schemas(), _fm(ROUTER, "backend", "be-001", exports=["router"],
                                                imports_from={MODELS: ["TodoItem"], SCHEMAS: ["TodoCreate"]}))
    fake = ScriptedLLMProvider("backend", files={ROUTER: STUB_REPLIES["backend"]})

    _backend(fake, workspace, state)

    [request] = fake.requests
    assert request.files == [ROUTER]
    assert "from app.models.todo import TodoItem" in request.user_message
    assert "from app.schemas.todo import TodoCreate" in request.user_message
    assert "is_completed: BOOLEAN" in request.user_message
    assert "due_on: date" in request.user_message
    assert state.code_manifest.files[ROUTER].written


def test_trivially_named_files_are_still_batched(workspace):
    paths = [TYPES, CONSTANTS, CONFIG]
    state = _state(*(_fm(p, "backend", f"be-{n}", exports=["VALUE"]) for n, p in enumerate(paths)))
    fake = ScriptedLLMProvider("backend", files={p: STUB_REPLIES["backend"] for p in paths})

    _backend(fake, workspace, state)

    assert [sorted(r.files) for r in fake.requests] == [sorted(paths)]
    for path in paths:
        assert state.code_manifest.files[path].written, state.errors
        assert workspace.abs_path(path).read_text(encoding="utf-8") == "LLM_WROTE_THIS = 1"


def test_every_batch_request_carries_each_batched_files_import_and_pending_contracts(workspace):
    state = _state(
        *_models_and_schemas(),
        _fm(TYPES, "backend", "be-001", imports_from={MODELS: ["TodoItem"]}),
        _fm(CONSTANTS, "backend", "be-002", imports_from={SCHEMAS: ["TodoCreate"]}),
    )
    fake = ScriptedLLMProvider("backend", files={p: STUB_REPLIES["backend"] for p in [TYPES, CONSTANTS]})

    _backend(fake, workspace, state)

    [request] = fake.requests
    assert sorted(request.files) == sorted([TYPES, CONSTANTS])
    assert "from app.models.todo import TodoItem" in request.user_message
    assert "from app.schemas.todo import TodoCreate" in request.user_message
    assert "is_completed: BOOLEAN" in request.user_message
    assert "due_on: date" in request.user_message


def test_a_batch_item_failing_validation_goes_through_the_repair_pipeline(workspace):
    state = _state(_fm(TYPES, "backend", "be-001"), _fm(CONSTANTS, "backend", "be-002"))
    repaired = {"code": "REPAIRED = 1", "classes": [], "functions": []}
    fake = ScriptedLLMProvider("backend", files={TYPES: STUB_REPLIES["backend"], CONSTANTS: INVALID_REPLY},
                               calls=[repaired])

    _backend(fake, workspace, state)

    batch, repair = fake.requests
    assert sorted(batch.files) == sorted([TYPES, CONSTANTS])
    assert repair.files == [] and "repairs JSON syntax" in repair.system_prompt
    assert workspace.abs_path(CONSTANTS).read_text(encoding="utf-8") == "REPAIRED = 1"
    assert state.code_manifest.files[TYPES].written and state.code_manifest.files[CONSTANTS].written


def test_an_unrepairable_batch_item_is_an_error_like_a_single_reply(workspace):
    state = _state(_fm(TYPES, "backend", "be-001"), _fm(CONSTANTS, "backend", "be-002"))
    fake = ScriptedLLMProvider("backend", files={TYPES: STUB_REPLIES["backend"], CONSTANTS: INVALID_REPLY},
                               calls=[INVALID_REPLY])

    _backend(fake, workspace, state)

    assert state.code_manifest.files[TYPES].written
    assert not state.code_manifest.files[CONSTANTS].written
    assert not workspace.abs_path(CONSTANTS).exists()
    assert any(CONSTANTS in e["message"] and "validation" in e["message"] for e in state.errors), state.errors


def test_a_lone_trivially_named_file_is_written_from_one_single_call(workspace):
    state = _state(_fm(TYPES, "backend", "be-001"), _fm(SERVICE, "backend", "be-002"))
    fake = ScriptedLLMProvider("backend", files={p: STUB_REPLIES["backend"] for p in [TYPES, SERVICE]})

    _backend(fake, workspace, state)

    assert sorted(r.files for r in fake.requests) == sorted([[TYPES], [SERVICE]])
    assert state.code_manifest.files[TYPES].written

