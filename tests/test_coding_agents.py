"""
Coding agents: how they stage their files and what a single-file request
carries. Ported from the demo blocks that used to sit inside the agent modules.
"""
from agents.backend import BackendAgent
from agents.utils import resolve_stages
from core.state import ArchData, CodeManifest, FileManifest, ProjectState, Task, TaskList
from providers.llm.scripted import ScriptedLLMProvider
from tests.fixtures.scripted_llm import STUB_REPLIES


def _ids(stages) -> list[set[str]]:
    return [{t.id for t in stage} for stage in stages]


def test_stages_follow_dependencies_inside_the_agent_and_ignore_other_agents():
    tasks = [
        Task(id="be-001", agent="backend", title="1", description="", depends_on=[], priority=2),
        Task(id="be-002", agent="backend", title="2", description="", depends_on=["be-001"], priority=2),
        Task(id="be-003", agent="backend", title="3", description="", depends_on=["be-001"], priority=2),
        Task(id="be-004", agent="backend", title="4", description="", depends_on=["be-002", "be-003"], priority=2),
        Task(id="be-005", agent="backend", title="5", description="", depends_on=["db-001"], priority=2),
    ]

    assert _ids(resolve_stages(tasks)) == [{"be-001", "be-005"}, {"be-002", "be-003"}, {"be-004"}]


def test_stages_by_priority_put_lower_priorities_first():
    tasks = [
        Task(id="db-001", agent="database", title="1", description="", depends_on=[], priority=1),
        Task(id="db-002", agent="database", title="2", description="", depends_on=[], priority=1),
        Task(id="db-003", agent="database", title="3", description="", depends_on=["db-001"], priority=2),
        Task(id="db-004", agent="database", title="4", description="", depends_on=["db-001"], priority=2),
        Task(id="db-005", agent="database", title="5", description="", depends_on=["db-003"], priority=3),
    ]

    assert _ids(resolve_stages(tasks, use_priority=True)) == [
        {"db-001", "db-002"}, {"db-003", "db-004"}, {"db-005"},
    ]


def test_a_single_file_request_carries_the_pending_contract_of_an_unwritten_import(workspace):
    models, crud = "output/backend/db/models.py", "output/backend/crud/todo.py"
    state = ProjectState(project_id="pending", project_name="Pending")
    state.architecture = ArchData(tech_stack={"backend": "FastAPI (Python 3.12)", "database": "PostgreSQL 16"})
    state.tasks = TaskList(tasks=[
        Task(id="be-002", agent="backend", title="crud", description="", depends_on=[], priority=2),
    ])
    state.code_manifest = CodeManifest(files={
        models: FileManifest(
            agent="database", task_id="db-001", filepath=models, purpose="ORM models", exports=["TodoItem"],
            contract={"TodoItem": {"id": "UUID", "title": "VARCHAR", "is_completed": "BOOLEAN"}},
        ),
        crud: FileManifest(
            agent="backend", task_id="be-002", filepath=crud, purpose="crud",
            exports=["TodoService", "create_todo", "list_todos"],
            imports_from={models: ["TodoItem"]},
        ),
    })
    fake = ScriptedLLMProvider("backend", files={crud: STUB_REPLIES["backend"]})

    BackendAgent(provider=fake, retries=1, workspace=workspace).run(state)

    [request] = fake.requests
    assert request.files == [crud]
    assert "TodoItem" in request.user_message
    assert "is_completed: BOOLEAN" in request.user_message
    assert state.code_manifest.files[crud].written
