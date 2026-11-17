import json

import pytest

from agents.backend import BackendAgent
from agents.frontend import FrontendAgent
from core.state import ArchData, CodeManifest, FileManifest, ProjectState, Task, TaskList
from providers.llm.base import BaseLLMProvider, LLMResponse


class QueuedProvider(BaseLLMProvider):
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.calls: list[tuple[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "queued"

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.2,
        stream: bool = False,
        chunk_callback=None,
    ) -> LLMResponse:
        self.calls.append((system_prompt, user_message))
        return LLMResponse(
            text=self.responses.pop(0),
            input_tokens=1,
            output_tokens=1,
        )


def make_state(agent_name: str) -> tuple[ProjectState, str]:
    task_id = "be-001" if agent_name == "backend" else "fe-001"
    filepath = (
        "output/backend/services/todo.py"
        if agent_name == "backend"
        else "output/frontend/components/Todo.tsx"
    )
    state = ProjectState(project_id="schema-test", project_name="Schema Test")
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI", "frontend": "React + TypeScript"}
    )
    state.tasks = TaskList(
        tasks=[
            Task(
                id=task_id,
                agent=agent_name,
                title="Generate todo implementation",
                description="Generate one non-trivial file",
                depends_on=[],
                priority=1,
            )
        ]
    )
    state.code_manifest = CodeManifest(
        files={
            filepath: FileManifest(
                agent=agent_name,
                task_id=task_id,
                filepath=filepath,
                purpose="Todo implementation",
                # Three exports keep this at the per-file seam rather than the
                # trivial-file batch path.
                exports=["Todo", "createTodo", "listTodos"],
            )
        }
    )
    return state, filepath


@pytest.mark.parametrize(
    ("agent_cls", "agent_name", "module_name"),
    [
        (BackendAgent, "backend", "agents.backend"),
        (FrontendAgent, "frontend", "agents.frontend"),
    ],
)
def test_valid_per_file_output_passes_without_repair(
    monkeypatch, agent_cls, agent_name, module_name
):
    provider = QueuedProvider([{"code": "generated code", "classes": [], "functions": []}])
    state, filepath = make_state(agent_name)
    monkeypatch.setattr(f"{module_name}.safe_write", lambda _path, code, **_kwargs: code)

    agent_cls(provider=provider).run(state)

    assert state.generated_code.model_dump()[agent_name][filepath] == "generated code"
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("agent_cls", "agent_name", "module_name"),
    [
        (BackendAgent, "backend", "agents.backend"),
        (FrontendAgent, "frontend", "agents.frontend"),
    ],
)
def test_schema_invalid_per_file_output_uses_repair_pipeline(
    monkeypatch, agent_cls, agent_name, module_name
):
    provider = QueuedProvider(
        [
            {"classes": [], "functions": []},  # valid JSON, invalid file schema
            {"code": "repaired code", "classes": [], "functions": []},
        ]
    )
    state, filepath = make_state(agent_name)
    monkeypatch.setattr(f"{module_name}.safe_write", lambda _path, code, **_kwargs: code)

    agent_cls(provider=provider).run(state)

    assert state.generated_code.model_dump()[agent_name][filepath] == "repaired code"
    assert len(provider.calls) == 2
    repair_system, repair_prompt = provider.calls[1]
    assert "repairs JSON syntax" in repair_system
    assert "Expected JSON schema" in repair_prompt
