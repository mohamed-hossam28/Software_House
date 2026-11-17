import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import pytest

from providers.llm.base import BaseLLMProvider, LLMResponse
from core.state import (
    CodeManifest,
    FileManifest,
    ProjectState,
    StaticFinding,
    StaticVerdict,
)
from agents.integration import IntegrationAgent


class FakeProvider(BaseLLMProvider):
    """Stubborn stub — every repair reply is empty code, so repair gives up.

    Used where the verdict must stay FAILED regardless of repair attempts;
    the clean/warn-only paths never reach the LLM at all.
    """

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

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
        self.calls += 1
        return LLMResponse(text='{"code": ""}', input_tokens=10, output_tokens=10)


def fenced(code: str, lang: str = "python") -> str:
    return f"```{lang}\n{code}```"


class ScriptedProvider(BaseLLMProvider):
    """Returns queued responses in order; the repair loop consumes them."""

    @property
    def provider_name(self) -> str:
        return "scripted"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

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
        text = self.responses.pop(0) if self.responses else ""
        return LLMResponse(text=text, input_tokens=17, output_tokens=23)


GOOD_TODO = '''\
from db.models import TodoItem

class TodoList:
    def all(self) -> list[TodoItem]:
        return []
'''

GOOD_MODELS = "class TodoItem:\n    name: str\n"

BROKEN_PY = "def broken(:\n    pass\n"


def _add_file(ps: ProjectState, path: str, code: str, layer: str = "backend") -> None:
    bucket = dict(getattr(ps.generated_code, layer) or {})
    bucket[path] = code
    setattr(ps.generated_code, layer, bucket)


def make_state(
    files: dict[str, str] | None = None,
    manifest_entries: dict[str, list[str]] | None = None,
    written: tuple[str, ...] = (),
) -> ProjectState:
    ps = ProjectState(project_id="t", project_name="t")
    for path, code in (files or {}).items():
        _add_file(ps, path, code)
    for path, exports in (manifest_entries or {}).items():
        ps.code_manifest.files[path] = FileManifest(
            agent="backend",
            task_id=f"task-{path}",
            filepath=path,
            purpose="synthetic",
            exports=exports,
        )
    for path in written:
        ps.code_manifest.files[path].written = True
    return ps


class TestIntegrationAgentDiff:
    def test_clean_project_passes_with_no_errors(self):
        ps = make_state(
            files={
                "output/backend/api/todo.py": GOOD_TODO,
                "output/backend/db/models.py": GOOD_MODELS,
            },
            manifest_entries={
                "output/backend/api/todo.py": ["TodoList"],
                "output/backend/db/models.py": ["TodoItem"],
            },
        )
        diff = IntegrationAgent(provider=FakeProvider()).run(ps)
        assert diff["static_checks"]["verdict"] == StaticVerdict.PASS.value
        assert diff["static_checks"]["findings"] == []
        assert "errors" not in diff
        assert diff["current_agent"] == "IntegrationAgent"

    def test_broken_python_fails_and_mirrors_errors(self):
        ps = make_state(
            files={"output/backend/main.py": BROKEN_PY},
            manifest_entries={"output/backend/main.py": ["main"]},
        )
        # The fake provider refuses every repair call, so the file survives
        # both repair rounds and its findings are mirrored into the error list.
        diff = IntegrationAgent(provider=FakeProvider(), retries=1, backoff=0.01).run(ps)
        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert any(f["check"] == "python_syntax" for f in diff["static_checks"]["findings"])
        assert diff["errors"], "error findings must flow into the state error list"
        assert all(e["agent"] == "IntegrationAgent" for e in diff["errors"])
        assert any("python_syntax" in e["message"] for e in diff["errors"])

    def test_undeclared_dependency_fails_when_unrepairable(self):
        # third_party is now an error; the fake provider refuses every repair
        # call, so the finding survives both rounds and the project fails
        ps = make_state(
            files={
                "output/backend/api/todo.py": "import requests\n\nr = requests\n",
                "output/requirements.txt": "fastapi==0.110.0\n",
            },
            manifest_entries={"output/backend/api/todo.py": ["r"]},
        )
        diff = IntegrationAgent(provider=FakeProvider(), retries=1, backoff=0.01).run(ps)
        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert "errors" in diff
        assert any(f["check"] == "third_party" for f in diff["static_checks"]["findings"])

    def test_manifest_written_missing_content_is_error(self):
        ps = make_state(
            files={"output/backend/api/todo.py": GOOD_TODO},
            manifest_entries={"output/backend/db/models.py": ["TodoItem"]},
            written=("output/backend/db/models.py",),
        )
        diff = IntegrationAgent(provider=FakeProvider()).run(ps)
        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert any(f["check"] == "manifest_written" for f in diff["static_checks"]["findings"])
        assert any("manifest_written" in e["message"] for e in diff["errors"])

    def test_checker_crash_is_logged_and_never_raises(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("checker exploded")

        monkeypatch.setattr("agents.integration.run_all_checks", boom)
        ps = make_state(files={"output/backend/main.py": "x = 1\n"})
        diff = IntegrationAgent(provider=FakeProvider()).run(ps)
        assert any("Static checks failed" in e["message"] for e in diff["errors"])
        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert diff["static_checks"]["findings"][0]["check"] == "static_checks"

    def test_unrepairable_file_gets_two_rounds_then_fails(self):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        # No provider cooperation: every repair call fails, so the file gets
        # its full allowance of repair rounds before the verdict stays failed.
        diff = IntegrationAgent(provider=FakeProvider(), retries=1, backoff=0.01).run(ps)
        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert diff["static_checks"]["rounds"] == 2
        assert diff["static_checks"]["repaired_files"] == []


class TestIntegrationNode:
    def test_node_returns_diff_for_dict_state(self, monkeypatch):
        # The provider keeps returning broken content, so both repair rounds
        # run and the verdict stays failed — with no retry sleeps.
        broken_provider = ScriptedProvider([fenced(BROKEN_PY), fenced(BROKEN_PY)])
        monkeypatch.setattr(
            "agents.base.llmFactory.get",
            staticmethod(lambda name, api_key=None: broken_provider),
        )
        from graph.nodes import integration_node

        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        diff = integration_node(ps.model_dump(mode="json"))
        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert diff["static_checks"]["rounds"] == 2
        assert diff["errors"]


class TestGraphWiring:
    def test_coding_agents_converge_on_integration_then_finalize(self):
        from graph.graph import graph

        pairs = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("backend", "integration") in pairs
        assert ("frontend", "integration") in pairs
        assert ("database", "integration") in pairs
        assert ("integration", "finalize") in pairs
        assert ("finalize", "__end__") in pairs
        assert ("backend", "finalize") not in pairs


class TestRunSummaryVerdict:
    def test_summary_includes_verdict_and_counts(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr("agents.utils._root_dir", tmp_path)
        from agents.utils import write_run_summary

        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        ps.static_checks.findings = [
            StaticFinding(
                file="output/backend/main.py", check="python_syntax",
                severity="error", line=1, message="SyntaxError: x",
            ),
            StaticFinding(
                file="output/backend/api/todo.py", check="third_party",
                severity="warn", line=1, message="warn msg",
            ),
        ]
        ps.static_checks.verdict = StaticVerdict.FAILED

        rel = write_run_summary(ps)
        assert rel == "output/run_summary.json"
        import json

        data = json.loads((tmp_path / "output" / "run_summary.json").read_text())
        assert data["static_checks"]["verdict"] == "failed"
        assert data["static_checks"]["errors"] == 1
        assert data["static_checks"]["warnings"] == 1
        assert data["static_checks"]["findings"][0]["check"] == "python_syntax"
