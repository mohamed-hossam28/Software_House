"""Ticket #8: public finalize/report behavior and broken-seed verification."""
import json
import logging
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.integration import IntegrationAgent
from agents.utils import format_static_checks_summary
from core.state import (
    FileManifest,
    ProjectState,
    StaticChecks,
    StaticFinding,
    StaticVerdict,
)
from graph.nodes import make_finalize_node
from providers.llm.base import BaseLLMProvider, LLMResponse
from tests.test_persistence_contract import TODO_MODEL_SOURCE


class BrokenSeedRepairProvider(BaseLLMProvider):
    """One-response provider that repairs the seeded Python syntax error."""

    calls = 0

    @property
    def provider_name(self) -> str:
        return "broken-seed-repair"

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
        return LLMResponse(
            text=json.dumps({
                "filepath": "output/backend/main.py",
                "code": "def main():\n    return 'todo'\n",
            }),
            input_tokens=30,
            output_tokens=20,
        )


def test_finalize_writes_summary_and_complete_findings_report(workspace, tmp_path):
    """The pipeline's finalize boundary persists verdict, findings, and repairs."""
    state = ProjectState(project_id="e2e", project_name="TodoApp")
    state.static_checks = StaticChecks(
        verdict=StaticVerdict.REPAIRED,
        findings=[
            StaticFinding(
                file="output/backend/main.py",
                check="python_syntax",
                severity="error",
                line=4,
                message="SyntaxError: invalid syntax",
            ),
            StaticFinding(
                file="output/backend/main.py",
                check="third_party",
                severity="warn",
                line=1,
                message="Package is not declared",
            ),
        ],
        rounds=2,
        repaired_files=["output/backend/main.py"],
    )

    assert make_finalize_node(workspace)(state.model_dump(mode="json")) == {}

    summary_path = tmp_path / "output" / "run_summary.json"
    report_path = tmp_path / "output" / "static_checks_report.json"
    assert summary_path.is_file()
    assert report_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = summary["static_checks"]
    assert checks["verdict"] == "repaired"
    assert checks["findings_count"] == 2
    assert checks["errors"] == 1
    assert checks["warnings"] == 1
    assert checks["repair_rounds"] == 2
    assert checks["repaired_count"] == 1
    assert checks["repaired_files"] == ["output/backend/main.py"]
    assert checks["report_path"] == "output/static_checks_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "repaired"
    assert report["counts"] == {
        "findings": 2,
        "errors": 1,
        "warnings": 1,
        "repair_rounds": 2,
        "repaired_files": 1,
    }
    assert report["repaired_files"] == ["output/backend/main.py"]
    assert report["findings"] == checks["findings"]

    rendered = format_static_checks_summary(state)
    assert "Integration verdict: repaired" in rendered
    assert "findings: 2" in rendered
    assert "repair rounds: 2" in rendered
    assert "repaired files: 1" in rendered


def test_broken_seed_repairs_and_is_recorded_in_reports(workspace, tmp_path, caplog):
    """A deliberately broken generated file heals before final reporting."""
    path = "output/backend/main.py"
    state = ProjectState(project_id="broken-seed", project_name="TodoAppBrokenSeed")
    state.generated_code.backend[path] = "def main(:\n    pass\n"
    state.code_manifest.files[path] = FileManifest(
        agent="backend",
        task_id="backend-main",
        filepath=path,
        purpose="Todo application entrypoint",
        exports=["main"],
        written=True,
    )
    provider = BrokenSeedRepairProvider()

    with caplog.at_level(logging.INFO, logger="agents.integration"):
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(state)

    assert diff["static_checks"]["verdict"] == StaticVerdict.REPAIRED.value
    assert diff["static_checks"]["rounds"] == 1
    assert diff["static_checks"]["repaired_files"] == [path]
    assert diff["metrics"]["total_llm_calls"] == 1
    assert diff["metrics"]["total_tokens_used"] == 50
    assert provider.calls == 1
    assert "verdict=repaired" in caplog.text
    assert "rounds=1" in caplog.text
    assert "repaired=1" in caplog.text

    make_finalize_node(workspace)(state.model_dump(mode="json"))
    report = json.loads(
        (tmp_path / "output" / "static_checks_report.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (tmp_path / "output" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert report["verdict"] == "repaired"
    assert report["counts"]["repair_rounds"] == 1
    assert report["repaired_files"] == [path]
    assert summary["static_checks"]["repair_rounds"] == 1
    assert summary["static_checks"]["repaired_count"] == 1


def test_main_logging_tees_stdout_stderr_and_logger_output(tmp_path):
    """A normal CLI process preserves console output in output/logs.txt."""
    log_path = tmp_path / "logs.txt"
    ignored_second_path = tmp_path / "second.txt"
    script = (
        "import logging\n"
        "from main import configure_run_logging\n"
        f"configure_run_logging({str(log_path)!r})\n"
        f"print(configure_run_logging({str(ignored_second_path)!r}))\n"
        "print('pipeline summary line')\n"
        "print('stderr line', file=__import__('sys').stderr)\n"
        "logging.getLogger('agents.integration').info('repair round succeeded')\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        capture_output=True,
        text=True,
        check=True,
    )

    content = log_path.read_text(encoding="utf-8")
    assert "pipeline summary line" in result.stdout
    assert "stderr line" in result.stderr
    assert "pipeline summary line" in content
    assert "stderr line" in content
    assert "repair round succeeded" in content
    assert str(log_path) in result.stdout
    assert not ignored_second_path.exists()


class AnyCodingFileProvider(BaseLLMProvider):
    """Answers every backend/database prompt (single file or trivial batch) with stub code."""

    @property
    def provider_name(self) -> str:
        return "any-coding-file"

    def complete(self, system_prompt, user_message, model="", max_tokens=8192,
                 temperature=0.2, stream=False, chunk_callback=None) -> LLMResponse:
        stub = {"code": "LLM_WROTE_THIS = 1\n", "migration_code": "LLM_WROTE_THIS = 1\n",
                "classes": [], "functions": []}
        if "Return a JSON ARRAY" in user_message:
            paths = [line.split("### File:", 1)[1].strip()
                     for line in user_message.splitlines() if line.startswith("### File:")]
            text = json.dumps([{**stub, "filepath": p} for p in paths])
        else:
            text = json.dumps(stub)
        return LLMResponse(text=text, input_tokens=10, output_tokens=10)


class TodoModelProvider(AnyCodingFileProvider):
    """Database-agent LLM stub that writes a real SQLAlchemy model."""

    MODEL = TODO_MODEL_SOURCE

    def complete(self, *args, **kwargs) -> LLMResponse:
        response = super().complete(*args, **kwargs)
        data = json.loads(response.text)
        items = data if isinstance(data, list) else [data]
        for item in items:
            item["migration_code"] = self.MODEL
        return LLMResponse(text=json.dumps(data), input_tokens=10, output_tokens=10)


def _run_planner_and_coding_agents(workspace, database_tech: str) -> ProjectState:
    """Planner → database + backend agents with mocked LLMs, writing under tmp_path."""
    from agents.backend import BackendAgent
    from agents.data_base import DatabaseAgent
    from agents.planner import PlannerAgent
    from core.state import ArchData, build_persistence_contract
    from tests.planner import MockSkeletonLLM

    state = ProjectState(project_id="seam", project_name="TodoSeam")
    state.requirements.raw_prompt = "Build a todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "frontend": "HTML + Vanilla JS",
                    "database": database_tech},
        persistence_contract=build_persistence_contract(database_tech),
    )

    PlannerAgent(provider=MockSkeletonLLM()).run(state)
    DatabaseAgent(provider=TodoModelProvider(), retries=1, workspace=workspace).run(state)
    BackendAgent(provider=AnyCodingFileProvider(), retries=1, workspace=workspace).run(state)
    return state


@pytest.mark.parametrize("database_tech", ["PostgreSQL 16", "SQLite 3"])
def test_database_agent_synthesizes_the_connection_module_without_the_llm(
    workspace, tmp_path, database_tech
):
    state = _run_planner_and_coding_agents(workspace, database_tech)

    session_path = "output/backend/db/session.py"
    on_disk = (tmp_path / session_path).read_text(encoding="utf-8")
    assert "LLM_WROTE_THIS" not in on_disk
    assert 'os.environ["DATABASE_URL"]' in on_disk
    assert "def get_db" in on_disk
    assert state.code_manifest.files[session_path].written
    assert state.generated_code.database[session_path] == on_disk


def test_sqlite_stack_init_command_runs_against_the_generated_project(workspace, tmp_path):
    state = _run_planner_and_coding_agents(workspace, "SQLite 3")
    contract = state.architecture.persistence_contract
    backend_root = tmp_path / "output" / "backend"
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    env = {**os.environ, contract.db_url_env_var: url}

    result = subprocess.run(
        [sys.executable, *contract.init_command.split()[1:]],
        cwd=backend_root, env=env, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    from sqlalchemy import create_engine, inspect
    assert inspect(create_engine(url)).get_table_names() == ["todo_items"]


def test_planned_persistence_seam_reaches_the_final_manifest_and_disk(workspace, tmp_path):
    """Planner → database + backend agents: the connection module is planned and
    written, main.py imports it, and .env.example is written verbatim."""
    state = _run_planner_and_coding_agents(workspace, "PostgreSQL 16")

    manifest = state.code_manifest.files
    session = manifest["output/backend/db/session.py"]
    assert (session.agent, session.exports) == ("database", ["engine", "SessionLocal", "get_db"])
    assert session.written
    assert (tmp_path / "output/backend/db/session.py").is_file()
    assert "output/backend/db/session.py" in manifest["output/backend/main.py"].imports_from

    env_path = "output/backend/.env.example"
    on_disk = (tmp_path / env_path).read_text(encoding="utf-8")
    assert on_disk == manifest[env_path].planned_content
    assert "DATABASE_URL=postgresql://" in on_disk
    assert manifest[env_path].written


class MandateObeyingProvider(TodoModelProvider):
    """Records every prompt; the backend side writes session-backed code with no in-memory state."""

    BACKEND_CODE = (
        "from db.session import get_db\n"
        "\n"
        "\n"
        "def list_items(db):\n"
        "    return db.execute(__import__('sqlalchemy').text('SELECT 1')).all()\n"
    )

    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system_prompt, user_message, *args, **kwargs) -> LLMResponse:
        self.prompts.append((system_prompt, user_message))
        response = super().complete(system_prompt, user_message, *args, **kwargs)
        data = json.loads(response.text)
        for item in data if isinstance(data, list) else [data]:
            item["code"] = self.BACKEND_CODE
        return LLMResponse(text=json.dumps(data), input_tokens=10, output_tokens=10)


def test_mandate_obeying_backend_persists_through_the_synthesized_module(workspace, tmp_path):
    """Every DB/backend prompt carries the contract; obeying code imports the real module."""
    import ast

    from agents.backend import BackendAgent
    from agents.data_base import DatabaseAgent
    from agents.planner import PlannerAgent
    from core.state import (ArchData, backend_persistence_rules, build_persistence_contract,
                            render_persistence_contract)
    from tests.planner import MockSkeletonLLM

    contract = build_persistence_contract("SQLite 3")
    state = ProjectState(project_id="mandate", project_name="TodoMandate")
    state.requirements.raw_prompt = "Build a todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "database": "SQLite 3"},
        persistence_contract=contract,
    )
    db_llm, be_llm = MandateObeyingProvider(), MandateObeyingProvider()

    PlannerAgent(provider=MockSkeletonLLM()).run(state)
    DatabaseAgent(provider=db_llm, retries=1, workspace=workspace).run(state)
    BackendAgent(provider=be_llm, retries=1, workspace=workspace).run(state)

    rendered = render_persistence_contract(contract)
    assert db_llm.prompts and be_llm.prompts
    for system_prompt, user_message in db_llm.prompts + be_llm.prompts:
        assert rendered in system_prompt + user_message
    for system_prompt, _ in be_llm.prompts:
        assert backend_persistence_rules(contract) in system_prompt

    backend_root = tmp_path / "output" / "backend"
    for path in ("app/services/todo.py", "app/api/routes/todo.py"):
        tree = ast.parse((backend_root / path).read_text(encoding="utf-8"))
        assert not [n for n in tree.body if isinstance(n, ast.Assign)
                    and isinstance(n.value, (ast.List, ast.Dict))], path

    env = {**os.environ, "DATABASE_URL": f"sqlite:///{(tmp_path / 'app.db').as_posix()}"}
    probe = "import app.services.todo, db.session as s; print(app.services.todo.list_items(next(s.get_db())))"
    result = subprocess.run([sys.executable, "-c", probe], cwd=backend_root, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[(1,)]"


class InMemoryBackendProvider(MandateObeyingProvider):
    """Backend LLM stub that ignores the mandate and keeps data in a module-level list."""

    BACKEND_CODE = "todos = []\n\n\ndef list_items():\n    return todos\n"


def test_in_memory_backend_is_caught_by_the_integration_checks(workspace):
    from agents.backend import BackendAgent
    from agents.data_base import DatabaseAgent
    from agents.planner import PlannerAgent
    from core.state import ArchData, build_persistence_contract
    from core.static_checks import run_all_checks
    from tests.planner import MockSkeletonLLM

    state = ProjectState(project_id="inmem", project_name="TodoInMemory")
    state.requirements.raw_prompt = "Build a todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "database": "SQLite 3"},
        persistence_contract=build_persistence_contract("SQLite 3"),
    )
    PlannerAgent(provider=MockSkeletonLLM()).run(state)
    DatabaseAgent(provider=MandateObeyingProvider(), retries=1, workspace=workspace).run(state)
    BackendAgent(provider=InMemoryBackendProvider(), retries=1, workspace=workspace).run(state)

    files = {**state.generated_code.backend, **state.generated_code.database}
    findings = run_all_checks(files, state.code_manifest, state.architecture, "sqlite")

    flagged = {f.file for f in findings if f.check == "in_memory_state"}
    assert "output/backend/app/services/todo.py" in flagged
    assert "output/backend/db/session.py" not in flagged
