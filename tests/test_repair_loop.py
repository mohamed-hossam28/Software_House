"""
Repair loop tests for the Integration Agent (ticket #7).

Seam: agents.integration.IntegrationAgent.run(state) -> dict, fed a fake
state with deliberately broken code and a scripted fake LLM provider.
External behavior only: verdicts, findings, error mirroring, rounds,
repaired files, and repair token/call metrics.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from providers.llm.base import BaseLLMProvider, LLMResponse
from core.state import (
    ArchData,
    APIContract,
    CodeManifest,
    FileManifest,
    ProjectState,
    StaticFinding,
    StaticVerdict,
)
from agents.integration import IntegrationAgent


@pytest.fixture(autouse=True)
def _tmp_root(monkeypatch, tmp_path):
    """Repaired files are written through the safe-write guard — keep that off the repo."""
    monkeypatch.setattr("agents.utils._root_dir", tmp_path)


BROKEN_PY = "def broken(:\n    pass\n"
FIXED_PY = "def broken():\n    pass"   # repair writes strip trailing whitespace

# Persisting-but-parseable error: the import name never exists in models.py,
# so every "repair" that echoes the same code keeps failing python_imports
# while still passing the safe-write corruption guard.
BAD_IMPORT = (
    "output/backend/api/todo.py",
    "from db.models import TodoItem\n\nclass TodoList:\n    pass\n",
)
GOOD_IMPORT = (
    "output/backend/api/todo.py",
    "from db.models import User\n\nclass TodoList:\n    pass\n",
)


class FakeRepairProvider(BaseLLMProvider):
    """Scripted provider: returns canned responses in order, records prompts."""

    def __init__(self, responses, tokens_per_call: int = 70):
        self._responses = list(responses)
        self.tokens_per_call = tokens_per_call
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "fake-repair"

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
        self.prompts.append((system_prompt, user_message))
        if not self._responses:
            raise AssertionError("unexpected extra LLM call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(
            text=response,
            input_tokens=self.tokens_per_call // 2,
            output_tokens=self.tokens_per_call // 2,
        )


def repair_reply(filepath: str, code: str) -> str:
    """LLM reply shape the repair loop must parse: a JSON object with 'code'."""
    return json.dumps({"filepath": filepath, "code": code})


def make_state(files: dict[str, str] | None = None) -> ProjectState:
    ps = ProjectState(project_id="t", project_name="t")
    for path, code in (files or {}).items():
        if "/frontend/" in f"/{path}":
            ps.generated_code.frontend[path] = code
        elif "/database/" in f"/{path}":
            ps.generated_code.database[path] = code
        else:
            ps.generated_code.backend[path] = code
    return ps


def make_import_state() -> ProjectState:
    """Project whose todo.py imports a name models.py does not export."""
    return make_state(files={
        "output/backend/api/todo.py": BAD_IMPORT[1],
        "output/backend/db/models.py": "class User:\n    pass\n",
    })


class TestRepairLoop:
    def test_repaired_file_fixed_within_two_rounds(self):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        provider = FakeRepairProvider([repair_reply("output/backend/main.py", FIXED_PY)])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["rounds"] == 1
        assert checks["repaired_files"] == ["output/backend/main.py"]
        assert not any(
            f["severity"] == "error" and f["file"] == "output/backend/main.py"
            for f in checks["findings"]
        )
        # a repaired file leaves no survivors in the state error list
        assert "errors" not in diff
        # the repaired content replaced the broken one in the generated code
        assert ps.generated_code.backend["output/backend/main.py"] == FIXED_PY
        assert provider.calls == 1

    def test_repair_retry_succeeds_on_second_round(self):
        ps = make_import_state()
        path, _ = BAD_IMPORT
        # round 1 returns still-broken code, round 2 returns the fix
        provider = FakeRepairProvider([
            repair_reply(path, BAD_IMPORT[1]),
            repair_reply(path, GOOD_IMPORT[1]),
        ])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["rounds"] == 2
        assert checks["repaired_files"] == [path]
        assert provider.calls == 2

    def test_unrepairable_file_stops_after_two_rounds_and_fails(self):
        ps = make_import_state()
        path, _ = BAD_IMPORT
        # always returns the same broken code — never fixable
        provider = FakeRepairProvider([
            repair_reply(path, BAD_IMPORT[1]),
            repair_reply(path, BAD_IMPORT[1]),
        ])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert checks["rounds"] == 2
        assert checks["repaired_files"] == []
        assert provider.calls == 2, "repair must stop after 2 rounds per file"
        # survivors are logged with file/line/message
        assert diff["errors"], "surviving errors must flow into the error list"
        assert all(e["agent"] == "IntegrationAgent" for e in diff["errors"])
        assert any("python_imports" in e["message"] and "todo.py" in e["message"]
                   for e in diff["errors"])
        # the error finding itself stays in findings
        assert any(
            f["check"] == "python_imports" and f["file"] == path
            for f in checks["findings"]
        )

    def test_multiple_surviving_findings_are_not_duplicated(self):
        path = "output/backend/api/todo.py"
        ps = make_state(files={
            path: (
                "from db.models import TodoItem\n"
                "from db.models import BoardItem\n"
            ),
            "output/backend/db/models.py": "class User:\n    pass\n",
        })
        provider = FakeRepairProvider([
            repair_reply(path, ps.generated_code.backend[path]),
            repair_reply(path, ps.generated_code.backend[path]),
        ])

        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        survivors = [
            f for f in diff["static_checks"]["findings"]
            if f["file"] == path and f["severity"] == "error"
        ]
        assert len(survivors) == 2
        assert {f["line"] for f in survivors} == {1, 2}

    def test_final_check_refreshes_cross_file_survivors(self):
        api_path = "output/backend/api/todo.py"
        model_path = "output/backend/db/models.py"
        api_code = "from db.models import TodoItem\n"
        ps = make_state(files={
            api_path: api_code,
            model_path: BROKEN_PY,
        })
        add_manifest(ps, api_path, exports=[])
        add_manifest(ps, model_path, exports=["TodoItem"])
        provider = FakeRepairProvider([
            # The API file cannot become clean while its dependency is broken.
            repair_reply(api_path, api_code),
            repair_reply(api_path, api_code),
            # Repairing the model later also resolves the API file's survivor.
            repair_reply(model_path, "class TodoItem:\n    pass\n"),
        ])

        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.REPAIRED.value
        assert diff["static_checks"]["findings"] == []
        assert diff["static_checks"]["repaired_files"] == [model_path]
        assert diff["static_checks"]["rounds"] == 3
        assert "errors" not in diff

    def test_final_error_removes_file_from_repaired_files(self):
        api_path = "output/backend/api/todo.py"
        model_path = "output/backend/db/models.py"
        ps = make_state(files={
            api_path: BROKEN_PY,
            model_path: (
                "from db.phantom import Missing\n"
                "class Thing:\n"
                "    pass\n"
            ),
        })
        add_manifest(ps, api_path, exports=[])
        add_manifest(ps, model_path, exports=["Thing"])
        provider = FakeRepairProvider([
            # The API is clean at this point because models.py exports Thing.
            repair_reply(api_path, "from db.models import Thing\n"),
            # Fixing models.py removes Thing and makes the API fail again.
            repair_reply(model_path, "class Other:\n    pass\n"),
        ])

        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert diff["static_checks"]["repaired_files"] == [model_path]
        assert any(
            f["file"] == api_path and f["check"] == "python_imports"
            for f in diff["static_checks"]["findings"]
        )

    def test_manifest_owner_removes_stale_non_owner_bucket_copy(self):
        path = "output/shared/module.py"
        ps = make_state(files={path: BROKEN_PY})
        add_manifest(ps, path, exports=["broken"], agent="database")
        provider = FakeRepairProvider([repair_reply(path, FIXED_PY)])

        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.REPAIRED.value
        assert path not in ps.generated_code.backend
        assert ps.generated_code.database[path] == FIXED_PY

    def test_final_check_crash_is_always_reported(self, monkeypatch):
        calls = 0

        def checker(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [StaticFinding(
                    file="",
                    check="static_checks",
                    severity="error",
                    message="initial failure",
                )]
            raise RuntimeError("final validation exploded")

        monkeypatch.setattr("agents.integration.run_all_checks", checker)
        diff = IntegrationAgent(provider=FakeRepairProvider([]), retries=1).run(
            make_state()
        )

        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert any(
            "Final static checks failed: final validation exploded" in f["message"]
            for f in diff["static_checks"]["findings"]
        )

    def test_warning_only_findings_never_trigger_repair(self):
        ps = make_state(
            files={
                "output/backend/api/todo.py": "import requests\n\nr = requests\n",
                "output/requirements.txt": "fastapi==0.110.0\n",
            }
        )
        provider = FakeRepairProvider([])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.PASS.value
        assert diff["static_checks"]["rounds"] == 0
        assert diff["static_checks"]["repaired_files"] == []
        assert provider.calls == 0, "warn-only findings must not call the LLM"

    def test_repair_llm_usage_counts_in_shared_metrics(self):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        provider = FakeRepairProvider(
            [repair_reply("output/backend/main.py", FIXED_PY)], tokens_per_call=120
        )
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        assert diff["metrics"]["total_llm_calls"] == 1
        assert diff["metrics"]["total_tokens_used"] == 120

    def test_unusable_reply_still_consumes_full_round_allowance(self):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        # both replies are unusable (empty code) — the file still gets 2 rounds
        provider = FakeRepairProvider([
            repair_reply("output/backend/main.py", ""),
            repair_reply("output/backend/main.py", ""),
        ])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert checks["rounds"] == 2
        assert checks["repaired_files"] == []
        assert provider.calls == 2

    def test_corruption_refused_write_consumes_a_round(self):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        # every reply re-pastes the unparseable file — the safe-write guard
        # refuses both writes, each refusal costs one round
        provider = FakeRepairProvider([
            repair_reply("output/backend/main.py", BROKEN_PY),
            repair_reply("output/backend/main.py", BROKEN_PY),
        ])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert checks["rounds"] == 2
        # the broken original content is untouched in the bucket
        assert ps.generated_code.backend["output/backend/main.py"] == BROKEN_PY


JS_BAD = (
    "async function load() {\n"
    "  const res = await fetch('/api/v1/nope');\n"
    "  return res.json();\n"
    "}\n"
)
JS_GOOD = (
    "async function load() {\n"
    "  const res = await fetch('/api/v1/boards');\n"
    "  return res.json();\n"
    "}\n"
)


def add_manifest(
    ps: ProjectState,
    path: str,
    exports: list[str],
    written: bool = False,
    contract: dict[str, dict[str, str]] | None = None,
    agent: str = "backend",
) -> None:
    ps.code_manifest.files[path] = FileManifest(
        agent=agent,
        task_id=f"task-{path}",
        filepath=path,
        purpose="synthetic",
        exports=exports,
        contract=contract or {},
        written=written,
    )


class TestRepairPromptAndTargets:
    def test_prompt_holds_content_findings_and_api_contract(self):
        path = "output/frontend/js/app.js"
        ps = make_state(files={path: JS_BAD})
        ps.architecture = ArchData(api_contracts=[
            APIContract(method="GET", path="/api/v1/boards", description="list boards",
                        response_body={"name": "str"}),
        ])
        provider = FakeRepairProvider([repair_reply(path, JS_GOOD)])
        IntegrationAgent(provider=provider, retries=1).run(ps)

        assert provider.calls == 1
        _, user = provider.prompts[0]
        assert path in user
        assert "fetch('/api/v1/nope')" in user, "the failing file content must be in the prompt"
        assert "[js_urls] line 1" in user, "findings must appear with line numbers"
        assert "GET /api/v1/boards" in user, "HTTP findings get the API contract excerpt"
        assert "name: str" in user, "contract request/response fields are included"

    def test_prompt_holds_manifest_field_contract(self):
        path = "output/backend/api/query.py"
        ps = make_state(files={
            path: "QUERY = 'SELECT id FROM todos'\n",
            "output/database/schema.sql": "CREATE TABLE users (id INTEGER, name TEXT);\n",
        })
        ps.architecture = ArchData(tech_stack={"database": "PostgreSQL"})
        add_manifest(ps, path, exports=["QUERY"])
        add_manifest(
            ps,
            "output/database/models.py",
            exports=["User"],
            contract={"User": {"id": "int", "name": "str"}},
            agent="database",
        )
        provider = FakeRepairProvider([
            repair_reply(path, "QUERY = 'SELECT id FROM users'\n")
        ])
        IntegrationAgent(provider=provider, retries=1).run(ps)

        _, user = provider.prompts[0]
        assert "[sql_tables] line 1" in user
        assert "SCHEMA FIELD CONTRACT" in user
        assert "`User` fields: id: int, name: str" in user

    def test_missing_written_file_is_regenerated(self):
        # Deliberately use no conventional layer segment: placement must come
        # from the manifest owner, not path inference.
        path = "output/shared/models.py"
        ps = make_state(files={})
        add_manifest(
            ps, path, exports=["TodoItem"], written=True, agent="database"
        )
        provider = FakeRepairProvider([repair_reply(path, "class TodoItem:\n    name: str\n")])
        diff = IntegrationAgent(provider=provider, retries=1).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["repaired_files"] == [path]
        assert not any(f["check"] == "manifest_written" for f in checks["findings"])
        assert "errors" not in diff
        # regenerated content landed in the owning bucket
        assert ps.generated_code.database[path] == "class TodoItem:\n    name: str"
