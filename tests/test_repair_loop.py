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
    def test_repaired_file_fixed_within_two_rounds(self, workspace):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        provider = FakeRepairProvider([repair_reply("output/backend/main.py", FIXED_PY)])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

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

    def test_repair_retry_succeeds_on_second_round(self, workspace):
        ps = make_import_state()
        path, _ = BAD_IMPORT
        # round 1 returns still-broken code, round 2 returns the fix
        provider = FakeRepairProvider([
            repair_reply(path, BAD_IMPORT[1]),
            repair_reply(path, GOOD_IMPORT[1]),
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["rounds"] == 2
        assert checks["repaired_files"] == [path]
        assert provider.calls == 2

    def test_unrepairable_file_stops_after_two_rounds_and_fails(self, workspace):
        ps = make_import_state()
        path, _ = BAD_IMPORT
        # always returns the same broken code — never fixable
        provider = FakeRepairProvider([
            repair_reply(path, BAD_IMPORT[1]),
            repair_reply(path, BAD_IMPORT[1]),
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

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

    def test_multiple_surviving_findings_are_not_duplicated(self, workspace):
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

        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        survivors = [
            f for f in diff["static_checks"]["findings"]
            if f["file"] == path and f["severity"] == "error"
        ]
        assert len(survivors) == 2
        assert {f["line"] for f in survivors} == {1, 2}

    def test_final_check_refreshes_cross_file_survivors(self, workspace):
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

        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.REPAIRED.value
        assert diff["static_checks"]["findings"] == []
        assert diff["static_checks"]["repaired_files"] == [model_path]
        assert diff["static_checks"]["rounds"] == 3
        assert "errors" not in diff

    def test_final_error_removes_file_from_repaired_files(self, workspace):
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

        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert diff["static_checks"]["repaired_files"] == [model_path]
        assert any(
            f["file"] == api_path and f["check"] == "python_imports"
            for f in diff["static_checks"]["findings"]
        )

    def test_manifest_owner_removes_stale_non_owner_bucket_copy(self, workspace):
        path = "output/shared/module.py"
        ps = make_state(files={path: BROKEN_PY})
        add_manifest(ps, path, exports=["broken"], agent="database")
        provider = FakeRepairProvider([repair_reply(path, FIXED_PY)])

        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.REPAIRED.value
        assert path not in ps.generated_code.backend
        assert ps.generated_code.database[path] == FIXED_PY

    def test_broken_html_asset_link_is_repaired_into_the_frontend_owner(self, workspace):
        index = "output/frontend/index.html"
        broken = '<html><head><link rel="stylesheet" href="/static/style.css"></head></html>'
        fixed = '<html><head><link rel="stylesheet" href="style.css"></head></html>'
        ps = make_state(files={index: broken, "output/frontend/style.css": "body { margin: 0; }"})
        add_manifest(ps, index, exports=[], agent="frontend")
        provider = FakeRepairProvider([repair_reply(index, fixed)])

        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["repaired_files"] == [index]
        assert "/static/style.css" in provider.prompts[0][1]
        assert ps.generated_code.frontend[index] == fixed

    def test_final_check_crash_is_always_reported(self, monkeypatch, workspace):
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
        diff = IntegrationAgent(provider=FakeRepairProvider([]), retries=1, workspace=workspace).run(
            make_state()
        )

        assert diff["static_checks"]["verdict"] == StaticVerdict.FAILED.value
        assert any(
            "Final static checks failed: final validation exploded" in f["message"]
            for f in diff["static_checks"]["findings"]
        )

    def test_warning_only_findings_never_trigger_repair(self, monkeypatch, workspace):
        """Findings below error severity must not consume repair rounds."""
        def checker(*args, **kwargs):
            return [StaticFinding(
                file="output/backend/api/todo.py",
                check="sql_columns",
                severity="warn",
                line=1,
                message="synthetic warning",
            )]

        monkeypatch.setattr("agents.integration.run_all_checks", checker)
        ps = make_state(files={"output/backend/api/todo.py": "x = 1\n"})
        provider = FakeRepairProvider([])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        assert diff["static_checks"]["verdict"] == StaticVerdict.PASS.value
        assert diff["static_checks"]["rounds"] == 0
        assert diff["static_checks"]["repaired_files"] == []
        assert provider.calls == 0, "warn-only findings must not call the LLM"

    def test_undeclared_dependency_triggers_repair(self, workspace):
        # third_party is now an error: the importing file is repaired
        ps = make_state(files={
            "output/backend/api/todo.py": "import requests\n\nr = requests\n",
            "output/requirements.txt": "fastapi==0.110.0\n",
        })
        provider = FakeRepairProvider([
            repair_reply("output/backend/api/todo.py", "x = 1\n")
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["repaired_files"] == ["output/backend/api/todo.py"]
        assert provider.calls == 1
        assert not any(f["check"] == "third_party" for f in checks["findings"])

    def test_missing_contract_routes_are_repaired_into_target_file(self, workspace):
        # route errors attribute to the manifest-planned entry file and the
        # repair round writes the missing route into it
        ps = make_state(files={
            "output/backend/main.py": (
                "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                '@app.get("/health")\ndef health(): ...\n'
            ),
        })
        add_manifest(ps, "output/backend/main.py", exports=["app"], written=True)
        ps.architecture = ArchData(api_contracts=[
            APIContract(method="GET", path="/health", description="health"),
            APIContract(method="POST", path="/api/v1/todos", description="create todo"),
        ])
        fixed = (
            "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
            '@app.get("/health")\ndef health(): ...\n\n'
            '@app.post("/api/v1/todos")\ndef create_todo(): ...\n'
        )
        provider = FakeRepairProvider([repair_reply("output/backend/main.py", fixed)])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["repaired_files"] == ["output/backend/main.py"]
        assert provider.calls == 1
        assert not any(f["check"] == "backend_routes" for f in checks["findings"])

    def test_route_error_without_planned_route_file_fails_without_repair(self, workspace):
        # no backend .py in the manifest → the gap cannot be attributed to any
        # file, so it fails honestly instead of guessing a repair target
        ps = make_state(files={
            "output/backend/main.py": (
                "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                '@app.get("/health")\ndef health(): ...\n'
            ),
        })
        ps.architecture = ArchData(api_contracts=[
            APIContract(method="POST", path="/api/v1/todos", description="create todo"),
        ])
        provider = FakeRepairProvider([])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert checks["rounds"] == 0
        assert provider.calls == 0, "file-less findings cannot be repaired"
        assert any(
            f["check"] == "backend_routes" and f["file"] == ""
            for f in checks["findings"]
        )
        assert "errors" in diff

    def test_repair_llm_usage_counts_in_shared_metrics(self, workspace):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        provider = FakeRepairProvider(
            [repair_reply("output/backend/main.py", FIXED_PY)], tokens_per_call=120
        )
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        assert diff["metrics"]["total_llm_calls"] == 1
        assert diff["metrics"]["total_tokens_used"] == 120

    def test_unusable_reply_still_consumes_full_round_allowance(self, workspace):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        # both replies are unusable (empty code) — the file still gets 2 rounds
        provider = FakeRepairProvider([
            repair_reply("output/backend/main.py", ""),
            repair_reply("output/backend/main.py", ""),
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert checks["rounds"] == 2
        assert checks["repaired_files"] == []
        assert provider.calls == 2

    def test_corruption_refused_write_consumes_a_round(self, workspace):
        ps = make_state(files={"output/backend/main.py": BROKEN_PY})
        # every reply re-pastes the unparseable file — the safe-write guard
        # refuses both writes, each refusal costs one round
        provider = FakeRepairProvider([
            repair_reply("output/backend/main.py", BROKEN_PY),
            repair_reply("output/backend/main.py", BROKEN_PY),
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

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
    def test_prompt_holds_content_findings_and_api_contract(self, workspace):
        path = "output/frontend/js/app.js"
        ps = make_state(files={path: JS_BAD})
        ps.architecture = ArchData(api_contracts=[
            APIContract(method="GET", path="/api/v1/boards", description="list boards",
                        response_body={"name": "str"}),
        ])
        provider = FakeRepairProvider([repair_reply(path, JS_GOOD)])
        IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        assert provider.calls == 1
        _, user = provider.prompts[0]
        assert path in user
        assert "fetch('/api/v1/nope')" in user, "the failing file content must be in the prompt"
        assert "[js_urls] line 1" in user, "findings must appear with line numbers"
        assert "GET /api/v1/boards" in user, "HTTP findings get the API contract excerpt"
        assert "name: str" in user, "contract request/response fields are included"

    def test_prompt_holds_manifest_field_contract(self, workspace):
        path = "output/backend/api/query.py"
        ps = make_state(files={
            path: "QUERY = 'SELECT id FROM todos'\n",
            "output/database/schema.sql": "CREATE TABLE users (id INTEGER, name TEXT);\n",
            "output/database/models.py": "class User:\n    id: int\n    name: str\n",
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
        IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        _, user = provider.prompts[0]
        assert "[sql_tables] line 1" in user
        assert "SCHEMA FIELD CONTRACT" in user
        assert "`User` fields: id: int, name: str" in user

    def test_missing_written_file_is_regenerated(self, workspace):
        # Deliberately use no conventional layer segment: placement must come
        # from the manifest owner, not path inference.
        path = "output/shared/models.py"
        ps = make_state(files={})
        add_manifest(
            ps, path, exports=["TodoItem"], written=True, agent="database"
        )
        # regenerated with the owner's coding prompt, so the reply is the owner's schema
        provider = FakeRepairProvider([
            json.dumps({"migration_code": "class TodoItem:\n    name: str\n"})
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["repaired_files"] == [path]
        assert not any(f["check"] == "manifest_written" for f in checks["findings"])
        assert "errors" not in diff
        # regenerated content landed in the owning bucket
        assert ps.generated_code.database[path] == "class TodoItem:\n    name: str"


class TestPersistenceRepair:
    """In-memory/bypass findings flow through the same repair loop (ticket #15)."""

    PATH = "output/backend/services/todo.py"
    IN_MEMORY = "todos = []\n\n\ndef add(item):\n    todos.append(item)\n"
    PERSISTED = (
        "from db.session import SessionLocal\n\n\n"
        "def add(item):\n    db = SessionLocal()\n    db.add(item)\n    db.commit()\n"
    )

    def _state(self) -> ProjectState:
        from core.state import build_persistence_contract

        ps = make_state(files={self.PATH: self.IN_MEMORY})
        ps.code_manifest = CodeManifest(files={self.PATH: FileManifest(
            agent="backend", task_id="be-1", filepath=self.PATH, purpose="todo service",
            exports=["add"], written=True)})
        ps.architecture = ArchData(persistence_contract=build_persistence_contract("SQLite 3"))
        return ps

    def test_in_memory_file_is_repaired_with_the_persistence_contract_in_the_prompt(self, workspace):
        from core.state import render_persistence_contract

        ps = self._state()
        provider = FakeRepairProvider([repair_reply(self.PATH, self.PERSISTED)])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        _, prompt = provider.prompts[0]
        assert "[in_memory_state] line 1" in prompt
        assert render_persistence_contract(ps.architecture.persistence_contract) in prompt
        assert diff["static_checks"]["verdict"] == StaticVerdict.REPAIRED.value
        assert diff["static_checks"]["repaired_files"] == [self.PATH]

    def test_unrepaired_in_memory_state_survives_as_error_after_two_rounds(self, workspace):
        ps = self._state()
        provider = FakeRepairProvider([repair_reply(self.PATH, self.IN_MEMORY)] * 2)
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert provider.calls == 2
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert [(f["file"], f["check"], f["severity"]) for f in checks["findings"]] == [
            (self.PATH, "in_memory_state", "error")
        ]


class TestMissingFileRecovery:
    """Planned files that were never written are regenerated with the owner's coding prompt (#39)."""

    MODELS = "output/backend/db/models.py"
    ROUTER = "output/backend/routers/todos.py"
    ROUTER_CODE = "from db.models import TodoItem\n\n\ndef new_item(name: str) -> TodoItem:\n    return TodoItem(name)\n"
    TRUNCATED = "class TodoItem(:\n    name: str\n"
    GOOD = "class TodoItem:\n    name: str\n"

    def _state(self) -> ProjectState:
        ps = make_state(files={self.ROUTER: self.ROUTER_CODE})
        add_manifest(ps, self.MODELS, exports=["TodoItem"])
        add_manifest(ps, self.ROUTER, exports=["new_item"], written=True)
        ps.code_manifest.files[self.ROUTER].imports_from = {self.MODELS: ["TodoItem"]}
        return ps

    @staticmethod
    def coding_reply(code: str) -> str:
        return json.dumps({"code": code, "classes": [], "functions": []})

    def test_refused_file_is_regenerated_and_run_ends_repaired(self, workspace):
        ps = self._state()
        provider = FakeRepairProvider([
            self.coding_reply(self.TRUNCATED),   # refused by the safe-write guard
            self.coding_reply(self.GOOD),
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.REPAIRED.value
        assert checks["rounds"] == 2
        assert self.MODELS in checks["repaired_files"]
        assert not any(f["severity"] == "error" for f in checks["findings"])
        assert provider.calls == 2
        # the file is on disk, and the importer was never edited
        on_disk = workspace.abs_path(self.MODELS).read_text(encoding="utf-8")
        assert on_disk.strip() == self.GOOD.strip()
        assert ps.generated_code.backend[self.ROUTER] == self.ROUTER_CODE
        assert ps.code_manifest.files[self.MODELS].written
        # both attempts used the owner's coding prompt; the retry carries the refusal reason
        first, second = (user for _, user in provider.prompts)
        assert first.startswith("<<<coding-request>>>\nfile: " + self.MODELS)
        assert "Integration Agent's static checks" not in first
        assert "does not parse as Python" in second
        # the refusal is cleared once written
        assert ps.write_refusals == {}

    def test_file_refused_twice_fails_naming_file_and_owner(self, workspace):
        ps = self._state()
        provider = FakeRepairProvider([
            self.coding_reply(self.TRUNCATED),
            self.coding_reply(self.TRUNCATED),
        ])
        diff = IntegrationAgent(provider=provider, retries=1, workspace=workspace).run(ps)

        checks = diff["static_checks"]
        assert checks["verdict"] == StaticVerdict.FAILED.value
        assert checks["rounds"] == 2
        assert provider.calls == 2, "the importer must not be repaired around the missing file"
        assert ps.generated_code.backend[self.ROUTER] == self.ROUTER_CODE
        assert any(
            self.MODELS in e["message"] and "backend" in e["message"]
            for e in diff["errors"]
        )
        assert self.MODELS in ps.write_refusals
