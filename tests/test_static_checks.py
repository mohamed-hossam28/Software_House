import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.state import (
    ProjectState,
    StaticChecks,
    StaticFinding,
    StaticVerdict,
    FindingSeverity,
    CodeManifest,
    FileManifest,
)
from core.static_checks import run_all_checks
from core.static_checks.python_checker import check_python


def make_manifest(entries: dict[str, list[str]] | None = None) -> CodeManifest:
    manifest = CodeManifest()
    for filepath, exports in (entries or {}).items():
        manifest.files[filepath] = FileManifest(
            agent="backend",
            task_id=f"task-{filepath}",
            filepath=filepath,
            purpose="synthetic",
            exports=exports,
        )
    return manifest


TODO_MODULE = '''\
from db.models import TodoItem

class TodoList:
    def all(self) -> list[TodoItem]:
        return []
'''

DB_MODELS = '''\
class TodoItem:
    name: str
'''


class TestFindingModels:
    def test_finding_fields(self):
        f = StaticFinding(file="a.py", check="x", severity="error", line=3, message="m")
        assert f.severity == FindingSeverity.ERROR
        assert f.line == 3

    def test_finding_defaults(self):
        f = StaticFinding()
        assert f.file == ""
        assert f.severity == FindingSeverity.ERROR
        assert f.line == 0
        assert f.message == ""

    def test_verdict_values(self):
        assert StaticVerdict.PASS == "pass"
        assert StaticVerdict.REPAIRED == "repaired"
        assert StaticVerdict.FAILED == "failed"
        assert StaticChecks().verdict == StaticVerdict.PASS

    def test_project_state_has_static_checks(self):
        state = ProjectState()
        assert isinstance(state.static_checks, StaticChecks)
        state.static_checks.verdict = StaticVerdict.FAILED
        assert state.static_checks.verdict == StaticVerdict.FAILED


class TestPythonSyntax:
    def test_syntax_broken_file_is_error_with_line(self):
        files = {"output/backend/main.py": "def broken(:\n    pass\n"}
        findings = check_python(files, make_manifest())
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "python_syntax"
        assert f.file == "output/backend/main.py"
        assert f.line == 1
        assert f.message

    def test_valid_python_clean(self):
        files = {"output/backend/main.py": "x = 1\n"}
        assert check_python(files, make_manifest()) == []


class TestInternalImports:
    def _project(self, todo_code: str = TODO_MODULE, exports: list[str] | None = None) -> dict:
        files = {
            "output/backend/api/todo.py": todo_code,
            "output/backend/db/models.py": DB_MODELS,
        }
        manifest = make_manifest({
            "output/backend/api/todo.py": exports or ["TodoList"],
            "output/backend/db/models.py": ["TodoItem"],
        })
        return files, manifest

    def test_clean_internal_import(self):
        files, manifest = self._project()
        assert check_python(files, manifest) == []

    def test_planned_module_never_written_is_error(self):
        files, manifest = self._project()
        del files["output/backend/db/models.py"]
        findings = check_python(files, manifest)
        assert findings and findings[0].severity == FindingSeverity.ERROR
        assert findings[0].check == "python_imports"
        assert "db.models" in findings[0].message

    def test_nonexistent_name_is_error(self):
        files, manifest = self._project()
        broken = TODO_MODULE.replace("import TodoItem", "import MissingThing")
        findings = check_python({"output/backend/api/todo.py": broken,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert findings and findings[0].severity == FindingSeverity.ERROR
        assert findings[0].check == "python_imports"
        assert "MissingThing" in findings[0].message
        assert findings[0].line == 1

    def test_phantom_module_not_in_manifest_is_error(self):
        files, manifest = self._project()
        phantom = "from api.ghost import Spook\n\n" + TODO_MODULE
        findings = check_python({"output/backend/api/todo.py": phantom,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert any(f.severity == FindingSeverity.ERROR and "api.ghost" in f.message
                   for f in findings)

    def test_relative_import_resolves(self):
        files, manifest = self._project()
        rel = "from .todo import TodoList\n\nx = TodoList\n"
        findings = check_python({"output/backend/api/__init__.py": rel,
                                 "output/backend/api/todo.py": TODO_MODULE,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert [f for f in findings if f.check == "python_imports"] == []

    def test_submodule_import_via_package_allowed(self):
        files, manifest = self._project()
        code = "from db import models\n\nm = models\n"
        findings = check_python({"output/backend/api/todo.py": code,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert [f for f in findings if f.check == "python_imports"] == []

    def test_third_party_and_stdlib_not_treated_as_internal(self):
        files, manifest = self._project()
        code = "import os\nimport json\nfrom fastapi import FastAPI\n\n" + TODO_MODULE
        findings = check_python({"output/backend/api/todo.py": code,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert [f for f in findings if f.check == "python_imports"] == []

    def test_relative_import_clamped_at_layer_root(self):
        files, manifest = self._project()
        deep = "output/backend/api/v1/sub/__init__.py"
        code = "from .....db.models import TodoItem\n"
        findings = check_python({deep: code,
                                 "output/backend/api/todo.py": TODO_MODULE,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert [f for f in findings if f.check == "python_imports"] == []

    def test_from_import_alone_at_layer_root_is_skipped(self):
        files, manifest = self._project()
        findings = check_python({"output/backend/main.py": "from . import helpers\n"}, manifest)
        assert [f for f in findings if f.check == "python_imports"] == []

    def test_plain_import_of_unwritten_internal_submodule_is_error(self):
        files, manifest = self._project()
        code = "import db.ghost\n\nx = db.ghost\n"
        findings = check_python({"output/backend/api/todo.py": code,
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert findings and findings[0].severity == FindingSeverity.ERROR
        assert findings[0].check == "python_imports"
        assert "db.ghost" in findings[0].message

    def test_planned_module_beats_stdlib_shadow(self):
        files, manifest = self._project()
        manifest.files["output/backend/types.py"] = FileManifest(
            agent="backend", task_id="task-types", filepath="output/backend/types.py",
            purpose="synthetic", exports=["Models"],
        )
        findings = check_python({"output/backend/api/todo.py": "import types\n\nx = types\n",
                                 "output/backend/db/models.py": DB_MODELS}, manifest)
        assert any(f.check == "python_imports" and "planned module" in f.message for f in findings)


class TestThirdParty:
    def _project(self, code: str, requirements: str = "fastapi==0.110.0\nuvicorn[standard]>=0.29\npython-jose==3.3.0\n") -> dict:
        files = {
            "output/backend/api/todo.py": code,
            "output/backend/db/models.py": DB_MODELS,
            "output/requirements.txt": requirements,
        }
        manifest = make_manifest({
            "output/backend/api/todo.py": ["TodoList"],
            "output/backend/db/models.py": ["TodoItem"],
        })
        return files, manifest

    def test_declared_dependency_clean(self):
        files, manifest = self._project(
            "from fastapi import FastAPI\napp = FastAPI()\n"
        )
        assert check_python(files, manifest) == []

    def test_undeclared_dependency_is_error(self):
        files, manifest = self._project(
            "import requests\n\nr = requests\n"
        )
        findings = check_python(files, manifest)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "third_party"
        assert "requests" in f.message

    def test_layer_prefixed_import_is_internal_not_third_party(self):
        files, manifest = self._project(
            "from backend.config import APP_NAME\n\nname = APP_NAME\n"
        )
        findings = check_python(files, manifest)
        assert findings
        assert all(f.check == "python_imports" for f in findings)
        assert not any(f.check == "third_party" for f in findings)
        assert all(
            "does not resolve" in f.message or "never written" in f.message
            for f in findings
        )

    def test_stdlib_never_errors(self):
        files, manifest = self._project("import hashlib\nimport logging\n\nx = hashlib\n")
        assert check_python(files, manifest) == []

    def test_no_requirements_file_skips_check(self):
        files, manifest = self._project("import requests\n\nr = requests\n", requirements="")
        del files["output/requirements.txt"]
        assert check_python(files, manifest) == []

    def test_dash_named_package_matches_underscore_import(self):
        files, manifest = self._project(
            "from jose import jwt\n\nt = jwt\n", requirements="python-jose==3.3.0\n"
        )
        assert check_python(files, manifest) == []


class TestManifestWritten:
    def test_written_entry_without_content_is_error(self):
        manifest = make_manifest({"output/backend/main.py": ["main"]})
        manifest.files["output/backend/main.py"].written = True
        files = {"output/backend/api/todo.py": "x = 1\n"}
        findings = check_python(files, manifest)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "manifest_written"
        assert f.file == "output/backend/main.py"

    def test_written_entry_with_content_clean(self):
        manifest = make_manifest({"output/backend/main.py": ["main"]})
        manifest.files["output/backend/main.py"].written = True
        files = {"output/backend/main.py": "def main():\n    pass\n"}
        assert check_python(files, manifest) == []


class TestRunAllChecks:
    def _project(self) -> dict:
        files = {
            "output/backend/api/todo.py": TODO_MODULE,
            "output/backend/db/models.py": DB_MODELS,
            "output/requirements.txt": "fastapi==0.110.0\n",
        }
        manifest = make_manifest({
            "output/backend/api/todo.py": ["TodoList"],
            "output/backend/db/models.py": ["TodoItem"],
        })
        return files, manifest

    def test_clean_project_zero_findings(self):
        files, manifest = self._project()
        assert run_all_checks(files, manifest, architecture=None, dialect="sqlite") == []

    def test_broken_project_yields_sorted_errors(self):
        files, manifest = self._project()
        files["output/backend/api/todo.py"] = "def broken(:\n"
        findings = run_all_checks(files, manifest)
        assert findings
        assert all(f.severity in (FindingSeverity.ERROR, FindingSeverity.WARN) for f in findings)

    def test_non_python_files_skipped_by_python_checker(self):
        files, manifest = self._project()
        files["output/frontend/global.css"] = ".a { color: red }"
        files["output/frontend/app.js"] = "function broken( {}"
        # the Python checker itself ignores non-.py files…
        assert check_python(files, manifest) == []
        # …and run_all_checks routes the broken JS file to the JS checker
        findings = run_all_checks(files, manifest)
        assert all(f.check == "js_syntax" for f in findings)
