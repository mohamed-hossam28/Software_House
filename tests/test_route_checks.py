import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.state import APIContract, ArchData, CodeManifest, FileManifest, FindingSeverity
from core.static_checks import run_all_checks
from core.static_checks.js_checker import check_javascript
from core.static_checks.route_checker import extract_backend_routes, check_backend_routes


def make_architecture(contracts: list[tuple[str, str]] | None) -> ArchData:
    arch = ArchData()
    arch.api_contracts = [
        APIContract(method=m, path=p, description="synthetic")
        for m, p in (contracts or [])
    ]
    return arch


BACKEND_APP = '''\
from fastapi import FastAPI, APIRouter

app = FastAPI()
router = APIRouter(prefix="/api/v1")


@app.get("/health")
def health():
    return {"ok": True}


@router.post("/todos")
def create_todo():
    return {}


@router.delete("/todos/{id}")
def delete_todo(id: str):
    return {}
'''


class TestExtractBackendRoutes:
    def test_simple_decorator_route_extracted(self):
        files = {"output/src/main.py": '@app.get("/health")\ndef health(): ...\n'}
        assert extract_backend_routes(files) == {("health",): frozenset({"GET"})}

    def test_router_prefix_prepended(self):
        routes = extract_backend_routes({"output/src/api.py": BACKEND_APP})
        assert routes[("api", "v1", "todos")] == frozenset({"POST"})
        assert routes[("api", "v1", "todos", "{}")] == frozenset({"DELETE"})

    def test_param_segment_becomes_slot(self):
        files = {"output/src/api.py": '@app.get("/todos/{item_id}")\ndef f(): ...\n'}
        assert extract_backend_routes(files) == {("todos", "{}"): frozenset({"GET"})}

    def test_api_route_methods_list(self):
        code = (
            '@app.api_route("/multi", methods=["GET", "POST"])\n'
            "def f(): ...\n"
        )
        files = {"output/src/api.py": code}
        assert extract_backend_routes(files) == {("multi",): frozenset({"GET", "POST"})}

    def test_dynamic_methods_list_skipped(self):
        code = '@app.api_route("/x", methods=SOME_LIST)\ndef f(): ...\n'
        files = {"output/src/api.py": code}
        assert extract_backend_routes(files) == {}

    def test_absent_methods_defaults_to_get(self):
        code = '@app.api_route("/x")\ndef f(): ...\n'
        files = {"output/src/api.py": code}
        assert extract_backend_routes(files) == {("x",): frozenset({"GET"})}

    def test_two_decorators_on_one_handler_merge(self):
        code = (
            '@app.get("/items")\n'
            '@app.post("/items")\n'
            "def f(): ...\n"
        )
        files = {"output/src/api.py": code}
        assert extract_backend_routes(files) == {("items",): frozenset({"GET", "POST"})}

    def test_non_literal_path_skipped(self):
        code = '@app.get(dynamic_path)\ndef f(): ...\n'
        files = {"output/src/api.py": code}
        assert extract_backend_routes(files) == {}

    def test_non_python_files_skipped(self):
        files = {"output/static/app.js": 'fetch("/api/v1/todos")\n'}
        assert extract_backend_routes(files) == {}

    def test_unparsable_file_skipped_without_crash(self):
        files = {"output/src/broken.py": "def broken( :\n"}
        assert extract_backend_routes(files) == {}

    def test_methods_merge_across_files(self):
        files = {
            "output/src/a.py": '@app.get("/todos")\ndef f(): ...\n',
            "output/src/b.py": '@app.post("/todos")\ndef f(): ...\n',
        }
        assert extract_backend_routes(files) == {("todos",): frozenset({"GET", "POST"})}


class TestContractCoverage:
    def test_backend_implementing_every_contract_is_clean(self):
        files = {"output/src/main.py": BACKEND_APP}
        contracts = [
            ("GET", "/health"),
            ("POST", "/api/v1/todos"),
            ("DELETE", "/api/v1/todos/{id}"),
        ]
        assert check_backend_routes(files, None, make_architecture(contracts)) == []

    def test_missing_contract_route_is_error(self):
        files = {"output/src/main.py": '@app.get("/health")\ndef health(): ...\n'}
        contracts = [("GET", "/health"), ("GET", "/api/v1/todos")]
        findings = check_backend_routes(files, None, make_architecture(contracts))
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "backend_routes"
        assert "/api/v1/todos" in f.message
        assert f.file == "", "no manifest → no repair target, honest fail"

    def test_method_mismatch_against_contract_is_error(self):
        files = {"output/src/main.py": '@app.post("/todos")\ndef f(): ...\n'}
        contracts = [("GET", "/todos")]
        findings = check_backend_routes(files, None, make_architecture(contracts))
        assert len(findings) == 1
        assert findings[0].severity == FindingSeverity.ERROR
        assert findings[0].check == "backend_routes"

    def test_route_findings_target_manifest_route_file(self):
        files = {"output/src/main.py": '@app.get("/health")\ndef health(): ...\n'}
        contracts = [("GET", "/health"), ("GET", "/api/v1/todos")]
        manifest = CodeManifest()
        for path in ("output/src/main.py", "output/src/api/todo.py"):
            manifest.files[path] = FileManifest(
                agent="backend", task_id=f"task-{path}", filepath=path,
                purpose="Todo CRUD endpoints", exports=[],
            )
        findings = check_backend_routes(files, manifest, make_architecture(contracts))
        assert findings
        assert all(f.file == "output/src/api/todo.py" for f in findings)

    def test_route_findings_fall_back_to_entry_file(self):
        files = {"output/src/main.py": '@app.get("/health")\ndef health(): ...\n'}
        contracts = [("GET", "/api/v1/todos")]
        manifest = CodeManifest()
        manifest.files["output/src/main.py"] = FileManifest(
            agent="backend", task_id="task-main", filepath="output/src/main.py",
            purpose="application entry point", exports=["app"],
        )
        findings = check_backend_routes(files, manifest, make_architecture(contracts))
        assert findings
        assert all(f.file == "output/src/main.py" for f in findings)

    def test_non_backend_files_are_never_route_targets(self):
        files = {"output/src/main.py": '@app.get("/health")\ndef health(): ...\n'}
        contracts = [("GET", "/api/v1/todos")]
        manifest = CodeManifest()
        manifest.files["output/static/routes.py"] = FileManifest(
            agent="frontend", task_id="task-f", filepath="output/static/routes.py",
            purpose="route helpers for the SPA", exports=[],
        )
        findings = check_backend_routes(files, manifest, make_architecture(contracts))
        assert findings
        assert all(f.file == "" for f in findings)

    def test_param_shape_matches_contract_param(self):
        files = {"output/src/main.py": '@app.delete("/todos/{item_id}")\ndef f(): ...\n'}
        contracts = [("DELETE", "/todos/{id}")]
        assert check_backend_routes(files, None, make_architecture(contracts)) == []

    def test_param_route_serves_literal_contract(self):
        # FastAPI: /users/{id} also serves GET /users/me, so it implements it
        files = {"output/src/main.py": '@app.get("/users/{id}")\ndef f(): ...\n'}
        contracts = [("GET", "/users/me")]
        assert check_backend_routes(files, None, make_architecture(contracts)) == []

    def test_no_contracts_means_no_findings(self):
        files = {"output/src/main.py": '@app.get("/health")\ndef f(): ...\n'}
        assert check_backend_routes(files, None, make_architecture([])) == []
        assert check_backend_routes(files, None, None) == []

    def test_extra_uncontracted_route_not_flagged(self):
        files = {"output/src/main.py": BACKEND_APP}
        contracts = [("GET", "/health")]  # backend implements more than contracted
        assert check_backend_routes(files, None, make_architecture(contracts)) == []


class TestFrontendTwoWayMatching:
    def test_call_to_implemented_route_without_contract_is_clean(self):
        files = {
            "output/src/main.py": '@app.get("/api/v1/todos")\ndef f(): ...\n',
            "output/static/app.js": 'fetch("/api/v1/todos").then(r => r.json());\n',
        }
        assert check_javascript(files, None, make_architecture([])) == []

    def test_call_matching_neither_contract_nor_route_is_error(self):
        files = {
            "output/src/main.py": '@app.get("/api/v1/todos")\ndef f(): ...\n',
            "output/static/app.js": 'fetch("/api/v1/ghosts").then(r => r.json());\n',
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        findings = check_javascript(files, None, arch)
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "js_urls"

    def test_call_to_implemented_route_with_wrong_contract_method_is_clean(self):
        # contract only defines POST on /todos, but the backend implements GET there
        files = {
            "output/src/main.py": '@app.get("/todos")\ndef f(): ...\n',
            "output/static/app.js": 'fetch("/todos").then(r => r.json());\n',
        }
        arch = make_architecture([("POST", "/todos")])
        assert check_javascript(files, None, arch) == []

    def test_call_to_contract_not_implemented_stays_frontend_clean(self):
        # the frontend did its part; the gap is the backend's — covered by the warn
        files = {"output/static/app.js": 'fetch("/api/v1/todos").then(r => r.json());\n'}
        arch = make_architecture([("GET", "/api/v1/todos")])
        assert check_javascript(files, None, arch) == []

    def test_url_check_skipped_when_no_contracts_and_no_routes(self):
        files = {"output/static/app.js": 'fetch("/api/v1/anything").then(r => r.json());\n'}
        assert check_javascript(files, None, make_architecture([])) == []
        assert check_javascript(files, None, None) == []


class TestRunAllChecksIntegration:
    def test_backend_gap_and_frontend_error_surface_together(self):
        files = {
            "output/src/main.py": '@app.get("/todos")\ndef f(): ...\n',
            "output/static/app.js": 'fetch("/api/v1/ghosts").then(r => r.json());\n',
        }
        # contract on /users is never implemented → backend_routes error;
        # ghosts call → js_urls error
        contracts = [("GET", "/todos"), ("GET", "/users")]
        findings = run_all_checks(files, None, architecture=make_architecture(contracts))
        errors = [f for f in findings if f.severity == FindingSeverity.ERROR]
        assert any(f.check == "backend_routes" for f in errors)
        assert any(f.check == "js_urls" for f in errors)

    def test_fully_aligned_project_is_clean(self):
        files = {
            "output/src/main.py": BACKEND_APP,
            "output/static/app.js": (
                'fetch("/health").then(r => r.json());\n'
                'axios.post("/api/v1/todos", {});\n'
                'axios.delete(`/api/v1/todos/${id}`);\n'
            ),
        }
        contracts = [
            ("GET", "/health"),
            ("POST", "/api/v1/todos"),
            ("DELETE", "/api/v1/todos/{id}"),
        ]
        findings = run_all_checks(files, None, architecture=make_architecture(contracts))
        assert findings == []
