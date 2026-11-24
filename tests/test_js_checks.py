import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.state import APIContract, ArchData, FindingSeverity, ProjectState
from core.static_checks import run_all_checks
from core.static_checks.js_checker import check_javascript


def make_architecture(contracts: list[tuple[str, str]] | None) -> ArchData:
    arch = ArchData()
    arch.api_contracts = [
        APIContract(method=m, path=p, description="synthetic")
        for m, p in (contracts or [])
    ]
    return arch


APP = {
    "output/frontend/app.js": "const x = 1;\n",
    "output/backend/api/todo.py": "x = 1\n",
}


class TestJsSyntax:
    def test_syntax_error_is_error_finding_with_line(self):
        files = dict(APP)
        files["output/frontend/app.js"] = "function broken( {}\nconst ok = 2;\n"
        findings = check_javascript(files, None, make_architecture([]), "")
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "js_syntax"
        assert f.file == "output/frontend/app.js"
        # esprima reports the token that breaks the parse ("const" on line 2)
        assert f.line == 2
        assert f.message

    def test_valid_module_file_is_clean(self):
        files = {"output/frontend/app.js": 'import x from "./y.js";\nexport const v = 1;\n'}
        assert check_javascript(files, None, make_architecture([]), "") == []

    def test_valid_script_file_is_clean(self):
        files = {"output/frontend/app.js": "const v = 1;\n"}
        assert check_javascript(files, None, make_architecture([]), "") == []

    def test_non_js_files_skipped(self):
        files = {"output/frontend/app.css": ".a { color: red }"}
        assert check_javascript(files, None, make_architecture([]), "") == []


class TestUrlExtraction:
    def test_fetch_matching_contract_is_clean(self):
        files = {
            "output/frontend/app.js":
                'fetch("/api/v1/todos").then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        assert check_javascript(files, None, arch, "") == []

    def test_axios_verb_matching_contract_is_clean(self):
        files = {
            "output/frontend/app.js":
                'axios.post("/api/v1/todos", { text: "x" });\n'
        }
        arch = make_architecture([("POST", "/api/v1/todos")])
        assert check_javascript(files, None, arch, "") == []

    def test_axios_config_object_matching_contract_is_clean(self):
        files = {
            "output/frontend/app.js":
                'axios({ url: "/api/v1/todos/3", method: "DELETE" });\n'
        }
        arch = make_architecture([("DELETE", "/api/v1/todos/{id}")])
        assert check_javascript(files, None, arch, "") == []

    def test_unknown_endpoint_is_error(self):
        files = {
            "output/frontend/app.js":
                'fetch("/api/v1/ghosts").then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        findings = check_javascript(files, None, arch, "")
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "js_urls"
        assert "ghosts" in f.message

    def test_method_mismatch_is_error(self):
        files = {
            "output/frontend/app.js":
                'axios.post("/api/v1/todos", {});\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        findings = check_javascript(files, None, arch, "")
        assert len(findings) == 1
        assert findings[0].severity == FindingSeverity.ERROR
        assert findings[0].check == "js_urls"


class TestNoFalsePositives:
    def test_template_param_matches_brace_contract(self):
        files = {
            "output/frontend/app.js":
                'fetch(`/api/v1/todos/${id}/items/${itemId}`).then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos/{id}/items/{item_id}")])
        assert check_javascript(files, None, arch, "") == []

    def test_trailing_slash_and_query_stripped(self):
        files = {
            "output/frontend/app.js":
                'fetch("/api/v1/todos/?full=1").then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        assert check_javascript(files, None, arch, "") == []

    def test_base_url_concatenation_resolves(self):
        files = {
            "output/frontend/app.js":
                'const BASE = "https://api.example.com";\n'
                'fetch(BASE + "/api/v1/todos").then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        assert check_javascript(files, None, arch, "") == []

    def test_env_prefix_template_resolves(self):
        files = {
            "output/frontend/app.js":
                'fetch(`${import.meta.env.VITE_API_URL}/api/v1/todos/${id}`).then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos/{id}")])
        assert check_javascript(files, None, arch, "") == []

    def test_full_url_literal_resolves_to_path(self):
        files = {
            "output/frontend/app.js":
                'fetch("https://api.example.com/api/v1/todos").then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        assert check_javascript(files, None, arch, "") == []

    def test_leading_path_param_template_not_mistaken_for_base_url(self):
        files = {
            "output/frontend/app.js":
                'fetch(`/${userId}/todos`).then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/{user_id}/todos")])
        assert check_javascript(files, None, arch, "") == []

    def test_ambiguous_leading_interpolation_still_flags_unknown_endpoint(self):
        files = {
            "output/frontend/app.js":
                'fetch(`${base}/ghosts`).then(r => r.json());\n'
        }
        arch = make_architecture([("GET", "/{id}/todos")])
        findings = check_javascript(files, None, arch, "")
        assert len(findings) == 1
        assert findings[0].severity == FindingSeverity.ERROR
        assert findings[0].check == "js_urls"

    def test_no_contracts_skips_url_check(self):
        files = {
            "output/frontend/app.js":
                'fetch("/api/v1/anything").then(r => r.json());\n'
        }
        assert check_javascript(files, None, make_architecture([]), "") == []
        assert check_javascript(files, None, None, "") == []

    def test_unresolvable_url_skipped_without_finding(self):
        files = {
            "output/frontend/app.js":
                'function go(url) { fetch(url).then(r => r.json()); }\n'
        }
        arch = make_architecture([("GET", "/api/v1/todos")])
        assert check_javascript(files, None, arch, "") == []


class TestPipelineVerdict:
    def test_js_finding_drives_failed_verdict_through_integration_agent(self, workspace):
        from agents.integration import IntegrationAgent
        from providers.llm.base import BaseLLMProvider

        class _RefusingProvider(BaseLLMProvider):
            """Never repairs — the js_urls error must survive both rounds."""

            @property
            def provider_name(self) -> str:
                return "refuse"

            def complete(self, *args, **kwargs):
                raise RuntimeError("refused")

        state = ProjectState()
        state.generated_code.frontend = {
            "output/frontend/app.js": 'fetch("/api/v1/ghosts").then(r => r.json());\n'
        }
        state.architecture = make_architecture([("GET", "/api/v1/todos")])

        diff = IntegrationAgent(provider=_RefusingProvider(), retries=1, backoff=0.01, workspace=workspace).run(state)

        assert diff["static_checks"]["verdict"] == "failed"
        assert any(
            f["check"] == "js_urls" for f in diff["static_checks"]["findings"]
        )


class TestRunAllChecksIntegration:
    def test_run_all_checks_includes_js_findings(self):
        files = {
            "output/frontend/app.js": "function broken( {}\n",
            "output/backend/api/todo.py": "x = 1\n",
        }
        findings = run_all_checks(files, None, architecture=make_architecture([]), dialect="")
        assert any(f.check == "js_syntax" for f in findings)

    def test_clean_js_and_backend_zero_findings(self):
        # since ticket #6 the two-way check also requires the backend to
        # implement the contract, so the backend file must register the route
        files = {
            "output/frontend/app.js": 'fetch("/api/v1/todos").then(r => r.json());\n',
            "output/backend/api/todo.py": '@app.get("/api/v1/todos")\ndef f(): ...\n',
        }
        findings = run_all_checks(
            files, None,
            architecture=make_architecture([("GET", "/api/v1/todos")]),
            dialect="",
        )
        assert findings == []
