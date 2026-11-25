"""HTML asset-link check: every local href/src in written HTML resolves to a
written frontend file, relative to the served root (the frontend directory).

Seam: core.static_checks.run_all_checks(files, manifest, architecture).
"""
from core.state import ArchData, CodeManifest, FindingSeverity
from core.static_checks import run_all_checks

INDEX = "output/frontend/index.html"


def _html(head: str) -> str:
    return f"<!doctype html><html><head>{head}</head><body></body></html>"


def _link_findings(files: dict[str, str]):
    return [f for f in run_all_checks(files, CodeManifest(), ArchData()) if f.check == "html_asset_links"]


def test_absolute_static_link_without_such_file_is_an_error():
    files = {
        INDEX: _html('<link rel="stylesheet" href="/static/style.css">'),
        "output/frontend/style.css": "body { margin: 0; }",
    }
    findings = _link_findings(files)
    assert len(findings) == 1
    assert findings[0].file == INDEX
    assert findings[0].severity == FindingSeverity.ERROR
    assert "/static/style.css" in findings[0].message


def test_relative_link_to_a_written_file_is_clean():
    files = {
        INDEX: _html('<link rel="stylesheet" href="style.css"><script src="./app.js"></script>'),
        "output/frontend/style.css": "body { margin: 0; }",
        "output/frontend/app.js": "console.log(1);",
    }
    assert _link_findings(files) == []


def test_root_absolute_link_resolves_against_the_served_root():
    files = {
        INDEX: _html('<script src="/app.js"></script>'),
        "output/frontend/app.js": "console.log(1);",
    }
    assert _link_findings(files) == []


def test_link_in_a_nested_page_resolves_relative_to_that_page():
    page = "output/frontend/pages/about.html"
    files = {
        page: _html('<link rel="stylesheet" href="../style.css"><script src="app.js"></script>'),
        "output/frontend/style.css": "body { margin: 0; }",
    }
    findings = _link_findings(files)
    assert len(findings) == 1
    assert "app.js" in findings[0].message


def test_external_urls_anchors_and_data_uris_are_ignored():
    files = {
        INDEX: _html(
            '<link rel="stylesheet" href="https://cdn.example.com/x.css">'
            '<script src="//cdn.example.com/y.js"></script>'
            '<a href="#top">top</a><a href="mailto:a@b.c">mail</a>'
            '<img src="data:image/png;base64,AAAA">'
        ),
    }
    assert _link_findings(files) == []


def test_query_and_fragment_are_stripped_before_resolving():
    files = {
        INDEX: _html('<script src="app.js?v=2"></script>'),
        "output/frontend/app.js": "console.log(1);",
    }
    assert _link_findings(files) == []


def test_relative_link_escaping_the_served_root_is_an_error():
    files = {
        INDEX: _html('<script src="../backend/main.py"></script>'),
        "output/backend/main.py": "app = None",
    }
    findings = _link_findings(files)
    assert len(findings) == 1
    assert findings[0].severity == FindingSeverity.ERROR
    assert "../backend/main.py" in findings[0].message
