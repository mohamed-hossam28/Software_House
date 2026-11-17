import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.workspace_paths import (
    _layer_root_for,
    _module_path_for,
    _normalized_src,
    _rel_import_path,
)


class TestLayerRootFor:
    def test_output_layer_root(self):
        assert _layer_root_for("output/backend/api/todo.py") == "output/backend/"

    def test_no_output_prefix_returns_none(self):
        assert _layer_root_for("main.py") is None
        assert _layer_root_for("") is None
        assert _layer_root_for("output") is None


class TestModulePathFor:
    def test_layer_root_relative_module(self):
        assert _module_path_for("output/backend/api/todo.py") == "api.todo"

    def test_init_file_collapses_to_package(self):
        assert _module_path_for("output/backend/db/__init__.py") == "db"

    def test_nested_init_collapses(self):
        assert _module_path_for("output/backend/api/v1/__init__.py") == "api.v1"

    def test_no_layer_root_uses_full_path(self):
        assert _module_path_for("main.py") == "main"

    def test_backslash_separators_without_layer_root(self):
        assert _module_path_for("output\\backend\\api\\todo.py") == "output.backend.api.todo"


class TestRelImportPath:
    def test_sibling_directory(self):
        assert _rel_import_path(
            "output/frontend/index.html", "output/frontend/js/app.js"
        ) == "./js/app"

    def test_same_directory(self):
        assert _rel_import_path(
            "output/frontend/js/app.js", "output/frontend/js/api.js"
        ) == "./api"

    def test_deep_from_sibling(self):
        assert _rel_import_path(
            "output/frontend/js/app.js", "output/frontend/js/lib/util.tsx"
        ) == "./lib/util"

    def test_extension_stripped(self):
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            assert _rel_import_path(
                "output/frontend/index.html", f"output/frontend/app{ext}"
            ) == "./app"

    def test_backslash_separators(self):
        assert _rel_import_path(
            "output\\frontend\\index.html", "output\\frontend\\js\\app.js"
        ) == "./js/app"


class TestNormalizedSrc:
    def test_prepends_output(self):
        assert _normalized_src("backend/api/todo.py") == "output/backend/api/todo.py"

    def test_strips_leading_slashes(self):
        assert _normalized_src("/backend/api/todo.py") == "output/backend/api/todo.py"

    def test_backslashes_converted(self):
        assert _normalized_src("backend\\api\\todo.py") == "output/backend/api/todo.py"

    def test_already_anchored_unchanged(self):
        assert _normalized_src("output/backend/api/todo.py") == "output/backend/api/todo.py"


def test_base_reexports_same_objects():
    """Original import sites keep working: agents.base re-exports the moved helpers."""
    from agents import base

    assert base._module_path_for is _module_path_for
    assert base._rel_import_path is _rel_import_path
    assert base._normalized_src is _normalized_src
    assert base._layer_root_for is _layer_root_for
