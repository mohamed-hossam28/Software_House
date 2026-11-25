"""
Workspace module: workspace paths, dotted imports, and writing generated files
under an injected output root.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import pytest

from core.workspace import (
    PROJECT_DIR,
    Workspace,
    dotted_import,
    module_workspace_path,
    relative_import,
    workspace_path,
)


class TestWorkspacePath:
    def test_prepends_output(self):
        assert workspace_path("backend/api/todo.py") == "output/backend/api/todo.py"

    def test_strips_leading_slashes(self):
        assert workspace_path("/backend/api/todo.py") == "output/backend/api/todo.py"

    def test_backslashes_converted(self):
        assert workspace_path("backend\\api\\todo.py") == "output/backend/api/todo.py"
        assert workspace_path("output\\backend\\api\\todo.py") == "output/backend/api/todo.py"

    def test_already_anchored_unchanged(self):
        assert workspace_path("output/backend/api/todo.py") == "output/backend/api/todo.py"

    def test_surrounding_whitespace_dropped(self):
        assert workspace_path("  backend/main.py ") == "output/backend/main.py"


class TestDottedImport:
    def test_layer_root_relative_module(self):
        assert dotted_import("output/backend/api/todo.py") == "api.todo"

    def test_init_file_collapses_to_package(self):
        assert dotted_import("output/backend/db/__init__.py") == "db"

    def test_nested_init_collapses(self):
        assert dotted_import("output/backend/api/v1/__init__.py") == "api.v1"

    def test_unanchored_path_is_normalised_first(self):
        assert dotted_import("backend/api/todo.py") == "api.todo"


class TestModuleWorkspacePath:
    def test_dotted_import_back_to_workspace_path(self):
        assert module_workspace_path("db.session", "output/backend/") == "output/backend/db/session.py"

    def test_round_trip(self):
        for path in ("output/backend/api/routes/todo.py", "output/backend/main.py"):
            assert module_workspace_path(dotted_import(path), "output/backend/") == path


class TestRelativeImport:
    def test_sibling_directory(self):
        assert relative_import(
            "output/frontend/index.html", "output/frontend/js/app.js"
        ) == "./js/app"

    def test_same_directory(self):
        assert relative_import(
            "output/frontend/js/app.js", "output/frontend/js/api.js"
        ) == "./api"

    def test_deep_from_sibling(self):
        assert relative_import(
            "output/frontend/js/app.js", "output/frontend/js/lib/util.tsx"
        ) == "./lib/util"

    def test_extension_stripped(self):
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            assert relative_import(
                "output/frontend/index.html", f"output/frontend/app{ext}"
            ) == "./app"

    def test_backslash_separators(self):
        assert relative_import(
            "output\\frontend\\index.html", "output\\frontend\\js\\app.js"
        ) == "./js/app"


class TestWorkspaceWrite:
    def test_writes_under_the_injected_root(self, tmp_path: Path):
        ws = Workspace(tmp_path)
        written = ws.write("output/backend/api/todo.py", "def list_items():\n    return []\n")

        assert written == "def list_items():\n    return []\n"
        on_disk = tmp_path / "output" / "backend" / "api" / "todo.py"
        assert on_disk.read_text(encoding="utf-8") == written

    def test_unanchored_path_lands_under_output(self, tmp_path: Path):
        Workspace(tmp_path).write("backend\\main.py", "app = None\n")
        assert (tmp_path / "output" / "backend" / "main.py").is_file()

    def test_corrupt_code_is_refused_and_reported(self, tmp_path: Path):
        refused: list[tuple[str, dict]] = []
        ws = Workspace(tmp_path)

        written = ws.write(
            "output/backend/main.py", "def main(:\n    pass\n",
            on_corrupt=lambda path, conflict: refused.append((path, conflict)),
        )

        assert written is None
        assert not (tmp_path / "output" / "backend" / "main.py").exists()
        assert [(p, c["type"]) for p, c in refused] == [("output/backend/main.py", "FILE_CORRUPTION")]

    def test_default_root_is_the_project_directory(self):
        assert Workspace().root == PROJECT_DIR
        assert Workspace().output_dir == PROJECT_DIR / "output"
        assert (PROJECT_DIR / "core" / "workspace.py").is_file()


class TestTrailingQuoteJunk:
    """A closing string quote at end of file is code, not leftover JSON."""

    @pytest.mark.parametrize("path,code", [
        ("backend/cache.py", 'def f():\n    return "down"'),
        ("backend/db/models.py", 'def r(self):\n    return f"<Todo id={self.id!r}>"'),
        ("backend/mod.py", 'def f():\n    """Doc."""\n    x = """\nmulti\n"""'),
        ("frontend/app.js", "const mode = 'x';\nconsole.log(\"done\")"),
    ])
    def test_balanced_trailing_quote_is_kept(self, tmp_path, path, code):
        written = Workspace(tmp_path).write(path, code)
        assert written == code

    @pytest.mark.parametrize("path,code,expected", [
        ("frontend/app.js", 'console.log(1);\n"', "console.log(1);\n"),
        ("frontend/app.js", 'console.log(1);"', "console.log(1);"),
        ("backend/main.py", 'x = 1\n"', "x = 1\n"),
    ])
    def test_unpaired_trailing_quote_is_stripped(self, tmp_path, path, code, expected):
        assert Workspace(tmp_path).write(path, code) == expected
