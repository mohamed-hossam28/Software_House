"""
Workspace path helpers shared by the agent and checker layers.

Import-path computations used by coding prompts (agents/base.py) and by the
static-check layer: layer-root resolution for workspace files, dotted Python
module paths, JS/TS relative import paths, and normalization of planned
import sources to the manifest's workspace-relative form.
"""
import os


def _layer_root_for(filepath: str) -> str | None:
    """Layer root dir of a workspace file, e.g. 'output/backend/'."""
    parts = (filepath or "").split("/")
    if len(parts) >= 2 and parts[0] == "output":
        return f"{parts[0]}/{parts[1]}/"
    return None


def _module_path_for(src: str) -> str:
    """
    Layer-root-relative dotted Python module for a manifest path.
    'output/backend/api/todo.py' → 'api.todo'; 'output/backend/db/__init__.py' → 'db'.
    """
    root = _layer_root_for(src)
    rel = src[len(root):] if root and src.startswith(root) else src.lstrip("/").lstrip("\\")
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")].rstrip("/")
    return rel.replace("/", ".").replace("\\", ".")


def _rel_import_path(from_file: str, to_file: str) -> str:
    """JS/TS relative import path from *from_file* to *to_file* (extension stripped)."""
    to_no_ext = to_file.replace("\\", "/")
    for ext in (".tsx", ".ts", ".jsx", ".js"):
        if to_no_ext.endswith(ext):
            to_no_ext = to_no_ext[: -len(ext)]
            break
    from_dir = os.path.dirname(from_file.replace("\\", "/"))
    rel = os.path.relpath(to_no_ext, from_dir).replace("\\", "/")
    if not rel.startswith("."):
        rel = "./" + rel
    return rel


def _normalized_src(src: str) -> str:
    """Normalize a planned import source to the manifest's workspace-relative form."""
    src = str(src).strip().lstrip("/").lstrip("\\").replace("\\", "/")
    if not src.startswith("output/"):
        src = f"output/{src}"
    return src
