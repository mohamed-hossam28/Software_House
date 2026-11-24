"""
Python static checks: syntax, internal import resolution, third-party imports,
and manifest written-files conformance.

Internal = a dotted module path that either exists in the written files
(module index) or is planned by the manifest, or whose root is a workspace
layer directory (e.g. 'backend' in 'output/backend/...'). Everything else is
third-party or stdlib; third-party roots are judged against the project's
requirements.txt and an undeclared one is an error.
"""
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from core.state import FindingSeverity, StaticFinding
from core.workspace import dotted_import


def check_python(
    files: dict[str, str],
    manifest: Any,
    architecture: Any = None,
    dialect: str = "",
) -> list[StaticFinding]:
    """Run all Python checks over *files* and return the findings.

    architecture and dialect are part of the uniform checker signature;
    they are unused by the Python checks.
    """
    findings: list[StaticFinding] = []
    findings.extend(_check_manifest_written(files, manifest))

    index = _build_module_index(files, manifest)
    declared = _declared_dependencies(files)

    for filepath in sorted(files):
        code = files[filepath]
        if not filepath.endswith(".py") or not code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            findings.append(StaticFinding(
                file=filepath,
                check="python_syntax",
                severity=FindingSeverity.ERROR,
                line=exc.lineno or 1,
                message=f"SyntaxError: {exc.msg}",
            ))
            continue
        findings.extend(_check_imports(filepath, tree, index))
        findings.extend(_check_third_party(filepath, tree, index, declared))
    return findings


def _manifest_files(manifest: Any) -> dict[str, Any]:
    if manifest is None:
        return {}
    return getattr(manifest, "files", None) or {}


def _module_exports(tree: ast.Module) -> set[str]:
    """Real exported names of a written module: its top-level defs, assignments, and imports."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


@dataclass
class ModuleIndex:
    """Internal module knowledge: written-and-parsed modules plus manifest-planned ones."""
    resolved: dict[str, dict[str, Any]] = field(default_factory=dict)   # module path -> {"file", "exports"}
    planned: set[str] = field(default_factory=set)                      # module paths the manifest plans
    roots: set[str] = field(default_factory=set)                        # first components of internal modules

    def is_resolved(self, candidate: str) -> bool:
        """True when *candidate* or any module beneath it is written (e.g. package via submodule)."""
        return candidate in self.resolved or any(
            m.startswith(candidate + ".") for m in self.resolved
        )

    def is_planned(self, candidate: str) -> bool:
        """True when *candidate* or any module beneath it is planned by the manifest."""
        return candidate in self.planned or any(
            m.startswith(candidate + ".") for m in self.planned
        )


def _build_module_index(files: dict[str, str], manifest: Any) -> ModuleIndex:
    index = ModuleIndex()
    for filepath in _manifest_files(manifest):
        if filepath.endswith(".py"):
            index.planned.add(dotted_import(filepath))

    for filepath in sorted(files):
        if not filepath.endswith(".py") or not files.get(filepath):
            continue
        try:
            tree = ast.parse(files[filepath])
        except SyntaxError:
            continue
        mod = dotted_import(filepath)
        if not mod:
            continue
        index.resolved[mod] = {
            "file": filepath,
            "exports": _module_exports(tree),
        }

    index.roots = {
        *(m.split(".")[0] for m in index.resolved),
        *(m.split(".")[0] for m in index.planned),
        *_workspace_layer_names(files, manifest),
    }
    return index


def _workspace_layer_names(files: dict[str, str], manifest: Any) -> set[str]:
    """Second path components of workspace files (backend, api, db, frontend, …).

    Generated code may import through a layer prefix (e.g.
    'from backend.config import …'). Those roots are internal: an unresolved
    layer import is a python_imports error, not a third-party finding.
    """
    layers: set[str] = set()
    for filepath in (*files, *_manifest_files(manifest)):
        parts = filepath.replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[0] == "output":
            layers.add(parts[1])
    return layers


def _package_parts(filepath: str, module_path: str) -> list[str]:
    if filepath.endswith("__init__.py"):
        return module_path.split(".")
    return module_path.split(".")[:-1]


def _classify_candidate(candidate: str, index: ModuleIndex) -> str | None:
    """'resolved' | 'planned-unwritten' | 'unresolved' for internal targets; None otherwise."""
    if index.is_resolved(candidate):
        return "resolved"
    if index.is_planned(candidate):
        return "planned-unwritten"
    root = candidate.split(".")[0]
    if root in sys.stdlib_module_names:
        return None
    if root in index.roots:
        return "unresolved"
    return None


def _resolve_relative(filepath: str, node: ast.ImportFrom) -> str:
    """Dotted module target of a relative import, clamped at the layer root."""
    module_path = dotted_import(filepath)
    package = _package_parts(filepath, module_path)
    drop = max(node.level - 1, 0)
    anchor = package[: max(0, len(package) - drop)]
    if node.module:
        return ".".join(anchor + node.module.split("."))
    return ".".join(anchor)


def _unresolved_finding(filepath: str, line: int, candidate: str, kind: str) -> StaticFinding:
    if kind == "planned-unwritten":
        message = f"Internal import '{candidate}' resolves to a planned module that was never written"
    else:
        message = f"Internal import '{candidate}' does not resolve to any written or planned module"
    return StaticFinding(
        file=filepath,
        check="python_imports",
        severity=FindingSeverity.ERROR,
        line=line,
        message=message,
    )


def _check_imports(
    filepath: str,
    tree: ast.Module,
    index: ModuleIndex,
) -> list[StaticFinding]:
    """Resolve internal imports and their names; relative imports are always internal."""
    findings: list[StaticFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                kind = _classify_candidate(alias.name, index)
                if kind in ("resolved", None):
                    continue
                findings.append(_unresolved_finding(filepath, node.lineno, alias.name, kind))
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            candidate = _resolve_relative(filepath, node)
        else:
            candidate = node.module or ""
            if not candidate:
                continue
            if _classify_candidate(candidate, index) is None:
                continue
        if not candidate:
            continue
        kind = _classify_candidate(candidate, index)
        if kind != "resolved":
            findings.append(_unresolved_finding(filepath, node.lineno, candidate, kind or "unresolved"))
            continue

        exports = index.resolved.get(candidate, {}).get("exports", set())
        for alias in node.names:
            name = alias.name
            if name == "*":
                continue
            if name in exports or index.is_resolved(f"{candidate}.{name}"):
                continue
            findings.append(StaticFinding(
                file=filepath,
                check="python_imports",
                severity=FindingSeverity.ERROR,
                line=node.lineno,
                message=f"Cannot import name '{name}' from '{candidate}'",
            ))
    return findings


def _declared_dependencies(files: dict[str, str]) -> set[str] | None:
    """Normalized dependency names from requirements.txt; None when absent (check skipped)."""
    req_path = next(
        (k for k in files if os.path.basename(k).lower() == "requirements.txt"),
        None,
    )
    if req_path is None:
        return None
    names: set[str] = set()
    for line in (files[req_path] or "").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        base = re.split(r"[<>=~!\[\]; ]", line)[0].strip().lower()
        if not base:
            continue
        names.add(base)
        names.add(base.replace("_", "-"))
        if base.startswith("python-"):
            names.add(base[len("python-"):])
        if "-" in base:
            names.add(base.replace("-", "_"))
    return names


def _check_third_party(
    filepath: str,
    tree: ast.Module,
    index: ModuleIndex,
    declared: set[str] | None,
) -> list[StaticFinding]:
    """Error on third-party top-level imports missing from requirements.txt. None = skip."""
    if declared is None:
        return []
    findings: list[StaticFinding] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        import_roots: list[str] = []
        if isinstance(node, ast.Import):
            import_roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            import_roots = [(node.module or "").split(".")[0]] if node.module else []
        for root in import_roots:
            if not root or root in seen:
                continue
            seen.add(root)
            if root in index.roots:
                continue
            if root in sys.stdlib_module_names:
                continue
            if root.lower() in declared or root.lower().replace("_", "-") in declared:
                continue
            findings.append(StaticFinding(
                file=filepath,
                check="third_party",
                severity=FindingSeverity.ERROR,
                line=node.lineno,
                message=f"Third-party import '{root}' is not listed in requirements.txt",
            ))
    return findings


def _check_manifest_written(files: dict[str, str], manifest: Any) -> list[StaticFinding]:
    """Error for every manifest entry marked written whose content is not among the files."""
    findings: list[StaticFinding] = []
    for filepath, fm in _manifest_files(manifest).items():
        if getattr(fm, "written", False) and filepath not in files:
            findings.append(StaticFinding(
                file=filepath,
                check="manifest_written",
                severity=FindingSeverity.ERROR,
                line=0,
                message=(
                    f"Manifest marks '{filepath}' as written, but its content is missing "
                    "from the generated files"
                ),
            ))
    return findings
