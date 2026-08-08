"""
agents/integration.py
─────────────────────
Integration Agent — performs static integration validation and mechanical auto-fixing.

Inputs:
  - ProjectState.code_manifest  (CodeManifest)
  - ProjectState.architecture   (ArchData)
  - ProjectState.generated_code  (CodeMap)

Never execute code or tests.
Integration may only perform mechanical fixes (AUTO_FIX) or report behavioral/logic decisions (HUMAN).
"""

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any, Optional, Union

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF
except ImportError:
    from base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF

from providers.llm.base import BaseLLMProvider

from core.state import (
    ProjectState,
    FileManifest,
    CodeManifest,
    IntegrationStatus,
    APIContract,
    Argument,
    ClassName,
    FunctionName,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 3
OUTPUT_DIR = "output"

#: Conflict types that integration must NEVER guess a fix for — they are routed
#: straight to the owning coding agent via the REWORK loop with grounded error text.
REWORK_ONLY_TYPES = frozenset({"SYNTAX", "IMPORT_NAME"})


# ── Phase 2: LLM Prompt & System Prompt ────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert integration and build engineer.
Your job is to automatically repair integration conflicts in the codebase with concrete, mechanical fixes — there is NO human in the loop, so every conflict that can possibly be fixed locally MUST receive a fix.

Prefer mechanical, safe, reversible fixes with zero ambiguity:
  ✓ Missing __init__.py file (action: create_file, replacement_content: "")
  ✓ Import/resolution path typo (action: replace_text)
  ✓ Casing-only rename (action: replace_text)
  ✓ Trailing slash mismatch in API paths (action: replace_text)
  ✓ Export exists in code but manifest missing it (action: update_manifest)
  ✓ Plausible, low-risk renames that match a sibling name (action: replace_text)

Only if a fix is genuinely infeasible without deciding business logic or rewriting a file's structure should you mark it "none":
  ✗ Deep behavioural rewrite / circular-import restructuring
  ✗ Where there are two plausible definitions and no way to pick automatically
When you mark a conflict "none", it is treated as unresolved and sent back to the owning coding agent for a REWORK round.

Return ONLY a single JSON object. No explanations outside the JSON.
Schema:
{
  "repairs": [
    {
      "conflict_index": <int: index of conflict in the list>,
      "action": "fix" | "none",
      "rationale": "<string: explanation>",
      "fix": {
        "action": "create_file" | "replace_text" | "update_manifest",
        "filepath": "<string: file path to modify>",
        "target_content": "<string: exact text/substring to replace (for replace_text)>",
        "replacement_content": "<string: replacement text (for replace_text/create_file)>"
      }
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """\
Here are the integration conflicts detected in the codebase:

{conflicts_json}

For each conflict, either propose a precise mechanical fix (action: "fix") or mark it (action: "none")
if it cannot be fixed without business decisions. Do not leave anything unanswered.
"""


# ── Parsing & Resolution Helpers ───────────────────────────────────────────────

def parse_python_imports(code: str) -> list[str]:
    imports = []
    # Match "import x, y.z"
    for match in re.finditer(r"^\s*import\s+([a-zA-Z0-9_\.,\s]+)", code, re.MULTILINE):
        modules = match.group(1).split(",")
        for m in modules:
            imports.append(m.strip())
    # Match "from x.y import z"
    for match in re.finditer(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import", code, re.MULTILINE):
        imports.append(match.group(1).strip())
    return imports


def parse_js_ts_imports(code: str) -> list[str]:
    imports = []
    # Match import ... from 'source'
    for match in re.finditer(r"import\s+[\s\S]*?\s+from\s+['\"]([^'\"]+)['\"]", code):
        imports.append(match.group(1).strip())
    # Match import 'source'
    for match in re.finditer(r"import\s+['\"]([^'\"]+)['\"]", code):
        imports.append(match.group(1).strip())
    # Match require('source')
    for match in re.finditer(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", code):
        imports.append(match.group(1).strip())
    return imports


def parse_python_imported_names(code: str) -> list[tuple[str, list[str], bool]]:
    """
    Extract (module_src, [exported_names], has_default) for every
    `from X import A, B` statement. Alias-aware: `from X import A as B`
    reports the *exported* name 'A' (what the target must define).
    """
    results: list[tuple[str, list[str], bool]] = []
    for match in re.finditer(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.+?)\s*$", code, re.MULTILINE):
        module_src = match.group(1).strip()
        import_part = match.group(2).strip()
        if import_part == "*":
            continue
        names: list[str] = []
        for item in import_part.split(","):
            item = item.strip()
            if not item:
                continue
            exported = item
            if " as " in item:
                exported = item.split(" as ")[0].strip()
            if exported:
                names.append(exported)
        if names:
            results.append((module_src, names, False))
    return results


def parse_js_ts_imported_names(code: str) -> list[tuple[str, list[str], bool]]:
    """
    Extract (module_src, [exported_names], has_default) for JS/TS imports.
    Alias-aware: `import { A as B } from 'x'` reports the *exported* name 'A'.
    Default imports / `default` renames set has_default (checked against a bare
    `export default` in the target). Type-only and star imports are skipped.
    """
    results: list[tuple[str, list[str], bool]] = []
    for match in re.finditer(r"import\s+[\s\S]*?\s+from\s+['\"]([^'\"]+)['\"]", code):
        header = match.group(0)
        if re.match(r"import\s+type\b", header):
            continue
        module_src = match.group(1).strip()
        names: list[str] = []
        has_default = False

        # Default import: `import A from 'x'` / `import A, { ... } from 'x'`
        default_match = re.match(r"import\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:,|\s+from)", header)
        if default_match:
            has_default = True

        # Named imports: `import { A, B as C, type T, default as D } from 'x'`
        for inner in re.finditer(r"\{([^}]*)\}", header):
            for item in inner.group(1).split(","):
                item = item.strip()
                if not item:
                    continue
                exported = item
                if " as " in item:
                    exported = item.split(" as ")[0].strip()
                if exported == "default":
                    has_default = True
                    continue
                if exported.startswith("type "):
                    exported = exported[5:].strip()
                if exported:
                    names.append(exported)

        if names or has_default:
            results.append((module_src, names, has_default))
    return results


def check_js_ts_extensions(base_path: str, manifest_files: list[str]) -> Optional[str]:
    for ext in ['.tsx', '.ts', '.jsx', '.js', '.css', '/index.tsx', '/index.ts', '/index.jsx', '/index.js']:
        path_with_ext = base_path + ext
        if path_with_ext in manifest_files:
            return path_with_ext
    if base_path in manifest_files:
        return base_path
    return None


def layer_root_for(filepath: str) -> Optional[str]:
    """
    Return the layer root directory for a generated file, e.g. 'output/backend/'
    for 'output/backend/db/session.py'. Returns None for paths that are not
    workspace-generated (e.g. test fixtures like 'app/...' or 'backend/app/...')
    so their resolution/fix behavior is left unchanged.
    """
    parts = (filepath or "").split('/')
    if len(parts) >= 2 and parts[0] == OUTPUT_DIR:
        return f"{OUTPUT_DIR}/{parts[1]}/"
    return None


def resolve_import_path(import_source: str, current_file: str, manifest_files: list[str], internal_prefixes: set[str]) -> Optional[str]:
    # Python resolution
    if current_file.endswith('.py'):
        if import_source.startswith('.'):
            dots_count = 0
            for char in import_source:
                if char == '.':
                    dots_count += 1
                else:
                    break
            rel_path = import_source[dots_count:]
            parts = current_file.split('/')
            if len(parts) > dots_count:
                base_dir = '/'.join(parts[:-dots_count])
                module_subpath = rel_path.replace('.', '/')
                candidate = f"{base_dir}/{module_subpath}" if module_subpath else base_dir
            else:
                return None
        else:
            first_part = import_source.split('.')[0]
            if first_part not in internal_prefixes:
                # Maybe a layer-root-relative import (e.g. `db.session` from a file
                # under output/backend/). Try resolving against the current file's
                # layer root before giving up.
                root = layer_root_for(current_file)
                if root:
                    candidate = root + import_source.replace('.', '/')
                    candidate = candidate.strip('/')
                    for ext in ['.py', '/__init__.py']:
                        path_with_ext = candidate + ext
                        if path_with_ext in manifest_files:
                            return path_with_ext
                    if candidate in manifest_files:
                        return candidate
                return None
            candidate = import_source.replace('.', '/')

        candidate = candidate.strip('/')
        for ext in ['.py', '/__init__.py']:
            path_with_ext = candidate + ext
            if path_with_ext in manifest_files:
                return path_with_ext
        if candidate in manifest_files:
            return candidate
        return None

    # JS/TS resolution
    elif current_file.endswith(('.js', '.jsx', '.ts', '.tsx', '.css')):
        if not import_source.startswith('.'):
            if import_source.startswith('@/'):
                sub = import_source[2:]
                for root in ['output/frontend/src/', 'frontend/src/', 'output/frontend/', 'frontend/']:
                    candidate = root + sub
                    resolved = check_js_ts_extensions(candidate, manifest_files)
                    if resolved:
                        return resolved
            # Check absolute workspace import path
            first_part = import_source.split('/')[0]
            if first_part in internal_prefixes:
                return check_js_ts_extensions(import_source, manifest_files)
            return None  # Third-party npm package

        parts = current_file.split('/')
        dir_path = '/'.join(parts[:-1]) if len(parts) > 1 else ""
        full_path = os.path.normpath(os.path.join(dir_path, import_source)).replace('\\', '/')
        return check_js_ts_extensions(full_path, manifest_files)

    return None


def extract_actual_definitions(code: str, filepath: str) -> set[str]:
    defs = set()
    if not code:
        return defs
    if filepath.endswith('.py'):
        for match in re.finditer(r"^\s*class\s+([a-zA-Z0-9_]+)", code, re.MULTILINE):
            defs.add(match.group(1))
        for match in re.finditer(r"^\s*def\s+([a-zA-Z0-9_]+)", code, re.MULTILINE):
            name = match.group(1)
            if not name.startswith('__'):
                defs.add(name)
        # Module-level constants / assignments (column-0 only, not inside functions),
        # so `from config import API_URL` can be validated against `API_URL = "..."`.
        _PY_KEYWORDS = {
            "and", "as", "assert", "async", "await", "break", "class", "continue",
            "def", "del", "elif", "else", "except", "finally", "for", "from",
            "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
            "or", "pass", "raise", "return", "try", "while", "with", "yield",
        }
        for line in code.splitlines():
            if line.startswith((" ", "\t")):
                continue
            stripped = line.lstrip()
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]*?)?=", stripped)
            if m:
                name = m.group(1)
                if name not in _PY_KEYWORDS and not name.startswith('__'):
                    defs.add(name)
    elif filepath.endswith(('.js', '.jsx', '.ts', '.tsx')):
        for match in re.finditer(r"\bexport\s+(?:default\s+)?(?:class|function|const|let|var)\s+([a-zA-Z0-9_]+)", code):
            defs.add(match.group(1))
        for match in re.finditer(r"\bexport\s+default\s+(?!function|class|const|let|var)([A-Za-z_$][A-Za-z0-9_$]*)", code):
            defs.add(match.group(1))
        for match in re.finditer(r"\bexport\s*\{\s*([^\}]+)\}", code):
            content = match.group(1)
            for item in content.split(','):
                item = item.strip()
                if not item:
                    continue
                if ' as ' in item:
                    name = item.split(' as ')[-1].strip()
                else:
                    name = item
                if name:
                    defs.add(name)
    elif filepath.endswith('.sql'):
        for match in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)", code, re.IGNORECASE):
            defs.add(match.group(1))
    return defs


def count_passed_arguments(args_str: str) -> int:
    args_str = args_str.strip()
    if not args_str:
        return 0
    count = 1
    depths = {'(': 0, '[': 0, '{': 0}
    in_quote = False
    quote_char = None
    escaped = False
    for char in args_str:
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if in_quote:
            if char == quote_char:
                in_quote = False
            continue
        if char in ('"', "'", '`'):
            in_quote = True
            quote_char = char
            continue
        if char in depths:
            depths[char] += 1
        elif char == ')':
            depths['('] = max(0, depths['('] - 1)
        elif char == ']':
            depths['['] = max(0, depths['['] - 1)
        elif char == '}':
            depths['{'] = max(0, depths['{'] - 1)
        elif char == ',':
            if all(d == 0 for d in depths.values()):
                count += 1
    return count


def normalize_api_path(path: str) -> str:
    path = re.sub(r'\{([^}]+)\}', r':\1', path)
    return path.strip().rstrip('/')


def check_type_consistency(field: str, expected_type: str, code: str) -> Optional[str]:
    expected_type_lower = expected_type.lower()
    # Pattern to search for "field: type" or "field?: type" or "field = ... as type"
    pattern = r"\b" + re.escape(field) + r"\b\s*\??\s*:\s*([a-zA-Z0-9_\[\]]+)"
    match = re.search(pattern, code)
    if not match:
        if field not in code:
            return f"Field '{field}' is missing entirely."
        return None

    actual_type = match.group(1).strip()
    actual_type_lower = actual_type.lower()

    is_match = False
    if "string" in expected_type_lower or "varchar" in expected_type_lower or "text" in expected_type_lower:
        is_match = actual_type_lower in ["str", "string", "varchar", "text"]
    elif "int" in expected_type_lower or "integer" in expected_type_lower or "number" in expected_type_lower:
        is_match = actual_type_lower in ["int", "number", "float", "integer"]
    elif "bool" in expected_type_lower:
        is_match = actual_type_lower in ["bool", "boolean"]
    elif "uuid" in expected_type_lower:
        is_match = actual_type_lower in ["uuid", "str", "string"]
    else:
        is_match = expected_type_lower in actual_type_lower or actual_type_lower in expected_type_lower

    if not is_match:
        return f"Type mismatch for '{field}': expected '{expected_type}', found '{actual_type}'"
    return None


def _normalize_output_path(filepath: str) -> str:
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith("output/") and not rel.startswith("output\\") and rel != "output":
        return f"output/{rel}"
    return rel


def get_file_code(filepath: str, fm: FileManifest, state: ProjectState) -> str:
    code = ""
    if fm.agent == "backend":
        code = state.generated_code.backend.get(filepath, "")
    elif fm.agent == "frontend":
        code = state.generated_code.frontend.get(filepath, "")
    elif fm.agent == "database":
        code = state.generated_code.database.get(filepath, "")
    elif fm.agent == "devops":
        code = state.generated_code.devops.get(filepath, "")

    if not code:
        abs_path = os.path.join(_root_dir, _normalize_output_path(filepath))
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception:
                pass
    return code


# ── Word-match deterministic resolver (Phase 1A) ───────────────────────────────

def _tokenize(s: str) -> set[str]:
    """Lower-cased alphanumeric tokens from an identifier/path, split on ./_- separators."""
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def word_match_score(a: str, b: str) -> float:
    """Jaccard token-overlap between two identifiers/paths (casing-normalized)."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def best_manifest_match(import_src: str, manifest_files: list[str], internal_prefixes: set[str], threshold: float = 0.4) -> Optional[str]:
    """
    Deterministic word-matching: find the manifest file whose path best matches
    *import_src* (a dotted/path import source). Returns the closest match or None.
    """
    best_path: Optional[str] = None
    best_score = threshold
    for path in manifest_files:
        path_score=word_match_score(import_src, path) 
        if path_score > best_score:
            best_path = path
            best_score = path_score
    return best_path


def dirname_matches(import_src: str, manifest_files: list[str]) -> Optional[str]:
    """
    If *import_src* looks like a Python package (e.g. 'backend.app.models') and any
    file in the manifest lives under that package directory, return the package
    directory path (so a missing __init__.py can be created).
    """
    pkg = import_src.replace(".", "/").strip("/")
    for path in manifest_files:
        prefix = pkg + "/"
        if path.startswith(prefix) and path.endswith(".py"):
            return pkg
    return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  IntegrationAgent                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

class IntegrationAgent(BaseAgent):
    """
    Validation and mechanical repairing agent for project code.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRIES,
        retries: int | None = None,
        backoff: float = DEFAULT_BACKOFF,
        temperature: float = 0.0,
        provider: BaseLLMProvider | str | None = None,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retries=retries,
            backoff=backoff,
            temperature=temperature,
            provider=provider,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT



    # ── Phase 1: Deterministic Static Checks ──────────────────────────────────

    def _check_single_file(self, filepath: str, fm: FileManifest, state: ProjectState, manifest_files: list[str], internal_prefixes: set[str], available_names: dict[str, set[str] | None]) -> tuple[list[dict], list[str]]:
        conflicts = []
        code = get_file_code(filepath, fm, state)

        if not code:
            # If not written, check if it's already declared as a known gap.
            # Planner/Agents mark unwritten dependencies, but if it is completely unwritten and has exports, we don't fail here.
            # Rule 9: "Do not report missing_dependencies or suggested_additions if already declared in the owning manifest entry."
            return conflicts, []

        # 1. Verify actual exports vs promised exports.
        #    A promised export may be satisfied by a definition OR a re-export
        #    (e.g. `__init__.py` doing `from .user import User`).
        actual_defs = extract_actual_definitions(code, filepath)
        reexport_names = set()
        if filepath.endswith('.py'):
            for _, names, _ in parse_python_imported_names(code):
                reexport_names.update(names)
        else:
            for _, names, _ in parse_js_ts_imported_names(code):
                reexport_names.update(names)
        for promised in fm.exports:
            if promised not in actual_defs and promised not in reexport_names:
                conflicts.append({
                    "filepath": filepath,
                    "task_id": fm.task_id,
                    "expected": promised,
                    "actual": "missing",
                    "reason": f"File '{filepath}' promised to export '{promised}', but it is missing in the generated code."
                })

        # Check for extra exports in code not declared in manifest exports
        for actual in actual_defs:
            # Filter out local helpers that aren't expected to be exported, but if JS/TS explicitly marks them as 'export', they are actual exports.
            if filepath.endswith(('.js', '.jsx', '.ts', '.tsx')) and actual not in fm.exports:
                conflicts.append({
                    "filepath": filepath,
                    "task_id": fm.task_id,
                    "expected": "not exported in manifest",
                    "actual": actual,
                    "reason": f"Export '{actual}' found in code of '{filepath}' but not declared in manifest exports list."
                })

        # 2. Verify imports resolve and verify function/method call signatures
        imports = []
        if filepath.endswith('.py'):
            imports = parse_python_imports(code)
        elif filepath.endswith(('.js', '.jsx', '.ts', '.tsx', '.css')):
            imports = parse_js_ts_imports(code)

        resolved_paths = []
        for imp in imports:
            resolved = resolve_import_path(imp, filepath, manifest_files, internal_prefixes)
            # If internal import but unable to resolve, it's a mismatch
            is_relative = imp.startswith('.') or imp.startswith('@/')
            is_internal_name = imp.split('.')[0] in internal_prefixes if filepath.endswith('.py') else imp.split('/')[0] in internal_prefixes

            if (is_relative or is_internal_name) and not resolved:
                conflicts.append({
                    "filepath": filepath,
                    "task_id": fm.task_id,
                    "expected": f"Resolvable module path for '{imp}'",
                    "actual": "unresolved",
                    "reason": f"Import '{imp}' in '{filepath}' cannot be resolved to any manifest file path."
                })
            elif resolved:
                resolved_paths.append(resolved)
                # Verify call signatures of functions/classes imported from 'resolved'
                ref_fm = state.code_manifest.files.get(resolved)
                if ref_fm:
                    sigs = {}
                    for c in ref_fm.classes:
                        sigs[c.name] = c.arguments
                    for f in ref_fm.functions:
                        sigs[f.name] = f.arguments

                    for name, args in sigs.items():
                        pattern = r"\b" + re.escape(name) + r"\s*\((.*?)\)"
                        for match in re.finditer(pattern, code, re.DOTALL):
                            args_str = match.group(1)
                            passed_count = count_passed_arguments(args_str)
                            min_args = sum(1 for a in args if not a.default)
                            max_args = len(args)
                            if passed_count < min_args or passed_count > max_args:
                                conflicts.append({
                                    "filepath": filepath,
                                    "task_id": fm.task_id,
                                    "expected": f"Signature for '{name}' requiring {min_args} to {max_args} args",
                                    "actual": f"Called with {passed_count} args",
                                    "reason": f"Call to '{name}' in '{filepath}' passed {passed_count} arguments, but target defines {len(args)} arguments ({min_args} required)."
                                })

        # 3. Verify imported names actually exist in their resolved target file.
        #    Catches runtime ImportErrors statically: `from app.models import User`
        #    where User is spelled wrong / missing / renamed in models.py.
        if filepath.endswith('.py'):
            name_specs = parse_python_imported_names(code)
        else:
            name_specs = parse_js_ts_imported_names(code)

        for module_src, names, has_default in name_specs:
            resolved = resolve_import_path(module_src, filepath, manifest_files, internal_prefixes)
            if not resolved:
                # Path-level mismatch — already reported by the main import loop above.
                continue

            target_avail = available_names.get(resolved)
            if target_avail is None:
                # Target code unavailable (not written / no source) — cannot verify, skip.
                continue

            if has_default:
                target_code = get_file_code(resolved, state.code_manifest.files.get(resolved), state)
                if "export default" not in (target_code or ""):
                    conflicts.append({
                        "type": "IMPORT_NAME",
                        "filepath": filepath,
                        "task_id": fm.task_id,
                        "expected": f"Default export in '{resolved}'",
                        "actual": "missing",
                        "reason": f"Import in '{filepath}' expects a default export from '{resolved}', but that file declares no 'export default'."
                    })

            for name in names:
                if name in target_avail:
                    continue
                avail_list = sorted(target_avail)[:10]
                conflicts.append({
                    "type": "IMPORT_NAME",
                    "filepath": filepath,
                    "task_id": fm.task_id,
                    "expected": f"Exported name '{name}' in '{resolved}'",
                    "actual": "missing",
                    "reason": f"Import 'from {module_src} import {name}' in '{filepath}' references '{name}', but '{resolved}' does not define or re-export it. Available names: {avail_list}"
                })

        return conflicts, resolved_paths

    def _compute_available_names(self, state: ProjectState, manifest_files: list[str]) -> dict[str, set[str] | None]:
        """
        For every manifest file, the set of names it makes importable:
        its own definitions (class/def/constants/exports) PLUS the names it
        imports itself (re-export support, e.g. `__init__.py` doing
        `from .user import User`). Returns None when the target's code is
        unavailable so callers can skip the name check instead of false-positive.
        """
        available: dict[str, set[str] | None] = {}
        for path in manifest_files:
            fm = state.code_manifest.files.get(path)
            code = get_file_code(path, fm, state)
            if not code:
                available[path] = None
                continue
            names = extract_actual_definitions(code, path)
            if path.endswith('.py'):
                for _, imported, _ in parse_python_imported_names(code):
                    names.update(imported)
            else:
                for _, imported, _ in parse_js_ts_imported_names(code):
                    names.update(imported)
            available[path] = names
        return available

    def _run_deterministic_checks(self, state: ProjectState) -> list[dict]:
        manifest = state.code_manifest
        manifest_files = list(manifest.files.keys())

        # Construct internal prefixes
        internal_prefixes = set()
        for path in manifest_files:
            parts = path.split('/')
            if parts and parts[0]:
                internal_prefixes.add(parts[0])

        conflicts = []
        resolved_imports_map: dict[str, list[str]] = {}

        # Precompute each file's importable names (own defs + re-exports) once.
        available_names = self._compute_available_names(state, manifest_files)

        # 1. Parallel verification per-file
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            futures: dict[Future, str] = {}
            for path, fm in manifest.files.items():
                if fm.written:
                    fut = pool.submit(self._check_single_file, path, fm, state, manifest_files, internal_prefixes, available_names)
                    futures[fut] = path

            for future, path in futures.items():
                try:
                    res_conflicts, resolved_paths = future.result()
                    if res_conflicts:
                        conflicts.extend(res_conflicts)
                    resolved_imports_map[path] = resolved_paths
                except Exception as exc:
                    logger.error("[IntegrationAgent] Verification task exception: %s", exc)

        # 2. Cycle Detection (report only, never auto-fix)
        graph = resolved_imports_map

        visited = {}
        path_stack = []

        def dfs(node: str) -> bool:
            visited[node] = 1
            path_stack.append(node)
            for neighbor in graph.get(node, []):
                if visited.get(neighbor, 0) == 1:
                    cycle_start = path_stack.index(neighbor)
                    cycle_path = path_stack[cycle_start:] + [neighbor]
                    conflicts.append({
                        "filepath": node,
                        "task_id": manifest.files[node].task_id,
                        "expected": "No circular imports",
                        "actual": f"Cycle: {' -> '.join(cycle_path)}",
                        "reason": f"Circular import cycle detected: {' -> '.join(cycle_path)}"
                    })
                    return True
                elif visited.get(neighbor, 0) == 0:
                    if dfs(neighbor):
                        return True
            path_stack.pop()
            visited[node] = 2
            return False

        for node in graph:
            if visited.get(node, 0) == 0:
                dfs(node)

        # 3. Cross-check Frontend and Backend API Contracts
        api_conflicts = self._check_api_contracts(state)
        conflicts.extend(api_conflicts)

        return conflicts

    def _check_api_contracts(self, state: ProjectState) -> list[dict]:
        conflicts = []
        contracts = state.architecture.api_contracts
        if not contracts:
            return conflicts

        backend_files = list(state.generated_code.backend.items())
        frontend_files = list(state.generated_code.frontend.items())

        for c in contracts:
            path = c.path.strip()
            path_norm = normalize_api_path(path)
            method = c.method.upper()

            # Check backend
            backend_found = False
            for filepath, code in backend_files:
                if path_norm in normalize_api_path(code) or path.rstrip('/') in code or path in code:
                    backend_found = True

                    # 1. Path slash consistency
                    has_contract_slash = path.endswith('/')
                    for quote in ["'", '"', '`']:
                        target_slash = path + '/' if not has_contract_slash else path
                        target_no_slash = path.rstrip('/') if has_contract_slash else path
                        if (quote + target_slash + quote) in code and not has_contract_slash:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Path without trailing slash: '{path}'",
                                "actual": f"Path with trailing slash: '{path}/'",
                                "reason": f"Trailing slash mismatch on route '{path}' in backend file '{filepath}'."
                            })
                            break
                        elif (quote + target_no_slash + quote) in code and has_contract_slash:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Path with trailing slash: '{path}'",
                                "actual": f"Path without trailing slash: '{path.rstrip('/')}'",
                                "reason": f"Trailing slash mismatch on route '{path}' in backend file '{filepath}'."
                            })
                            break

                    # 2. HTTP method
                    method_pattern = r"\b" + re.escape(method.lower()) + r"\b|\b" + re.escape(method.upper()) + r"\b"
                    if not re.search(method_pattern, code):
                        conflicts.append({
                            "filepath": filepath,
                            "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                            "expected": f"HTTP method '{method}'",
                            "actual": "Method not found or mismatched",
                            "reason": f"Backend route in '{filepath}' does not declare or use the HTTP method '{method}' for path '{path}'."
                        })

                    # 3. Request fields & types
                    for field, ftype in c.request_body.items():
                        err = check_type_consistency(field, ftype, code)
                        if err:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Request field '{field}' ({ftype})",
                                "actual": err,
                                "reason": f"Backend file '{filepath}' for path '{path}' has contract field issue: {err}."
                            })

                    # 4. Response fields & types
                    for field, ftype in c.response_body.items():
                        err = check_type_consistency(field, ftype, code)
                        if err:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Response field '{field}' ({ftype})",
                                "actual": err,
                                "reason": f"Backend file '{filepath}' for path '{path}' has contract field issue: {err}."
                            })

            if not backend_found:
                conflicts.append({
                    "filepath": "backend",
                    "task_id": "unknown",
                    "expected": f"Backend implementation for endpoint '{method} {path}'",
                    "actual": "Missing",
                    "reason": f"API contract endpoint '{method} {path}' is not implemented in any backend file."
                })

            # Check frontend
            frontend_found = False
            for filepath, code in frontend_files:
                if path_norm in normalize_api_path(code) or path.rstrip('/') in code or path in code:
                    frontend_found = True

                    # 1. Path slash consistency
                    has_contract_slash = path.endswith('/')
                    for quote in ["'", '"', '`']:
                        target_slash = path + '/' if not has_contract_slash else path
                        target_no_slash = path.rstrip('/') if has_contract_slash else path
                        if (quote + target_slash + quote) in code and not has_contract_slash:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Path without trailing slash: '{path}'",
                                "actual": f"Path with trailing slash: '{path}/'",
                                "reason": f"Trailing slash mismatch on API fetch of '{path}' in frontend file '{filepath}'."
                            })
                            break
                        elif (quote + target_no_slash + quote) in code and has_contract_slash:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Path with trailing slash: '{path}'",
                                "actual": f"Path without trailing slash: '{path.rstrip('/')}'",
                                "reason": f"Trailing slash mismatch on API fetch of '{path}' in frontend file '{filepath}'."
                            })
                            break

                    # 2. HTTP method
                    method_pattern = r"\b" + re.escape(method.lower()) + r"\b|\b" + re.escape(method.upper()) + r"\b"
                    if not re.search(method_pattern, code):
                        conflicts.append({
                            "filepath": filepath,
                            "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                            "expected": f"HTTP method '{method}' for call to '{path}'",
                            "actual": "Method not found or mismatched",
                            "reason": f"Frontend file '{filepath}' references route '{path}' but does not seem to specify HTTP method '{method}'."
                        })

                    # 3. Request fields & types
                    for field, ftype in c.request_body.items():
                        err = check_type_consistency(field, ftype, code)
                        if err:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Request field '{field}' ({ftype})",
                                "actual": err,
                                "reason": f"Frontend file '{filepath}' for API call '{path}' has contract field issue: {err}."
                            })

                    # 4. Response fields & types
                    for field, ftype in c.response_body.items():
                        err = check_type_consistency(field, ftype, code)
                        if err:
                            conflicts.append({
                                "filepath": filepath,
                                "task_id": state.code_manifest.files.get(filepath).task_id if filepath in state.code_manifest.files else "unknown",
                                "expected": f"Response field '{field}' ({ftype})",
                                "actual": err,
                                "reason": f"Frontend file '{filepath}' for API call '{path}' has contract field issue: {err}."
                            })

            if not frontend_found and backend_found:
                conflicts.append({
                    "filepath": "frontend",
                    "task_id": "unknown",
                    "expected": f"Frontend call/reference to endpoint '{method} {path}'",
                    "actual": "Missing",
                    "reason": f"API contract endpoint '{method} {path}' is implemented in backend but not called/referenced in any frontend file."
                })

        return conflicts


    # ── Phase 2: Deterministic word-match repair ─────────────────────────────

    def _resolve_by_word_match(self, state: ProjectState, conflicts: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Deterministically repair mechanical conflicts before invoking the LLM.

        Returns (remaining_conflicts, applied_logs). Conflicts it can fix with
        zero ambiguity are resolved inline and removed from the returned list.
        """
        manifest = state.code_manifest
        manifest_files = list(manifest.files.keys())
        internal_prefixes = set()
        for path in manifest_files:
            parts = path.split('/')
            if parts and parts[0]:
                internal_prefixes.add(parts[0])

        remaining: list[dict] = []
        applied_logs: list[str] = []

        for conflict in conflicts:
            filepath = conflict.get("filepath", "")
            reason = conflict.get("reason", "")
            fixed = False

            # 1. Import path typo — "Import 'X' in 'Y' cannot be resolved ..."
            m = re.search(r"Import '([^']+)' in '([^']+)' cannot be resolved", reason)
            if m and not fixed:
                import_src = m.group(1)
                current_file = m.group(2)
                import_src = import_src.strip()

                # Skip relative imports that point outside the tree
                if import_src.startswith(".."):
                    remaining.append(conflict)
                    continue

                if import_src.startswith("."):
                    dots_count = len(import_src) - len(import_src.lstrip("."))
                    rel_path = import_src[dots_count:]
                    parts = current_file.split('/')
                    if len(parts) > dots_count:
                        base_dir = '/'.join(parts[:-dots_count])
                        candidate = f"{base_dir}/{rel_path.replace('.', '/')}" if rel_path else base_dir
                    else:
                        remaining.append(conflict)
                        continue
                else:
                    candidate = import_src.replace('.', '/')

                candidate = candidate.strip('/')

                # 1a. Exact or extension-resolved path already known to the resolver?
                resolved_path = resolve_import_path(import_src, current_file, manifest_files, internal_prefixes)
                if resolved_path:
                    fixed = True
                else:
                    # 1b. Word-match the (wrong) import against every manifest path.
                    best = best_manifest_match(candidate, manifest_files, internal_prefixes)
                    if best:
                        fix_target = self._import_fix_target(import_src, current_file, best)
                        if fix_target:
                            applied = self._apply_repairs(state, [{
                                "conflict_index": 0,
                                "action": "fix",
                                "rationale": conflict.get("reason", ""),
                                "fix": fix_target,
                            }], [conflict])
                            applied_logs.extend(applied)
                            fixed = bool(applied)
                            if fixed:
                                logger.info("[IntegrationAgent] Word-match fixed import '%s' -> %s", import_src, best)

                # 1c. Missing package __init__.py that can be auto-created.
                if not fixed:
                    pkg_dir = dirname_matches(import_src, manifest_files)
                    if pkg_dir:
                        init_path = pkg_dir + "/__init__.py"
                        if init_path not in manifest_files:
                            applied = self._apply_repairs(state, [{
                                "conflict_index": 0,
                                "action": "fix",
                                "rationale": conflict.get("reason", ""),
                                "fix": {
                                    "action": "create_file",
                                    "filepath": init_path,
                                    "replacement_content": "",
                                },
                            }], [conflict])
                            applied_logs.extend(applied)
                            fixed = bool(applied)
                            if fixed:
                                logger.info("[IntegrationAgent] Word-match created missing package file %s", init_path)

            # 2. Export declared in code but missing from the manifest → sync manifest.
            m = re.search(r"Export '([^']+)' found in code of '([^']+)' but not declared in manifest", reason)
            if m and not fixed:
                actual_export, code_file = m.group(1), m.group(2)
                fm = manifest.files.get(code_file)
                if fm and actual_export not in (fm.exports or []):
                    fm.exports.append(actual_export)
                    applied_logs.append(f"Synchronized manifest exports for {code_file} with '{actual_export}'")
                    fixed = True

            if not fixed:
                remaining.append(conflict)

        return remaining, applied_logs

    def _import_fix_target(self, import_src: str, current_file: str, best_path: str) -> Optional[dict]:
        """
        Build a concrete replace_text fix that rewrites *import_src* to the
        resolved manifest path *best_path*, preserving the import statement style.
        """
        if current_file.endswith(".py"):
            module_path = best_path
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            if module_path.endswith("__init__"):
                module_path = module_path[: -len("__init__")].rstrip("/")
            # Import paths are layer-root-relative: strip the `output/<layer>/`
            # prefix so `output/backend/db/session.py` → `db.session` (works when
            # the app runs from inside `output/backend/`).
            root = layer_root_for(current_file)
            if root and module_path.startswith(root):
                module_path = module_path[len(root):]
            if import_src.startswith("."):
                replacement = "." * (len(import_src) - len(import_src.lstrip(".")))
                # Keep the relative form if we can recompute it; else use the absolute dotted form.
                import_path = module_path.replace("/", ".")
            else:
                import_path = module_path.replace("/", ".")
            return {
                "action": "replace_text",
                "filepath": current_file,
                "target_content": import_src,
                "replacement_content": import_path,
            }

        # JS/TS: rewrite the source string to the resolved path (strip extension for TS path aliases).
        import_path = best_path
        if import_path.endswith((".ts", ".tsx")):
            import_path = import_path[:-3]
        elif import_path.endswith((".js", ".jsx")):
            import_path = import_path[:-3]
        return {
            "action": "replace_text",
            "filepath": current_file,
            "target_content": import_src,
            "replacement_content": import_path,
        }

    # ── Phase 3: Batched LLM repair ───────────────────────────────────────────

    def _get_llm_repairs(self, conflicts: list[dict]) -> list[dict]:
        try:
            prompt = USER_PROMPT_TEMPLATE.format(conflicts_json=json.dumps(conflicts, indent=2))
            resp = self.llm_call(prompt)
            from agents.schemas.integration import IntegrationRepairSchema
            data = self.extract_json_and_validate(resp.text, IntegrationRepairSchema)
            return data.get("repairs", [])
        except Exception as exc:
            logger.error("[IntegrationAgent] Batched LLM repair call failed: %s", exc)
            return []

    def _apply_repairs(self, state: ProjectState, repairs: list[dict], conflicts: list[dict]) -> list[str]:
        applied_logs = []
        for item in repairs:
            idx = item.get("conflict_index")
            if idx is None or idx < 0 or idx >= len(conflicts):
                continue

            conflict = conflicts[idx]
            if item.get("action") != "fix":
                continue

            fix = item.get("fix")
            if not fix:
                continue

            action = fix.get("action")
            filepath = fix.get("filepath")
            if not filepath:
                continue

            fm = state.code_manifest.files.get(filepath)
            agent = fm.agent if fm else ("backend" if "backend" in filepath else "frontend")

            abs_path = os.path.join(_root_dir, _normalize_output_path(filepath))

            if action == "create_file":
                content = fix.get("replacement_content", "")
                try:
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    with open(abs_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    if agent == "backend":
                        state.generated_code.backend[filepath] = content
                    elif agent == "frontend":
                        state.generated_code.frontend[filepath] = content
                    elif agent == "database":
                        state.generated_code.database[filepath] = content
                    elif agent == "devops":
                        state.generated_code.devops[filepath] = content

                    if filepath not in state.code_manifest.files:
                        state.code_manifest.files[filepath] = FileManifest(
                            agent=agent,
                            task_id=conflict.get("task_id", "auto-fix"),
                            filepath=filepath,
                            purpose="Auto-created file",
                            written=True
                        )
                    else:
                        state.code_manifest.files[filepath].written = True

                    applied_logs.append(f"Created file: {filepath}")
                except Exception as e:
                    logger.warning("[IntegrationAgent] Failed to create file %s: %s", filepath, e)

            elif action == "replace_text":
                target = fix.get("target_content")
                replacement = fix.get("replacement_content")
                if target is None or replacement is None:
                    continue

                code = get_file_code(filepath, fm, state) if fm else ""
                if not code and os.path.exists(abs_path):
                    try:
                        with open(abs_path, "r", encoding="utf-8") as f:
                            code = f.read()
                    except Exception:
                        pass

                if target in code:
                    new_code = code.replace(target, replacement)
                    try:
                        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(new_code)

                        if agent == "backend":
                            state.generated_code.backend[filepath] = new_code
                        elif agent == "frontend":
                            state.generated_code.frontend[filepath] = new_code
                        elif agent == "database":
                            state.generated_code.database[filepath] = new_code
                        elif agent == "devops":
                            state.generated_code.devops[filepath] = new_code

                        applied_logs.append(f"Replaced text in {filepath}")
                    except Exception as e:
                        logger.warning("[IntegrationAgent] Failed to write file %s: %s", filepath, e)

            elif action == "update_manifest":
                expected_export = conflict.get("expected")
                if expected_export and fm:
                    if expected_export not in fm.exports:
                        fm.exports.append(expected_export)
                        applied_logs.append(f"Synchronized manifest exports for {filepath} with '{expected_export}'")

        return applied_logs

    # ── Phase 1B: Grounded syntax/compile smoke ───────────────────────────────

    @staticmethod
    def _syntax_check_python(code: str) -> list[dict]:
        """ast.parse the code string — catches SyntaxError without executing or resolving imports."""
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            line = e.lineno or 0
            col = e.offset or 0
            lines = code.splitlines()
            src_line = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            return [{"line": line, "col": col, "msg": e.msg or "syntax error", "src_line": src_line}]
        except Exception as exc:
            return [{"line": 0, "col": 0, "msg": f"parse failure: {exc}", "src_line": ""}]

    @staticmethod
    def _syntax_check_js(code: str, node_bin: str) -> list[dict]:
        """`node --check` via stdin — pure syntax validation, no npm/node_modules/disk."""
        try:
            proc = subprocess.run(
                [node_bin, "--check"],
                input=code,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as exc:
            logger.warning("[IntegrationAgent] node --check invocation failed: %s", exc)
            return []
        if proc.returncode == 0:
            return []
        msg = (proc.stderr or proc.stdout or "").strip()
        m = re.search(r"\[stdin\]:(\d+)(?::(\d+))?", msg)
        line = int(m.group(1)) if m else 0
        col = int(m.group(2)) if m and m.group(2) else 0
        return [{"line": line, "col": col, "msg": msg, "src_line": ""}]

    def _run_syntax_smoke(self, state: ProjectState) -> list[dict]:
        """
        Per-file parse/compile check on in-memory generated code. Produces
        grounded SYNTAX conflicts that are routed to the owning coding agent.
        Graceful degradation: when tooling is absent (no node) the check is
        skipped with a debug log — never a fabricated conflict, never a crash.
        """
        node_bin = shutil.which("node")
        conflicts: list[dict] = []
        for path, fm in state.code_manifest.files.items():
            if not fm.written:
                continue
            code = get_file_code(path, fm, state)
            if not code:
                continue

            if path.endswith(".py"):
                errors = self._syntax_check_python(code)
                lang = "Python"
            elif path.endswith((".js", ".mjs", ".cjs")):
                if node_bin is None:
                    logger.debug("[IntegrationAgent] node unavailable; skipping JS syntax check for %s", path)
                    continue
                errors = self._syntax_check_js(code, node_bin)
                lang = "JavaScript"
            else:
                # .ts/.tsx/.css/.sql/.html → no local syntax tooling; skip.
                continue

            for err in errors:
                loc = f"line {err['line']}" if err.get("line") else "unknown line"
                conflicts.append({
                    "type": "SYNTAX",
                    "filepath": path,
                    "task_id": fm.task_id,
                    "expected": f"Valid {lang} syntax",
                    "actual": "syntax error",
                    "reason": f"Syntax error in '{path}' ({loc}): {err['msg']}",
                })
        return conflicts

    # ── Phase 1C: Workspace-prefix import normalization ────────────────────────

    def _normalize_workspace_imports(self, state: ProjectState) -> list[str]:
        """
        Rewrite Python imports written with the workspace root prefix
        (`from output.backend.db.session import …`) into layer-root-relative form
        (`from db.session import …`). This is the form that works when the app is
        run from inside its own folder (e.g. `output/backend/`).

        Mechanical + reversible (string prefix strip only), no LLM involved.
        Returns the list of applied fix logs; no-op when nothing to normalize.
        """
        applied_logs: list[str] = []
        for path, fm in state.code_manifest.files.items():
            if not fm.written or not path.endswith(".py"):
                continue
            root = layer_root_for(path)
            if not root:
                continue
            layer_dotted = root.strip("/").replace("/", ".")
            code = get_file_code(path, fm, state)
            if not code:
                continue

            repairs: list[dict] = []
            # `from output.backend.X import ...` / `import output.backend.X`
            # (dotted submodule form only — bare `import output.backend` is left alone).
            for match in re.finditer(
                r"(from|import)\s+" + re.escape(layer_dotted) + r"\.", code
            ):
                # Match at the *start* of an import statement (line anchor),
                # so `from output.backend...` becomes `from ...` while leaving
                # already-relative imports untouched.
                line_start = code.rfind("\n", 0, match.start()) + 1
                prefix = code[line_start:match.start()]
                if not re.match(r"^\s*$", prefix):
                    continue
                target = match.group(0)
                replacement = match.group(1) + " "
                repairs.append({
                        "conflict_index": 0,
                        "action": "fix",
                        "rationale": "Normalize workspace-prefixed import to layer-root-relative form",
                        "fix": {
                            "action": "replace_text",
                            "filepath": path,
                            "target_content": target,
                            "replacement_content": replacement,
                        },
                    })

            if repairs:
                applied_logs.extend(self._apply_repairs(state, repairs, [{}] * len(repairs)))
        if applied_logs:
            logger.info("[IntegrationAgent] Normalized %d workspace-prefixed import(s).", len(applied_logs))
        return applied_logs

    # ── Main Execution ────────────────────────────────────────────────────────

    def run(self, state: dict | ProjectState) -> dict[str, Any]:
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)
        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        # Bump the graph-level REWORK counter — each invocation is one rework round.
        ps.integration_rework_rounds += 1

        logger.info("[IntegrationAgent] Starting Phase 1 checks…")
        conflicts = self._run_deterministic_checks(ps)

        # Phase 1A — deterministic word-match repair (no LLM).
        if conflicts:
            conflicts, word_applied = self._resolve_by_word_match(ps, conflicts)
            if word_applied:
                logger.info("[IntegrationAgent] Word-match applied %d mechanical fix(es). Re-checking…", len(word_applied))
                conflicts = self._run_deterministic_checks(ps)

        # Phase 1C — workspace-prefix import normalization (no LLM). Rewrites
        # `from output.backend.db.session import …` → `from db.session import …`
        # so generated code is root-relative to its own layer folder.
        if self._normalize_workspace_imports(ps):
            logger.info("[IntegrationAgent] Re-checking after import normalization…")
            conflicts = self._run_deterministic_checks(ps)

        # Phase 1B — grounded syntax/compile smoke. Real parser errors are routed
        # straight to the owning coding agents; integration never guesses a parse fix.
        syntax_conflicts = self._run_syntax_smoke(ps)
        if syntax_conflicts:
            logger.info("[IntegrationAgent] Syntax smoke found %d real error(s).", len(syntax_conflicts))
        conflicts = conflicts + syntax_conflicts

        # Phase 2 — batched LLM repair on static/contract conflicts only.
        # SYNTAX / IMPORT_NAME conflicts bypass the LLM (REWORK_ONLY_TYPES).
        llm_repairable = [c for c in conflicts if c.get("type") not in REWORK_ONLY_TYPES]
        if llm_repairable:
            max_repair_rounds = 2
            for round_idx in range(max_repair_rounds):
                logger.info(
                    "[IntegrationAgent] Repair round %d/%d — %d conflicts. Running batched LLM repair…",
                    round_idx + 1, max_repair_rounds, len(llm_repairable),
                )
                repairs = self._get_llm_repairs(llm_repairable)
                if not repairs:
                    logger.warning("[IntegrationAgent] No repair proposals returned; stopping repair loop.")
                    break

                applied = self._apply_repairs(ps, repairs, llm_repairable)
                conflicts = self._run_deterministic_checks(ps)
                conflicts = conflicts + self._run_syntax_smoke(ps)

                llm_repairable = [c for c in conflicts if c.get("type") not in REWORK_ONLY_TYPES]
                if not llm_repairable:
                    break
                if not applied:
                    logger.info("[IntegrationAgent] No fixes could be applied; stopping repair loop.")
                    break

        merged = len(conflicts) == 0
        conflict_strings = [
            f"[{c.get('type', 'MISMATCH').upper()}] File: {c.get('filepath')} (task: {c.get('task_id')}) | Expected: {c.get('expected')} | Actual: {c.get('actual')} | Reason: {c.get('reason')}"
            for c in conflicts
        ]

        # Group truly-unresolved conflicts by owning agent for the graph REWORK edge.
        conflicts_by_agent: dict[str, list[str]] = {}
        rework_targets: dict[str, list[str]] = {}
        for c, c_str in zip(conflicts, conflict_strings):
            fpath = c.get("filepath", "")
            fm = ps.code_manifest.files.get(fpath)
            agent = fm.agent if fm else "backend"
            if "frontend" in fpath:
                agent = "frontend"
            elif "database" in fpath or fpath.endswith(".sql"):
                agent = "database"
            elif "backend" in fpath:
                agent = "backend"
            conflicts_by_agent.setdefault(agent, []).append(c_str)
            rework_targets.setdefault(fpath, []).append(c_str)

        ps.integration = IntegrationStatus(
            merged=merged,
            conflicts=conflict_strings,
            unified_entry_point="output/frontend/index.html" if "output/frontend/index.html" in ps.code_manifest.files else "",
            conflicts_by_agent=conflicts_by_agent,
            rework_targets=rework_targets,
        )

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)

        logger.info(
            "[IntegrationAgent] Done in %.2fs. Merged: %s. Remaining conflicts: %d. Rework rounds: %d/%d.",
            duration, merged, len(conflicts), ps.integration_rework_rounds, ps.max_integration_rounds,
        )
        return self._compute_diff(before, ps)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Smoke-tests  (python agents/integration.py)                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╗
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IntegrationAgent — unit tests")
    print("=" * 60)

    # ── 1. Count passed arguments ──────────────────────────────────────────────
    assert count_passed_arguments("a, b, c") == 3
    assert count_passed_arguments("a, [b, c], d") == 3
    assert count_passed_arguments("a, fn(b, c), d") == 3
    assert count_passed_arguments("'a, b', c") == 2
    assert count_passed_arguments("") == 0
    print("[PASS] count_passed_arguments — complex nested cases correct")

    # ── 2. Mock integration check run ─────────────────────────────────────────
    from providers.llm.base import BaseLLMProvider, LLMResponse
    from core.state import CodeMap, ArchData

    class MockLLM(BaseLLMProvider):
        @property
        def provider_name(self) -> str: return "mock"
        def complete(self, system_prompt, user_message, model, max_tokens=8192, temperature=0.2):
            payload = {
                "repairs": [
                    {
                        "conflict_index": 0,
                        "action": "fix",
                        "rationale": "Typo in import path",
                        "fix": {
                            "action": "replace_text",
                            "filepath": "backend/app/routes/auth.py",
                            "target_content": "backend.app.model.user",
                            "replacement_content": "backend.app.models.user"
                        }
                    }
                ]
            }
            return LLMResponse(text=f"```json\n{json.dumps(payload)}\n```",
                               input_tokens=10, output_tokens=10, model="mock", provider="mock")

    state = ProjectState(project_id="int-test", project_name="Int Test")
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI", "frontend": "React"},
        api_contracts=[
            APIContract(method="POST", path="/api/v1/auth/login", auth_required=False,
                        description="Login", request_body={"email": "string"}, response_body={})
        ]
    )
    
    fm1 = FileManifest(
        agent="backend", task_id="be-001",
        filepath="backend/app/models/user.py",
        purpose="User model", exports=["User"], written=True,
        classes=[ClassName(name="User", arguments=[], docstring="")]
    )
    fm2 = FileManifest(
        agent="backend", task_id="be-002",
        filepath="backend/app/routes/auth.py",
        purpose="Auth router", exports=["login_router"], written=True,
        functions=[FunctionName(name="login_router", arguments=[], return_type="None", docstring="")]
    )
    fm3 = FileManifest(
        agent="frontend", task_id="fe-001",
        filepath="frontend/src/pages/Login.tsx",
        purpose="Login page", exports=["LoginPage"], written=True,
        functions=[FunctionName(name="LoginPage", arguments=[], return_type="JSX.Element", docstring="")]
    )
    state.code_manifest = CodeManifest(files={fm1.filepath: fm1, fm2.filepath: fm2, fm3.filepath: fm3})

    code_user = "class User:\n    pass\n"
    code_auth = """\
from backend.app.model.user import User
# @router.post("/api/v1/auth/login")
# email: str
def login_router():
    pass
"""
    code_login = """\
export const LoginPage = () => {
    // email: string
    fetch('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ email }) });
}
"""
    state.generated_code = CodeMap(
        backend={
            "backend/app/models/user.py": code_user,
            "backend/app/routes/auth.py": code_auth
        },
        frontend={
            "frontend/src/pages/Login.tsx": code_login
        }
    )

    agent = IntegrationAgent(provider=MockLLM())
    result = agent.run(state)

    # Clean up created files/folders during smoke test
    import shutil
    for path in ["backend", "frontend"]:
        abs_p = os.path.join(_root_dir, path)
        if os.path.exists(abs_p):
            shutil.rmtree(abs_p, ignore_errors=True)

    assert state.integration.merged, f"Expected clean integration merge, got conflicts: {state.integration.conflicts}"
    assert "models.user" in state.generated_code.backend["backend/app/routes/auth.py"]
    print("[PASS] IntegrationAgent mock run — auto-fixed typo and re-verified successfully")

    # ── 3. Imported-name parsers ─────────────────────────────────────────────
    py_specs = parse_python_imported_names("from app.models import User, create_user as mk\nfrom x import *\nimport y\n")
    assert py_specs == [("app.models", ["User", "create_user"], False)], py_specs
    js_specs = parse_js_ts_imported_names(
        "import App from './App';\n"
        "import { getUsers, type T as Alias } from './api';\n"
        "import type { Props } from './types';\n"
    )
    assert js_specs[0] == ("./App", [], True), js_specs
    assert js_specs[1] == ("./api", ["getUsers", "T"], False), js_specs
    assert len(js_specs) == 2, js_specs  # type-only import skipped
    print("[PASS] parse_python_imported_names / parse_js_ts_imported_names — aliases, defaults, star/type skips")

    # ── 4. Broadened definitions (constants + default exports) ───────────────
    py_defs = extract_actual_definitions("API_URL = 'http://x'\nclass User:\n    pass\n", "app.py")
    assert {"API_URL", "User"} <= py_defs, py_defs
    js_defs = extract_actual_definitions("export default App\nexport const API_URL = 1\n", "app.ts")
    assert {"App", "API_URL"} <= js_defs, js_defs
    print("[PASS] extract_actual_definitions — module-level constants + bare default export")

    # ── 5. Syntax smoke helpers ──────────────────────────────────────────────
    assert IntegrationAgent._syntax_check_python("def f(:\n    pass\n"), "expected a SyntaxError from broken python"
    errs = IntegrationAgent._syntax_check_python("def f(:\n    pass\n")
    assert errs and errs[0]["line"] == 1 and errs[0]["col"], errs
    assert IntegrationAgent._syntax_check_python("x = 1\n") == []
    node_bin = shutil.which("node")
    if node_bin:
        assert IntegrationAgent._syntax_check_js("const x = 1;\n", node_bin) == []
        assert IntegrationAgent._syntax_check_js("const x = ;\n", node_bin), "expected a SyntaxError from broken js"
    else:
        print("[WARN] node unavailable — skipping node --check assertions")
    print("[PASS] Syntax smoke helpers — ast.parse + node --check")

    # ── 6. End-to-end: syntax + imported-name conflicts route to rework ──────
    state2 = ProjectState(project_id="int-test2", project_name="Int Test 2")
    code_user2 = "class User:\n    pass\n"
    code_bad_syntax = "def broken(:\n    return 1\n"
    code_bad_import = "from app.models import NotAThing\n"
    code_reexport = "from .user import User\n"
    fm_a = FileManifest(agent="backend", task_id="be-a", filepath="app/models/user.py",
                        purpose="User", exports=["User"], written=True)
    fm_b = FileManifest(agent="backend", task_id="be-b", filepath="app/broken.py",
                        purpose="Broken", exports=["broken"], written=True)
    fm_c = FileManifest(agent="backend", task_id="be-c", filepath="app/badimport.py",
                        purpose="Bad import", exports=["badimport"], written=True)
    fm_d = FileManifest(agent="backend", task_id="be-d", filepath="app/models/__init__.py",
                        purpose="Re-export", exports=["User"], written=True)
    state2.code_manifest = CodeManifest(files={
        "app/models/user.py": fm_a,
        "app/broken.py": fm_b,
        "app/badimport.py": fm_c,
        "app/models/__init__.py": fm_d,
    })
    state2.generated_code = CodeMap(backend={
        "app/models/user.py": code_user2,
        "app/broken.py": code_bad_syntax,
        "app/badimport.py": code_bad_import,
        "app/models/__init__.py": code_reexport,
    })

    agent2 = IntegrationAgent(provider=MockLLM())
    result2 = agent2.run(state2)
    assert not state2.integration.merged
    conflict_strings2 = state2.integration.conflicts
    assert any(s.startswith("[SYNTAX]") for s in conflict_strings2), conflict_strings2
    assert any(s.startswith("[IMPORT_NAME]") for s in conflict_strings2), conflict_strings2
    rework_for = {c for c in state2.integration.rework_targets.get("app/badimport.py", [])}
    assert any("NotAThing" in c for c in rework_for), state2.integration.rework_targets
    print("[PASS] End-to-end — SYNTAX + IMPORT_NAME conflicts routed to rework targets")

    # ── 7. Workspace-prefix import normalization (layer-root-relative) ────────
    state3 = ProjectState(project_id="int-test3", project_name="Int Test 3")
    fm_e = FileManifest(agent="backend", task_id="be-e", filepath="output/backend/norm_main.py",
                        purpose="Main", exports=["app"], written=True)
    fm_f = FileManifest(agent="backend", task_id="be-f", filepath="output/backend/db/session.py",
                        purpose="Session", exports=["async_engine"], written=True)
    state3.code_manifest = CodeManifest(files={
        "output/backend/norm_main.py": fm_e,
        "output/backend/db/session.py": fm_f,
    })
    code_main3 = "from output.backend.db.session import async_engine\nfrom db.session import async_engine as rel\n\napp = None\n"
    state3.generated_code = CodeMap(backend={
        "output/backend/norm_main.py": code_main3,
        "output/backend/db/session.py": "async_engine = None\n",
    })
    agent3 = IntegrationAgent(provider=MockLLM())
    result3 = agent3.run(state3)
    norm_code3 = state3.generated_code.backend["output/backend/norm_main.py"]
    assert "from db.session import async_engine" in norm_code3, norm_code3
    assert "output.backend" not in norm_code3, norm_code3
    assert "as rel" in norm_code3, norm_code3  # already-relative import untouched
    print("[PASS] Workspace-prefix imports normalized to layer-root-relative form")

    # Clean up the file written by the normalization smoke run (it lives under output/).
    abs_p3 = os.path.join(_root_dir, "output/backend/norm_main.py")
    if os.path.exists(abs_p3):
        os.remove(abs_p3)
    print("=" * 60 + "\n")
