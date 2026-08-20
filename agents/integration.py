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
    from .base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF, deep_unescape, record_corrupt_write
except ImportError:
    from base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF, deep_unescape, record_corrupt_write

from agents.utils import detect_corruption, safe_write

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
REWORK_ONLY_TYPES = frozenset({
    "SYNTAX", "IMPORT_NAME", "STARTUP", "SCHEMA_COLUMN", "WIRING",
    "FRONTEND_HTTP", "FRONTEND_DOM", "FILE_SET", "FILE_CORRUPTION",
})

#: Pydantic base-model names that mark a class as a request/response schema.
_PYDANTIC_BASES = ("BaseModel", "PydanticBaseModel", "BaseSettings")

#: Conventional app-layer directory names. Imports whose first segment is one
#: of these are INTERNAL even when the planner never registered the module
#: (e.g. `from api.routes import router` with no api/ package on disk) — they
#: must be flagged so the owning agent is told to create the module or fix the
#: import, instead of silently passing as a third-party package.
_LAYER_DIR_NAMES = frozenset({
    "api", "routers", "routes", "app", "services", "models", "schemas",
    "db", "crud", "database", "controllers", "views", "utils", "core",
    "middleware", "config", "static", "assets", "migrations",
})

#: HTTP verbs that routers can declare.
_HTTP_VERBS = ("get", "post", "put", "patch", "delete")

#: DB foundation symbols that must be defined in EXACTLY ONE backend file.
#: Two files defining them (e.g. db/models.py + db/init_db.py) create duplicate
#: connections + two declarative bases and the import graph never converges.
_FOUNDATION_SYMBOLS = ("engine", "Base", "SessionLocal")

_REFUSED_FILEPATH_RE = re.compile(r"File:\s*([^\s|]+)")


def _extract_refused_filepath(conflict_str: str) -> str:
    """Pull the filepath out of a safe-write refusal string."""
    m = _REFUSED_FILEPATH_RE.search(conflict_str or "")
    return m.group(1) if m else ""

#: Fetch-response methods that are NOT entity fields (frontend usage analysis).
_JS_MEMBER_SAFE = frozenset({
    "json", "text", "ok", "status", "statusText", "headers", "then", "catch",
    "finally", "push", "pop", "shift", "unshift", "splice", "concat", "slice",
    "map", "filter", "reduce", "forEach", "find", "includes", "indexOf",
    "length", "trim", "toLowerCase", "toUpperCase", "replace", "split", "join",
    "appendChild", "removeChild", "addEventListener", "classList", "dataset",
    "setAttribute", "getAttribute", "focus", "innerHTML", "textContent", "value",
})


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
    for match in re.finditer(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+([^\n]*)", code, re.MULTILINE):
        module_src = match.group(1).strip()
        import_part = match.group(2).strip()
        if import_part == "*":
            continue
        # Multiline parenthesized import: `from x import (\n    a,\n    b,\n)`
        if "(" in import_part:
            if ")" in import_part:
                import_part = import_part[import_part.index("(") + 1: import_part.rindex(")")]
            else:
                rest = code[match.end():]
                close = rest.find(")")
                import_part = import_part[import_part.index("(") + 1:] + (rest[:close] if close != -1 else "")
        names: list[str] = []
        for item in import_part.replace("\n", " ").split(","):
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
                candidates = [import_source]
                if root:
                    root_name = root.strip('/').split('/')[-1]
                    if first_part == root_name:
                        # `backend.schemas.todo` from a file under output/backend/
                        # means the app runs with the layer root on sys.path.
                        candidates.append(import_source.split('.', 1)[1])
                for src in candidates:
                    candidate = (root + src.replace('.', '/')) if root else (src.replace('.', '/'))
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
            if first_part in _LAYER_DIR_NAMES:
                # Conventional layer-relative import (e.g. `api/todo` from a
                # frontend module): try the current file's layer root, then the
                # workspace frontend roots, before declaring it third-party.
                root = layer_root_for(current_file)
                if root:
                    resolved = check_js_ts_extensions(root + import_source, manifest_files)
                    if resolved:
                        return resolved
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
        for match in re.finditer(r"^\s*(?:async\s+)?def\s+([a-zA-Z0-9_]+)", code, re.MULTILINE):
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
        for match in re.finditer(r"\bexport\s+(?:default\s+)?(?:async\s+)?(?:class|function|const|let|var)\s+([a-zA-Z0-9_]+)", code):
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


def extract_call_arg_strings(code: str, name: str) -> list[str]:
    """
    Inner argument strings of every call to `name(` in `code`, extracted with a
    balanced-paren, quote-aware scan (unlike a naive regex, which stops at the
    first ')' — e.g. inside `str(x)` — and miscounts nested calls).
    `def name(` declarations are skipped; attribute/import usages never match
    because a '(' must directly follow the name.
    """
    results: list[str] = []
    pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    pos = 0
    while True:
        m = pattern.search(code, pos)
        if not m:
            break
        if re.search(r"\bdef\s+$", code[: m.start()]):
            pos = m.end()
            continue
        depth = 0
        in_quote = False
        quote_char = None
        escaped = False
        idx = m.end() - 1
        while idx < len(code):
            ch = code[idx]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif in_quote:
                if ch == quote_char:
                    in_quote = False
            elif ch in ('"', "'", "`"):
                in_quote = True
                quote_char = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    results.append(code[m.end() : idx])
                    break
            idx += 1
        pos = m.end()
    return results


def normalize_api_path(path: str) -> str:
    # Path-typed params ({file_path:path}) are catch-alls — drop the param
    # entirely so `/static/{file_path:path}` normalizes to `/static`.
    path = re.sub(r'\{[^}]*:\w+\}', '', path)
    path = re.sub(r'\{([^}]+)\}', r':\1', path)
    # Parameter NAMES are irrelevant to matching — `{id}` == `{todo_id}`.
    path = re.sub(r':\w+', ':', path)
    return path.strip().rstrip('/')


def strip_leading_api(path: str) -> str:
    """Remove a conventional leading `api/` segment, if present."""
    if path.startswith("/api/"):
        return path[4:]
    return path


_PRIMITIVE_TYPE_SETS: dict[str, frozenset[str]] = {
    "string": frozenset({"str"}), "str": frozenset({"str"}), "varchar": frozenset({"str"}),
    "text": frozenset({"str"}), "char": frozenset({"str"}),
    "int": frozenset({"num"}), "integer": frozenset({"num"}), "number": frozenset({"num"}),
    "float": frozenset({"num"}), "double": frozenset({"num"}), "decimal": frozenset({"num"}),
    "bool": frozenset({"bool"}), "boolean": frozenset({"bool"}),
    "uuid": frozenset({"uuid"}),
    "datetime": frozenset({"date"}), "date": frozenset({"date"}), "timestamp": frozenset({"date"}),
    "time": frozenset({"date"}),
    "object": frozenset({"obj"}), "dict": frozenset({"obj"}), "record": frozenset({"obj"}),
    "any": frozenset({"obj"}), "json": frozenset({"obj"}),
    "array": frozenset({"arr"}), "list": frozenset({"arr"}), "sequence": frozenset({"arr"}),
    "none": frozenset({"none"}), "null": frozenset({"none"}),
}


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas that are not nested inside [] or ()."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current or not parts:
        parts.append("".join(current).strip())
    return parts


def _type_sets(annotation: str) -> frozenset[str]:
    """Normalize a type annotation/contract type into a set of primitive tokens.
    Handles Optional[x], Union[a, b], a | b, List[x], typing. prefixes and
    JS-style 'string|null'. Unknown names (model classes etc.) are kept as-is
    and compared literally."""
    token = (annotation or "").strip().lower().replace(" ", "")
    token = re.sub(r"\btyping\.", "", token)

    def expand(s: str) -> set[str]:
        s = s.strip()
        if not s:
            return set()
        if s.startswith("optional") and s.endswith("]"):
            return expand(s[len("optional["):-1]) | {"none"}
        if s.startswith("union") and s.endswith("]"):
            out: set[str] = set()
            for part in _split_top_level_commas(s[len("union["):-1]):
                out |= expand(part)
            return out
        if "|" in s:
            out = set()
            for part in s.split("|"):
                out |= expand(part)
            return out
        if s.startswith("list") and s.endswith("]"):
            return expand(s[len("list["):-1]) | {"arr"}
        if s.startswith("dict") and s.endswith("]"):
            return expand(s[4:s.rfind("[")]) | {"obj"}
        if s in _PRIMITIVE_TYPE_SETS:
            return set(_PRIMITIVE_TYPE_SETS[s])
        return {s}

    return frozenset(expand(token))


def _capture_annotation(code: str, field: str) -> Optional[str]:
    """Capture a `field: <annotation>` token, balancing [] so Union/Optional
    generics (which contain commas) are not truncated at depth 0."""
    m = re.search(r"(?<![\w-])\b" + re.escape(field) + r"\b\s*\??\s*:\s*", code)
    if not m:
        return None
    start = m.end()
    i = start
    depth = 0
    while i < len(code):
        ch = code[i]
        if depth == 0 and ch in ",=)#" or (depth == 0 and ch == ":" and i > start):
            break
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        i += 1
    return code[start:i].strip() or None


def check_type_consistency(field: str, expected_type: str, code: str) -> Optional[str]:
    """Check a contract field's declared type against its annotation in code.
    Comparisons are set-based with nullable tolerance: `Optional[str]`,
    `Union[str, None]`, `str | None` all satisfy a contract of `string|null`;
    a nullable actual also satisfies a non-null contract."""
    actual_type = _capture_annotation(code, field)
    if actual_type is None:
        if re.search(r"(?<![\w-])\b" + re.escape(field) + r"\b", code):
            return None
        return f"Field '{field}' is missing entirely."

    exp_set = _type_sets(expected_type)
    act_set = _type_sets(actual_type)
    if not exp_set or not act_set:
        if expected_type.lower().strip() == actual_type.lower().strip():
            return None
        return f"Type mismatch for '{field}': expected '{expected_type}', found '{actual_type}'"

    if exp_set == act_set:
        return None
    if exp_set - {"none"} == act_set - {"none"} and ({"none"} & exp_set or {"none"} & act_set):
        return None
    return f"Type mismatch for '{field}': expected '{expected_type}', found '{actual_type}'"


def _normalize_output_path(filepath: str) -> str:
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith("output/") and not rel.startswith("output\\") and rel != "output":
        return f"output/{rel}"
    return rel


def _module_path_for(src: str) -> str:
    """Layer-root-relative dotted Python module for a manifest path:
    'output/backend/db/models.py' → 'db.models'; 'output/backend/db/__init__.py' → 'db'."""
    root = layer_root_for(src)
    rel = src[len(root):] if root and src.startswith(root) else src.lstrip("/").lstrip("\\")
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")].rstrip("/")
    return rel.replace("/", ".").replace("\\", ".")


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


def imported_names_for_code(code: str, import_src: str, filepath: str) -> list[str]:
    """Names imported *from import_src* in *code* (exported-name form, alias-aware)."""
    if not code:
        return []
    parser = parse_python_imported_names if filepath.endswith(".py") else parse_js_ts_imported_names
    for mod, names, _ in parser(code):
        if mod == import_src:
            return names
    return []


def target_available_names(state: ProjectState, target_path: str) -> set[str]:
    """
    Names importable from *target_path*: manifest-promised exports + actual
    definitions found in the written code (if any). Used to gate deterministic
    import rewrites so a path guess never renames away the underlying bug.
    """
    names: set[str] = set()
    t_fm = state.code_manifest.files.get(target_path)
    if t_fm:
        names.update(str(e) for e in (t_fm.exports or []))
        t_code = get_file_code(target_path, t_fm, state)
        if t_code:
            names.update(extract_actual_definitions(t_code, target_path))
    return names


def unambiguous_import_target(
    import_src: str,
    current_file: str,
    manifest_files: list[str],
    state: ProjectState,
    threshold: float = 0.25,
) -> Optional[str]:
    """
    Deterministic import rewriter (no LLM): find the SINGLE manifest file that
    both word-matches *import_src* and can provide every name imported from it
    (promised exports or real definitions). Returns None when zero or several
    candidates qualify — ambiguity goes to rework, never a blind rewrite.
    """
    candidate = import_src.replace(".", "/").strip("/")
    wanted = []
    cur_fm = state.code_manifest.files.get(current_file)
    cur_code = get_file_code(current_file, cur_fm, state) if cur_fm else ""
    wanted = imported_names_for_code(cur_code, import_src, current_file)

    scored: list[tuple[float, str]] = []
    for p in manifest_files:
        s = word_match_score(candidate, p)
        if s >= threshold:
            scored.append((s, p))
    scored.sort(key=lambda x: (-x[0], x[1]))

    passing = [p for _, p in scored if all(w in target_available_names(state, p) for w in wanted)]
    if len(passing) == 1:
        return passing[0]
    return None


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


# ── Phase A: cross-layer semantic check helpers ────────────────────────────────

def parse_sqlalchemy_models(code: str) -> dict[str, set[str]]:
    """
    Parse SQLAlchemy ORM models: {ModelName -> {column names}}.
    Only classes inheriting from a '*Base' (declarative_base()) are considered.
    """
    models: dict[str, set[str]] = {}
    for m in re.finditer(r"^\s*class\s+(\w+)\s*\(([^)]*)\)\s*:", code, re.MULTILINE):
        name, bases = m.group(1), m.group(2)
        if "Base" not in bases:
            continue
        if any(b in bases for b in _PYDANTIC_BASES):
            continue  # pydantic BaseModel — not an ORM model
        body = _class_body_lines(code, m.end())
        cols = set(re.findall(r"\b(\w+)\s*=\s*Column\s*\(", body))
        if cols:
            models[name] = cols
    return models


def parse_pydantic_schemas(code: str) -> dict[str, set[str]]:
    """
    Parse Pydantic request/response models: {ClassName -> {field names}}.
    Subclasses of other schema classes are included even without their own
    fields (e.g. `class TodoItemCreate(TodoItemBase):` with only a docstring),
    so route -> model resolution finds the defining file for every model.
    """
    header = re.compile(r"^\s*class\s+(\w+)\s*\(([^)]*)\)\s*:", re.MULTILINE)
    decls: list[tuple[str, str, set[str]]] = []
    for m in header.finditer(code):
        name, bases = m.group(1), m.group(2)
        body = _class_body_lines(code, m.end())
        fields: set[str] = set()
        for line in body.splitlines():
            fm = re.match(r"^\s{4}(\w+)\s*:\s*(?:[^#]*)$", line) or \
                 re.match(r"^\s{4}(\w+)\s*=\s*Field\s*\(", line)
            if fm and fm.group(1) != "model_config":
                fields.add(fm.group(1))
        decls.append((name, bases, fields))
    accepted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, bases, _ in decls:
            if name in accepted:
                continue
            base_names = re.findall(r"[\w.]+", bases)
            if any(b in _PYDANTIC_BASES for b in base_names) or any(b in accepted for b in base_names):
                accepted.add(name)
                changed = True
    return {name: fields for name, _, fields in decls if name in accepted}


def _class_body_lines(code: str, header_end: int) -> str:
    """Lines after a class header until the next top-level class/def statement."""
    rest = code[header_end:].splitlines()
    block: list[str] = []
    for line in rest:
        if line and not line.startswith((" ", "\t")) and \
                re.match(r"^(class|def|async def|@)\s", line):
            break
        block.append(line)
    return "\n".join(block)


_FASTAPI_SKIP_TYPES = frozenset({
    "request", "response", "backgroundtasks", "str", "int", "bool", "float",
    "list", "dict", "any", "optional", "depends", "session", "none", "status",
    "tuple", "set", "bytes", "none",
})


def parse_backend_routes(code: str) -> list[dict]:
    """
    Parse FastAPI route declarations: [{"method", "path", "request_model",
    "response_model"}] — reads router prefixes, decorators, handler signatures.
    Commented-out decorators are ignored.
    """
    routes: list[dict] = []
    router_prefix = ""
    pm = re.search(r"\bAPIRouter\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code)
    if pm:
        router_prefix = pm.group(1).rstrip("/")

    for dm in re.finditer(
        r"@(\w+)\.(" + "|".join(_HTTP_VERBS) + r")\s*\(\s*['\"]([^'\"]+)['\"]", code
    ):
        line_start = code.rfind("\n", 0, dm.start()) + 1
        if code[line_start:dm.start()].lstrip().startswith("#"):
            continue  # commented-out decorator
        method = dm.group(2).upper()
        path = dm.group(3)
        if not path.startswith("/"):
            path = "/" + path
        full_path = (router_prefix + path) or "/"

        request_model = ""
        response_model = ""
        handler = re.search(r"\bdef\s+\w+\s*\(([^)]*)\)", code[dm.end():])
        if handler:
            for arg, anno in re.findall(r"(\w+)\s*:\s*(\w+)", handler.group(1)):
                key = anno.lower()
                if key not in _FASTAPI_SKIP_TYPES and not key.endswith("response") and "depends" not in key:
                    request_model = anno
                    break
        rm = re.search(r"response_model\s*=\s*(?:List\[(\w+)\]|(\w+))", code[dm.end():dm.end() + 600])
        if rm:
            response_model = rm.group(1) or rm.group(2)

        routes.append({
            "method": method,
            "path": full_path,
            "request_model": request_model,
            "response_model": response_model,
        })
    return routes


def _fetch_base_urls(code: str) -> dict[str, str]:
    """Discover base-URL assignments: `const API_URL = '...'`, `this.API_BASE_URL = '...'`.
    Returns {name: url} with 'this.X' keys for member assignments."""
    bases: dict[str, str] = {}
    for m in re.finditer(r"\b(?:const|let|var)\s+(\w+)\s*=\s*['\"]([^'\"]+)['\"]", code):
        if re.search(r"(api|base|url|host|endpoint)", m.group(1), re.I):
            bases[m.group(1)] = m.group(2).rstrip("/")
    for m in re.finditer(r"this\.(\w+)\s*=\s*['\"]([^'\"]+)['\"]", code):
        if re.search(r"(api|base|url|host|endpoint)", m.group(1), re.I):
            bases["this." + m.group(1)] = m.group(2).rstrip("/")
    return bases


def _resolve_fetch_segment(inner: str, bases: dict[str, str]) -> str:
    """Resolve one `${...}` template segment (or bare identifier) to a path
    fragment: known base → its url; unknown base-like → '' (dropped, the route
    tail stays); anything else → a path parameter '{id}'."""
    expr = inner.strip()
    key = expr if expr.startswith("this.") else expr.split(".")[-1] if "." in expr else expr
    if key in bases:
        return bases[key]
    member = expr[5:] if expr.startswith("this.") else expr
    if re.fullmatch(r"\w+", member) and re.search(r"(api|base|url|host|endpoint)", member, re.I):
        return ""
    return "/{id}"


def parse_frontend_fetches(code: str) -> list[dict]:
    """
    Parse frontend `fetch(...)` calls: [{"path", "method", "body_keys", "after"}].
    Resolves template-literal URL building (`${API_BASE_URL}/${id}`) and bare
    identifier args (`fetch(this.API_BASE_URL)`). Unresolvable base segments are
    dropped instead of being faked as parameters; unverifiable dynamic fetches
    yield no entry.
    """
    bases = _fetch_base_urls(code)

    fetches: list[dict] = []
    for m in re.finditer(r"\bfetch\s*\(\s*([`'\"])([^`'\"]*)\1", code):
        raw = m.group(2)
        resolved = re.sub(
            r"\$\{([^}]*)\}",
            lambda mm: _resolve_fetch_segment(mm.group(1), bases),
            raw,
        )
        resolved = re.sub(r"/{2,}", "/", resolved)
        resolved = resolved.strip().strip("'\"`")
        if not resolved:
            continue
        _append_fetch(fetches, code, m.end(), resolved)

    for m in re.finditer(r"\bfetch\s*\(\s*([A-Za-z_$][\w.$]*)\s*[,)]", code):
        key = m.group(1).strip()
        resolved = _resolve_fetch_segment(key, bases)
        if not resolved:
            continue
        _append_fetch(fetches, code, m.end(), resolved)

    # drop duplicates the two loops may have produced
    seen: set[tuple] = set()
    unique: list[dict] = []
    for f in fetches:
        k = (f["path"], f["method"])
        if k in seen:
            continue
        seen.add(k)
        unique.append(f)
    return unique


def _append_fetch(fetches: list[dict], code: str, after_start: int, path: str) -> None:
    method = "GET"
    body_keys: list[str] = []
    window = code[after_start:after_start + 500]
    mm = re.search(r"method\s*:\s*['\"]([A-Za-z]+)['\"]", window)
    if mm:
        method = mm.group(1).upper()
    sm = re.search(r"JSON\.stringify\s*\(\s*\{([^}]*)\}", window)
    if sm:
        for item in re.split(r",", sm.group(1)):
            item = item.strip()
            if not item:
                continue
            key = item.split(":")[0].strip().strip("'\"")
            if key:
                body_keys.append(key)
    fetches.append({
        "path": path,
        "method": method,
        "body_keys": body_keys,
        "after": window,
    })


def parse_html_ids(code: str) -> set[str]:
    return set(re.findall(r"\bid\s*=\s*['\"]([^'\"]+)['\"]", code))


def parse_js_used_ids(code: str) -> set[str]:
    ids = set(re.findall(r"getElementById\s*\(\s*['\"]([^'\"]+)['\"]", code))
    ids.update(re.findall(r"querySelector(?:All)?\s*\(\s*['\"]#([^'\"]+)['\"]", code))
    return ids


def parse_html_classes(code: str) -> set[str]:
    classes: set[str] = set()
    for m in re.finditer(r"class\s*=\s*['\"]([^'\"]+)['\"]", code):
        classes.update(m.group(1).split())
    return classes


def parse_js_classes(code: str) -> set[str]:
    classes: set[str] = set()
    for m in re.finditer(r"class(?:Name)?\s*=\s*['\"]([^'\"]+)['\"]", code):
        classes.update(m.group(1).split())
    for m in re.finditer(r"classList\.(?:add|toggle|remove)\(\s*['\"]([^'\"]+)['\"]", code):
        classes.add(m.group(1))
    return classes


def parse_css_classes(code: str) -> set[str]:
    """Class selectors declared in a CSS file (selector part before '{' only)."""
    classes: set[str] = set()
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("/*", "*", "//")):
            continue
        if "{" not in stripped:
            continue
        selector = stripped.split("{")[0]
        for m in re.finditer(r"\.([a-zA-Z_][\w-]*)", selector):
            classes.add(m.group(1))
    return classes


def parse_js_response_field_usage(code: str) -> dict[str, set[str]]:
    """
    Map every variable that received `await response.json()` (plus its
    forEach/map/filter loop variables) to the set of entity fields read off it.
    Used to verify frontend response-field usage against the API response schema.
    """
    resp_vars: set[str] = set(re.findall(r"\bconst\s+(\w+)\s*=\s*await\s+\w+\.json\s*\(\)", code))
    fields_by_var: dict[str, set[str]] = {v: set() for v in resp_vars}
    if not resp_vars:
        return fields_by_var

    loop_vars: dict[str, str] = {}
    for v in resp_vars:
        for m in re.finditer(
            r"\b" + re.escape(v) + r"\.(?:forEach|map|filter|find)\s*\(\s*(?:\(?(\w+)\)?\s*=>|function\s*\((\w+)\))",
            code,
        ):
            loop_vars[m.group(1) or m.group(2)] = v
        for m in re.finditer(
            r"\bfor\s*\(\s*(?:const|let|var)\s+(\w+)\s+of\s+" + re.escape(v), code
        ):
            loop_vars[m.group(1)] = v

    for v in resp_vars:
        fields_by_var[v].update(
            f for f in re.findall(r"\b" + re.escape(v) + r"\.(\w+)", code)
            if f not in _JS_MEMBER_SAFE
        )
    for lv, v in loop_vars.items():
        fields_by_var[v].update(
            f for f in re.findall(r"\b" + re.escape(lv) + r"\.(\w+)", code)
            if f not in _JS_MEMBER_SAFE
        )
    return fields_by_var


def match_backend_route(fetch_path: str, fetch_method: str, routes: list[dict],
                        allow_api_prefix: bool = False) -> tuple[Optional[dict], list[dict]]:
    """
    Match a frontend fetch call against the backend's actual route list.
    Returns (matched_route_or_None, close_candidates). Paths are normalized
    (trailing slash, {x}->:x) and matched exactly; close candidates are
    same-method routes whose path token-similarity is high. When
    allow_api_prefix is set, a conventional leading `api/` segment on either
    side is tolerated (frontend/backend often agree on `/api/...` even though
    the architecture contract spelled the path without it).
    """
    fp = normalize_api_path(fetch_path)
    fp_alt = strip_leading_api(fp) if allow_api_prefix else None
    fp_tokens = set(re.findall(r"[a-zA-Z0-9]+", fp.lower()))
    candidates = []
    for r in routes:
        rp = normalize_api_path(r["path"])
        rp_alt = strip_leading_api(rp) if allow_api_prefix else None
        if fp == rp or (fp_alt is not None and (fp_alt == rp or (rp_alt is not None and fp == rp_alt))):
            if r["method"] == fetch_method:
                return r, candidates
            candidates.append({**r, "_match": "method", "_score": 1.0})
            continue
        r_tokens = set(re.findall(r"[a-zA-Z0-9]+", rp.lower()))
        if not r_tokens:
            continue
        score = len(fp_tokens & r_tokens) / len(fp_tokens | r_tokens)
        if score >= 0.5:
            candidates.append({**r, "_match": "path", "_score": score})
    candidates.sort(key=lambda c: (c.get("_match") == "method", c.get("_score", 0.0)), reverse=True)
    return None, candidates[:3]



def _check_promised_exports(filepath: str, fm: FileManifest, code: str, conflicts: list[dict]) -> None:
    """Verify a code file's actual definitions against its promised manifest exports.
    A promised export may be satisfied by a definition OR a re-export
    (e.g. `__init__.py` doing `from .user import User`)."""
    actual_defs = extract_actual_definitions(code, filepath)
    reexport_names: set[str] = set()
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

    # Check for extra exports in code not declared in manifest exports.
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
        #    Only importable code files can export symbols — README.md,
        #    requirements.txt, .env.example, SQL and stylesheets promise
        #    "exports" but are never importable, so they are not verified.
        if filepath.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
            _check_promised_exports(filepath, fm, code, conflicts)

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
            first_seg = imp.split('.')[0] if filepath.endswith('.py') else imp.split('/')[0]
            is_internal_name = first_seg in internal_prefixes or first_seg in _LAYER_DIR_NAMES

            if (is_relative or is_internal_name) and not resolved:
                hint = (
                    f" First segment '{first_seg}' matches a conventional app-layer "
                    f"directory — create the module or rewrite the import so it points "
                    f"at an existing file."
                    if first_seg in _LAYER_DIR_NAMES else ""
                )
                conflicts.append({
                    "filepath": filepath,
                    "task_id": fm.task_id,
                    "expected": f"Resolvable module path for '{imp}'",
                    "actual": "unresolved",
                    "reason": f"Import '{imp}' in '{filepath}' cannot be resolved to any manifest file path.{hint}"
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
                        for args_str in extract_call_arg_strings(code, name):
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

        return conflicts

    def _check_contract_coverage(self, state: ProjectState) -> list[dict]:
        """
        A8 (CONTRACT_COVERAGE) — parse-based replacement for the legacy substring
        `_check_api_contracts`. Every architecture API contract must be
        implemented by a PARSED backend route (incl. StaticFiles mounts, which
        satisfy HTML-serving contracts). Field/type conformance is checked only
        against the route's owning file plus the files defining its
        request/response models — never against README/css/requirements text.
        Frontend coverage: a backend-implemented contract not called from any
        frontend fetch is flagged (static/asset routes are exempt — the browser
        references them, not fetch()).
        """
        conflicts: list[dict] = []
        contracts = state.architecture.api_contracts or []
        if not contracts:
            return conflicts

        routes, _api_prefixes = self._collect_backend_routes(state)
        schemas = self._collect_schema_fields(state)

        # Class name → defining file (schemas/ wins where duplicated).
        model_to_file: dict[str, str] = {}
        for path in state.generated_code.backend:
            code = get_file_code(path, state.code_manifest.files.get(path), state)
            if not code:
                continue
            for name in parse_pydantic_schemas(code):
                if "/schemas/" in path or name not in model_to_file:
                    model_to_file[name] = path

        # All fetch calls across written frontend files.
        frontend_fetches: list[dict] = []
        for path, fm in state.code_manifest.files.items():
            if (
                fm.agent != "frontend" or not fm.written
                or not path.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"))
            ):
                continue
            code = get_file_code(path, fm, state)
            if not code:
                continue
            for fetch in parse_frontend_fetches(code):
                frontend_fetches.append({**fetch, "filepath": path, "task_id": fm.task_id})

        for c in contracts:
            method = c.method.upper()
            contract_path = c.path.strip()
            path_key = normalize_api_path(contract_path)

            route, candidates = match_backend_route(contract_path, method, routes, allow_api_prefix=True)
            if route is None:
                # Attribute to the closest-matching route's OWNER when a
                # similar route exists (e.g. path/prefix drift): the router file
                # is the only file whose rework can fix the path. Fall back to
                # the backend ENTRYPOINT so the conflict always lands in a real
                # file's per-file rework instructions (a virtual "backend" key
                # is never delivered to any file).
                owner = "backend"
                hint = ""
                if candidates:
                    owner = candidates[0].get("filepath") or "backend"
                    hint = f" Closest route is '{candidates[0]['method']} {candidates[0]['path']}' in '{owner}' — align the path (prefix/params), don't duplicate."
                if owner == "backend" or owner not in state.code_manifest.files:
                    owner = next(
                        (p for p in state.code_manifest.files
                         if p.endswith(("main.py", "app.py", "server.py")) and "backend" in p),
                        "backend",
                    )
                conflicts.append({
                    "type": "CONTRACT_COVERAGE",
                    "filepath": owner,
                    "task_id": state.code_manifest.files.get(owner).task_id if owner in state.code_manifest.files else "unknown",
                    "expected": f"Backend implementation for endpoint '{method} {contract_path}'",
                    "actual": "Missing",
                    "reason": f"API contract endpoint '{method} {contract_path}' is not implemented in any backend file.{hint}",
                })
                continue

            owner = route.get("filepath") or "backend"
            if route.get("is_mount"):
                # An HTML static mount satisfies an HTML-serving contract — the
                # page itself is the frontend's domain, so no field checks and
                # no fetch() requirement.
                continue
            if route.get("static"):
                # File/HTML-serving routes are referenced by the browser (link/
                # script/img), not via fetch() — skip the frontend-call check.
                pass

            scope_files = set()
            if owner in state.code_manifest.files:
                scope_files.add(owner)
            for model in (route.get("request_model") or "", route.get("response_model") or ""):
                if model and model in model_to_file:
                    scope_files.add(model_to_file[model])
            scoped_code = "\n".join(
                get_file_code(p, state.code_manifest.files.get(p), state) or ""
                for p in scope_files
            )
            if scoped_code.strip():
                for field, ftype in c.request_body.items():
                    err = check_type_consistency(field, ftype, scoped_code)
                    if err:
                        conflicts.append({
                            "type": "CONTRACT_COVERAGE",
                            "filepath": owner,
                            "task_id": state.code_manifest.files.get(owner).task_id if owner in state.code_manifest.files else "unknown",
                            "expected": f"Request field '{field}' ({ftype})",
                            "actual": err,
                            "reason": f"Backend file '{owner}' for path '{contract_path}' has contract field issue: {err}.",
                        })
                for field, ftype in c.response_body.items():
                    err = check_type_consistency(field, ftype, scoped_code)
                    if err:
                        conflicts.append({
                            "type": "CONTRACT_COVERAGE",
                            "filepath": owner,
                            "task_id": state.code_manifest.files.get(owner).task_id if owner in state.code_manifest.files else "unknown",
                            "expected": f"Response field '{field}' ({ftype})",
                            "actual": err,
                            "reason": f"Backend file '{owner}' for path '{contract_path}' has contract field issue: {err}.",
                        })

            # Frontend coverage: a backend-implemented contract must be called
            # from some frontend fetch — unless it is a static/asset route.
            if route.get("static"):
                continue
            called = any(
                fetch["method"].upper() == method
                and (
                    normalize_api_path(fetch["path"]) == path_key
                    or strip_leading_api(normalize_api_path(fetch["path"])) == path_key
                    or normalize_api_path(fetch["path"]) == strip_leading_api(path_key)
                )
                for fetch in frontend_fetches
            )
            if not called:
                # Attribute to the first written frontend module so the conflict
                # is delivered through per-file rework instructions.
                frontend_target = next(
                    (f["filepath"] for f in frontend_fetches),
                    next(
                        (p for p, fm in state.code_manifest.files.items()
                         if fm.agent == "frontend" and fm.written
                         and p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"))),
                        "frontend",
                    ),
                )
                conflicts.append({
                    "type": "CONTRACT_COVERAGE",
                    "filepath": frontend_target,
                    "task_id": state.code_manifest.files.get(frontend_target).task_id if frontend_target in state.code_manifest.files else "unknown",
                    "expected": f"Frontend call/reference to endpoint '{method} {contract_path}'",
                    "actual": "Missing",
                    "reason": f"API contract endpoint '{method} {contract_path}' is implemented in backend but not called/referenced in any frontend file.",
                })

        return conflicts


    # ── Phase 1D: Cross-layer semantic checks (A1–A6) ─────────────────────────

    def _collect_backend_routes(self, state: ProjectState) -> tuple[list[dict], set[str]]:
        """
        Ground-truth backend route list. When routers are mounted under an api
        prefix (app.include_router(..., prefix=...)), only the PREFIXED routes
        are actually served — raw router paths are excluded so frontend calls
        that miss the prefix get flagged instead of silently matching.
        Returns (served_routes, api_prefixes).
        """
        api_prefixes: set[str] = set()
        for path in state.generated_code.backend:
            code = get_file_code(path, state.code_manifest.files.get(path), state)
            if not code:
                continue
            for m in re.finditer(r"include_router\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
                api_prefixes.add(m.group(1).rstrip("/"))

        raw_routes: list[dict] = []
        for path in state.generated_code.backend:
            code = get_file_code(path, state.code_manifest.files.get(path), state)
            if not code:
                continue
            static = ("FileResponse" in code or "HTMLResponse" in code or "StaticFiles" in code)
            # Only ROUTER files (those defining an APIRouter) are mounted under
            # api prefixes via include_router — app-level routes (app.get, ...)
            # are not prefixed.
            prefixed = "APIRouter" in code
            for r in parse_backend_routes(code):
                raw_routes.append({**r, "static": static, "prefixed": prefixed, "filepath": path})
            # StaticFiles mounts serve GET (html=True → the dir index too). They
            # are recorded so HTML-serving contracts (e.g. `GET /`) are satisfied
            # by the mount instead of flagged as missing.
            for m in re.finditer(
                r"\.mount\s*\(\s*['\"]([^'\"]*)['\"]\s*,\s*StaticFiles\s*\(",
                code,
            ):
                mount_path = m.group(1).rstrip("/") or "/"
                mount_html = bool(re.search(r"html\s*=\s*True", code[m.end():m.end() + 120]))
                raw_routes.append({
                    "method": "GET",
                    "path": mount_path,
                    "request_model": "",
                    "response_model": "",
                    "static": True,
                    "prefixed": False,
                    "filepath": path,
                    "is_mount": True,
                    "mount_html": mount_html,
                })

        if api_prefixes:
            routes: list[dict] = []
            for r in raw_routes:
                if r.get("prefixed"):
                    for ap in api_prefixes:
                        routes.append({**r, "path": ap + r["path"]})
                else:
                    routes.append(r)
        else:
            routes = raw_routes
        return routes, api_prefixes

    def _collect_schema_fields(self, state: ProjectState) -> dict[str, set[str]]:
        """
        {ClassName -> {fields}} across all backend files. When a class name is
        defined in both a schemas/ module and a router, the schemas/ definition
        wins (single-source rule, Phase C).
        """
        by_name: dict[str, tuple[str, set[str]]] = {}
        for path in state.generated_code.backend:
            code = get_file_code(path, state.code_manifest.files.get(path), state)
            if not code:
                continue
            for name, fields in parse_pydantic_schemas(code).items():
                cur = by_name.get(name)
                is_schema_file = "/schemas/" in path
                if cur is None or (is_schema_file and "/schemas/" not in cur[0]):
                    by_name[name] = (path, fields)
        return {name: fields for name, (_, fields) in by_name.items()}

    def _check_startup_assets(self, state: ProjectState) -> list[dict]:
        """
        A1 (STARTUP) — StaticFiles(directory=...)/FileResponse(...) paths must
        resolve from the layer root the app is run from. Catches static
        directories that point at the wrong base folder.
        """
        conflicts: list[dict] = []
        if not any("frontend" in p for p in state.code_manifest.files):
            return conflicts
        all_written = all(
            fm.written for p, fm in state.code_manifest.files.items() if "frontend" in p
        )
        if not all_written:
            return conflicts  # disk layout not final yet — avoid false positives

        for path, fm in state.code_manifest.files.items():
            if not fm.written or not path.endswith(".py"):
                continue
            root = layer_root_for(path)
            if not root:
                continue
            code = get_file_code(path, fm, state)
            if not code:
                continue
            refs: list[str] = []
            refs += [m.group(1) for m in re.finditer(
                r"StaticFiles\s*\(\s*directory\s*=\s*['\"]([^'\"]+)['\"]", code)]
            refs += [m.group(1) for m in re.finditer(
                r"FileResponse\s*\(\s*['\"]([^'\"]+)['\"]", code)]
            refs += [m.group(1) for m in re.finditer(
                r"HTMLResponse\s*\(\s*(?:content\s*=\s*)?open\s*\(\s*['\"]([^'\"]+)['\"]", code)]
            for ref in refs:
                layer_abs = os.path.join(_root_dir, root.strip("/"))
                cand = os.path.normpath(os.path.join(layer_abs, ref))
                if os.path.exists(cand):
                    continue
                alt = os.path.normpath(os.path.join(_root_dir, OUTPUT_DIR, ref.lstrip("/")))
                location = f"'{alt}'" if os.path.exists(alt) else "(nowhere in the workspace)"
                conflicts.append({
                    "type": "STARTUP",
                    "filepath": path,
                    "task_id": fm.task_id,
                    "expected": f"Asset path '{ref}' exists relative to the '{root}' layer root",
                    "actual": "missing",
                    "reason": (
                        f"'{root}{ref.lstrip('/')}' does not exist on disk. Assets appear to "
                        f"live at {location}. Use a path relative to the layer root "
                        f"(e.g. '../frontend/static')."
                    ),
                })
        return conflicts

    def _check_model_schema_consistency(self, state: ProjectState) -> list[dict]:
        """
        A2 (SCHEMA_COLUMN) — every attribute read off an ORM model and every
        constructor kwarg of a model/schema must exist on it. Catches
        `TodoItem.user_id` / `TodoResponse(completed=...)` style drift.
        """
        conflicts: list[dict] = []
        models: dict[str, set[str]] = {}
        schemas: dict[str, set[str]] = {}
        for path in state.generated_code.backend:
            code = get_file_code(path, state.code_manifest.files.get(path), state)
            if not code:
                continue
            models.update(parse_sqlalchemy_models(code))
            schemas.update(parse_pydantic_schemas(code))
        # schemas/ definitions win over router-local duplicates (single source).
        for name, fields in self._collect_schema_fields(state).items():
            schemas[name] = fields

        for path in state.generated_code.backend:
            fm = state.code_manifest.files.get(path)
            code = get_file_code(path, fm, state)
            if not code:
                continue
            for model, columns in models.items():
                for m in re.finditer(r"\b" + re.escape(model) + r"\s*\.\s*(\w+)", code):
                    attr = m.group(1)
                    if attr in columns or attr in {"query", "metadata", "select", "registry", "objects"}:
                        continue
                    conflicts.append({
                        "type": "SCHEMA_COLUMN",
                        "filepath": path,
                        "task_id": fm.task_id if fm else "unknown",
                        "expected": f"Column '{attr}' on ORM model '{model}'",
                        "actual": "undefined column",
                        "reason": (
                            f"'{path}' accesses '{model}.{attr}' but the model only defines "
                            f"columns: {sorted(columns) or '(none)'}. Fix the attribute name "
                            f"or add the column to the model."
                        ),
                    })
                for m in re.finditer(r"\b" + re.escape(model) + r"\s*\(([^)]*)\)", code):
                    for kw in re.finditer(r"(\w+)\s*=", m.group(1)):
                        key = kw.group(1)
                        if key in columns:
                            continue
                        conflicts.append({
                            "type": "SCHEMA_COLUMN",
                            "filepath": path,
                            "task_id": fm.task_id if fm else "unknown",
                            "expected": f"Column '{key}' on ORM model '{model}'",
                            "actual": "undefined column",
                            "reason": (
                                f"'{path}' constructs '{model}' with field '{key}' but the "
                                f"model only defines columns: {sorted(columns) or '(none)'}."
                            ),
                        })
            for schema, fields in schemas.items():
                for m in re.finditer(r"\b" + re.escape(schema) + r"\s*\(([^)]*)\)", code):
                    for kw in re.finditer(r"(\w+)\s*=", m.group(1)):
                        key = kw.group(1)
                        if key in fields or key in {"model_config"}:
                            continue
                        conflicts.append({
                            "type": "SCHEMA_COLUMN",
                            "filepath": path,
                            "task_id": fm.task_id if fm else "unknown",
                            "expected": f"Field '{key}' on schema '{schema}'",
                            "actual": "unknown field",
                            "reason": (
                                f"'{path}' constructs '{schema}' with field '{key}' but the "
                                f"schema only defines: {sorted(fields) or '(none)'}."
                            ),
                        })
        return conflicts

    def _check_layer_wiring(self, state: ProjectState) -> list[dict]:
        """
        A3 (WIRING) — structural integrity of the backend layer:
        no schema redefinitions next to a schemas/ module, no unimported modules
        when an entrypoint exists, and no inline data-access in routers when a
        services layer exists and is unused.
        """
        conflicts: list[dict] = []
        files = state.code_manifest.files
        backend_paths = [p for p in files if files[p].agent == "backend" and p.endswith(".py")]
        if not backend_paths:
            return conflicts

        manifest_files = list(files.keys())
        internal_prefixes = {p.split("/")[0] for p in manifest_files if p.split("/")[0]}

        # 1. Schema duplication vs a shared schemas/ module.
        schema_names: dict[str, list[str]] = {}
        for p in backend_paths:
            code = get_file_code(p, files.get(p), state)
            if not code:
                continue
            for name in parse_pydantic_schemas(code):
                schema_names.setdefault(name, []).append(p)
        for name, owners in schema_names.items():
            schema_file = next((p for p in owners if "/schemas/" in p), None)
            if not schema_file:
                continue
            for p in owners:
                if "/schemas/" in p:
                    continue
                code = get_file_code(p, files.get(p), state)
                if re.search(r"from\s+[\w.]*schemas[\w.]*\s+import", code or ""):
                    continue
                conflicts.append({
                    "type": "WIRING",
                    "filepath": p,
                    "task_id": files[p].task_id,
                    "expected": f"Schema '{name}' imported from the shared schemas module",
                    "actual": "redefined locally",
                    "reason": (
                        f"'{p}' redefines schema class '{name}' which already exists in "
                        f"'{schema_file}'. Define it once in the schemas module and import "
                        f"it here — never redefine request/response models in routers."
                    ),
                })

        # 2. Dead modules — only when the layer has an entrypoint to run.
        entrypoints = {p for p in backend_paths if os.path.basename(p) in ("main.py", "app.py", "server.py")}
        has_entrypoint = bool(entrypoints)
        imported_paths: set[str] = set()
        if has_entrypoint:
            for p in backend_paths:
                code = get_file_code(p, files.get(p), state)
                if not code:
                    continue
                for imp in parse_python_imports(code):
                    resolved = resolve_import_path(imp, p, manifest_files, internal_prefixes)
                    if resolved:
                        imported_paths.add(resolved)
            for p in backend_paths:
                if p in entrypoints or p in imported_paths:
                    continue
                if os.path.basename(p) == "__init__.py":
                    continue
                # Attribute to the entrypoint: it is the only file whose rework
                # can add the missing import (reworking the dead module itself
                # can never make it imported).
                target = next(iter(sorted(entrypoints))) if entrypoints else p
                conflicts.append({
                    "type": "WIRING",
                    "filepath": target,
                    "task_id": files[target].task_id if target in files else files[p].task_id,
                    "expected": f"Module '{p}' is imported by at least one other backend file",
                    "actual": "never imported",
                    "reason": (
                        f"'{p}' is not imported by any other backend file — it is dead "
                        f"code. Either wire it into the entrypoint/router or remove it."
                    ),
                })

        # 3. Inline data-access in routers while an unused services layer exists.
        service_paths = [p for p in backend_paths if "/services/" in p]
        if service_paths and (not has_entrypoint or not any(p in imported_paths for p in service_paths)):
            business = re.compile(
                r"\.commit\(|\.rollback\(|db\.query|SessionLocal|create_engine|sessionmaker|"
                r"mock_db|\.append\(|\.pop\(|\.save\(|\.delete\(|\.filter\("
            )
            for p in backend_paths:
                if not any(k in p.lower() for k in ("/routers/", "/routes/", "controller")):
                    continue
                code = get_file_code(p, files.get(p), state)
                if not code or ("@router." not in code and "@app." not in code):
                    continue
                if business.search(code):
                    conflicts.append({
                        "type": "WIRING",
                        "filepath": p,
                        "task_id": files[p].task_id,
                        "expected": "Routers delegate data access to the services layer",
                        "actual": "inline data access in router",
                        "reason": (
                            f"'{p}' performs data-access/business logic inline while a "
                            f"services layer exists ({', '.join(service_paths)}) that is "
                            f"never imported. Routers must be thin: import and call the "
                            f"service functions."
                        ),
                    })
        return conflicts

    def _check_router_prefixes(self, state: ProjectState) -> list[dict]:
        """
        A9 (WIRING) — a router mounted under app.include_router(prefix=...) that
        ALSO declares its own APIRouter(prefix=...) produces doubled served
        paths (e.g. '/api' + '/api/todos' → '/api/api/todos'). Flag the router
        file so rework removes one of the two prefixes; contracts can then be
        satisfied at the canonical path.
        """
        conflicts: list[dict] = []
        files = state.code_manifest.files
        include_prefixes: set[str] = set()
        for p in files:
            if files[p].agent != "backend" or not p.endswith(".py"):
                continue
            code = get_file_code(p, files.get(p), state)
            if not code:
                continue
            for m in re.finditer(r"include_router\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
                include_prefixes.add(m.group(1).rstrip("/"))
        if not include_prefixes:
            return conflicts

        for p in files:
            fm = files[p]
            if fm.agent != "backend" or not p.endswith(".py"):
                continue
            code = get_file_code(p, fm, state)
            if not code or "APIRouter" not in code:
                continue
            for m in re.finditer(r"APIRouter\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
                rp = m.group(1).rstrip("/")
                for ip in include_prefixes:
                    if rp != ip and not rp.startswith(ip + "/"):
                        continue
                    conflicts.append({
                        "type": "WIRING",
                        "filepath": p,
                        "task_id": fm.task_id,
                        "expected": f"Router mounted under include_router(prefix='{ip}') without its own prefix",
                        "actual": f"duplicate prefix: APIRouter(prefix='{rp}')",
                        "reason": (
                            f"'{p}' declares APIRouter(prefix='{rp}') while the layer mounts it via "
                            f"include_router(prefix='{ip}') — every route is served under the doubled "
                            f"path '{ip}{rp}'. Remove the router's own prefix (keep bare paths like "
                            f"'/todos') and let the include_router prefix supply the API root."
                        ),
                    })
                    break
        return conflicts

    def _check_duplicate_foundation(self, state: ProjectState) -> list[dict]:
        """
        Phase 3 (WIRING) — two+ backend files defining the same DB foundation
        symbol (engine / Base / SessionLocal) produce duplicate connections and
        two declarative bases; the import graph can never converge. Flag the
        non-canonical file so the owning agent removes the duplicate definition
        and imports the symbol from the canonical module instead.
        """
        conflicts: list[dict] = []
        files = state.code_manifest.files
        owners: dict[str, list[str]] = {s: [] for s in _FOUNDATION_SYMBOLS}
        for p in files:
            fm = files[p]
            if fm.agent != "backend" or not p.endswith(".py"):
                continue
            code = get_file_code(p, fm, state)
            if not code:
                continue
            for sym in _FOUNDATION_SYMBOLS:
                if re.search(r"^" + sym + r"\s*=", code, re.MULTILINE):
                    owners[sym].append(p)

        for sym, paths in owners.items():
            if len(paths) < 2:
                continue
            canonical = self._canonical_foundation_file(state, paths)
            mod = _module_path_for(canonical)
            for p in paths:
                if p == canonical:
                    continue
                conflicts.append({
                    "type": "WIRING",
                    "filepath": p,
                    "task_id": files[p].task_id,
                    "expected": f"'{sym}' defined once (in '{canonical}')",
                    "actual": "duplicate foundation definition",
                    "reason": (
                        f"'{p}' redefines '{sym}' which is already defined in '{canonical}'. "
                        f"DB foundation modules must define engine/Base/SessionLocal ONCE — "
                        f"remove the duplicate definition here and import it: "
                        f"'from {mod} import {sym}'."
                    ),
                })
        return conflicts

    def _canonical_foundation_file(self, state: ProjectState, paths: list[str]) -> str:
        """
        Deterministic pick of the single file that owns DB foundation symbols:
        a dedicated foundation module name first (session.py / database.py /
        db.py / base.py / init_db.py), then the file defining the most
        foundation symbols, then the lexicographically smallest path.
        """
        dedicated = {"session.py", "database.py", "db.py", "base.py", "init_db.py"}
        for p in sorted(paths):
            if os.path.basename(p).lower() in dedicated:
                return p

        def count(p: str) -> int:
            code = get_file_code(p, state.code_manifest.files.get(p), state)
            return sum(
                1 for s in _FOUNDATION_SYMBOLS
                if code and re.search(r"^" + s + r"\s*=", code, re.MULTILINE)
            )

        return max(paths, key=lambda p: (count(p), -len(p), p))

    def _check_frontend_backend_contracts(self, state: ProjectState) -> list[dict]:
        """
        A4 (FRONTEND_HTTP) — frontend fetch calls must match the backend's ACTUAL
        routes: path (incl. /api prefix), HTTP verb, request-body fields and
        response fields read by the UI.
        """
        conflicts: list[dict] = []
        frontend_paths = [
            p for p, fm in state.code_manifest.files.items()
            if fm.agent == "frontend" and p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".html"))
        ]
        if not frontend_paths:
            return conflicts
        routes, api_prefixes = self._collect_backend_routes(state)
        schemas = self._collect_schema_fields(state)

        for path in frontend_paths:
            fm = state.code_manifest.files.get(path)
            if not fm or not fm.written:
                continue
            code = get_file_code(path, fm, state)
            if not code or "fetch" not in code:
                continue
            for fetch in parse_frontend_fetches(code):
                route, close = match_backend_route(fetch["path"], fetch["method"], routes)
                prefix_issue = None
                if route is None and api_prefixes:
                    # Missing /api prefix? e.g. calls '/todos' but served at '/api/todos'.
                    # Fall through to body/static checks against the prefixed match.
                    for ap in api_prefixes:
                        match, _ = match_backend_route(ap + fetch["path"], fetch["method"], routes)
                        if match:
                            route = match
                            prefix_issue = ap
                            break
                if route is None:
                    close_str = "; ".join(f"{c['method']} {c['path']}" for c in close[:3])
                    conflicts.append({
                        "type": "FRONTEND_HTTP",
                        "filepath": path,
                        "task_id": fm.task_id,
                        "expected": f"Backend route for '{fetch['method']} {fetch['path']}'",
                        "actual": "no matching route",
                        "reason": (
                            f"Frontend '{path}' calls '{fetch['method']} {fetch['path']}' but "
                            f"no backend route matches."
                            + (f" Closest backend routes: {close_str}." if close_str else "")
                        ),
                    })
                    continue

                if prefix_issue:
                    conflicts.append({
                        "type": "FRONTEND_HTTP",
                        "filepath": path,
                        "task_id": fm.task_id,
                        "expected": f"Backend route for '{fetch['method']} {fetch['path']}'",
                        "actual": "missing api prefix",
                        "reason": (
                            f"Frontend '{path}' calls '{fetch['method']} {fetch['path']}' "
                            f"but the backend mounts its routers under '{prefix_issue}' — use "
                            f"'{prefix_issue}{fetch['path']}' instead."
                        ),
                    })

                # Static/HTML endpoint parsed as JSON in the fetch's own handling?
                if route.get("static") and ".json()" in fetch.get("after", ""):
                    conflicts.append({
                        "type": "FRONTEND_HTTP",
                        "filepath": path,
                        "task_id": fm.task_id,
                        "expected": f"JSON response from '{route['method']} {route['path']}'",
                        "actual": "endpoint returns HTML",
                        "reason": (
                            f"Frontend '{path}' parses the response of "
                            f"'{route['method']} {route['path']}' as JSON (.json()), but the "
                            f"backend endpoint serves static/HTML content."
                        ),
                    })

                # Request-body fields vs the route's request model.
                if fetch["body_keys"] and route.get("request_model"):
                    fields = schemas.get(route["request_model"])
                    if fields:
                        for key in fetch["body_keys"]:
                            if key in fields:
                                continue
                            conflicts.append({
                                "type": "FRONTEND_HTTP",
                                "filepath": path,
                                "task_id": fm.task_id,
                                "expected": f"Request field '{key}' accepted by '{route['method']} {route['path']}'",
                                "actual": "unknown field",
                                "reason": (
                                    f"Frontend sends body field '{key}' to "
                                    f"'{route['method']} {route['path']}', but request model "
                                    f"'{route['request_model']}' only accepts: {sorted(fields)}."
                                ),
                            })

                # Response fields the UI reads vs the route's response model.
                # Only checked when THIS fetch parses JSON — otherwise the used
                # fields come from an unrelated fetch and the flag is noise.
                if (
                    route.get("response_model") and schemas.get(route["response_model"])
                    and ".json()" in fetch.get("after", "")
                ):
                    resp_fields = schemas[route["response_model"]]
                    for var, used in parse_js_response_field_usage(code).items():
                        for f in used:
                            if f in resp_fields:
                                continue
                            conflicts.append({
                                "type": "FRONTEND_HTTP",
                                "filepath": path,
                                "task_id": fm.task_id,
                                "expected": f"Response field '{f}' on '{route['response_model']}'",
                                "actual": "unknown field",
                                "reason": (
                                    f"Frontend reads field '{var}.{f}' from the response of "
                                    f"'{route['method']} {route['path']}', but response model "
                                    f"'{route['response_model']}' only has: {sorted(resp_fields)}."
                                ),
                            })
        return conflicts

    def _check_frontend_dom(self, state: ProjectState) -> list[dict]:
        """
        A5 (FRONTEND_DOM) — JS element ids must exist in HTML; classes used in
        HTML/JS must be defined in CSS; every JS/CSS file must be loaded by an
        HTML entrypoint (and every referenced asset must exist).
        """
        conflicts: list[dict] = []
        files = state.code_manifest.files
        html_paths = [p for p in files if p.endswith((".html", ".htm"))]
        js_paths = [p for p in files if p.endswith((".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"))]
        css_paths = [p for p in files if p.endswith((".css", ".scss", ".sass"))]
        if not html_paths:
            return conflicts

        html_code: dict[str, str] = {}
        html_ids: set[str] = set()
        html_classes: set[str] = set()
        for p in html_paths:
            code = get_file_code(p, files.get(p), state)
            if code:
                html_code[p] = code
                html_ids |= parse_html_ids(code)
                html_classes |= parse_html_classes(code)

        # 1. JS-referenced element ids must exist.
        for p in js_paths:
            fm = files.get(p)
            code = get_file_code(p, fm, state)
            if not code:
                continue
            for i in parse_js_used_ids(code):
                if i in html_ids:
                    continue
                conflicts.append({
                    "type": "FRONTEND_DOM",
                    "filepath": p,
                    "task_id": fm.task_id if fm else "unknown",
                    "expected": f"Element with id='{i}' in an HTML file",
                    "actual": "missing",
                    "reason": (
                        f"JS '{p}' references element id '{i}' but no HTML file defines it. "
                        f"Defined ids: {sorted(html_ids) or '(none)'}."
                    ),
                })

        # 2. Classes used in HTML/JS must be defined in CSS.
        css_classes: set[str] = set()
        for p in css_paths:
            code = get_file_code(p, files.get(p), state)
            if code:
                css_classes |= parse_css_classes(code)
        if css_classes:
            used = set(html_classes)
            for p in js_paths:
                code = get_file_code(p, files.get(p), state)
                used |= parse_js_classes(code or "")
            for c in sorted(used):
                if c in css_classes:
                    continue
                # Attribute to a CSS file — the only place the class can be
                # defined. (Attributing to app.js/index.html just reworked the
                # wrong file forever.)
                css_target = css_paths[0] if css_paths else (
                    p if any(c in (html_code.get(p, "") or "") for p in html_code)
                    else (js_paths[0] if js_paths else html_paths[0])
                )
                conflicts.append({
                    "type": "FRONTEND_DOM",
                    "filepath": css_target,
                    "task_id": files.get(css_target).task_id if css_target in files else "unknown",
                    "expected": f"CSS class '.{c}'",
                    "actual": "undefined in CSS",
                    "reason": (
                        f"Class '{c}' is used in HTML/JS but never defined in any CSS "
                        f"file (defined: {sorted(css_classes)[:12] or '(none)'})."
                    ),
                })

        # 3. Asset loading: every JS/CSS file must be referenced by some HTML.
        asset_exts = (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".css", ".scss", ".sass")
        loaded: set[str] = set()
        for p in html_paths:
            code = html_code.get(p, "")
            layer = layer_root_for(p)
            for m in re.finditer(r"(?:src|href)\s*=\s*['\"]([^'\"]+)['\"]", code):
                ref = m.group(1)
                if ref.startswith(("http://", "https://", "//", "data:", "#")) or "{{" in ref:
                    continue
                if ref.startswith("/"):
                    # Web-root-relative reference — resolve against the layer root
                    # (output/frontend/ for index.html → /static/... works).
                    base_dir = layer or os.path.dirname(p)
                    ref = ref.lstrip("/")
                else:
                    base_dir = os.path.dirname(p)
                loaded.add(os.path.normpath(os.path.join(base_dir, ref)).replace("\\", "/"))
        for p in list(js_paths) + list(css_paths):
            fm = files.get(p)
            normalized = p.lstrip("./")
            if p in loaded or normalized in {x.lstrip("./") for x in loaded}:
                continue
            conflicts.append({
                "type": "FRONTEND_DOM",
                "filepath": p,
                "task_id": fm.task_id if fm else "unknown",
                "expected": f"'{p}' is loaded by an HTML entrypoint",
                "actual": "not referenced",
                "reason": (
                    f"Script/stylesheet '{p}' is never referenced by any HTML file — no "
                    f"<script src> or <link href> points at it. Wire it into the HTML "
                    f"entrypoint (and do NOT duplicate its logic inline in HTML)."
                ),
            })
        return conflicts

    def _ensure_package_init_files(self, state: ProjectState) -> list[str]:
        """
        Deterministically create missing `__init__.py` for every Python package
        directory under output/ backend layers (namespace packages are fragile).
        """
        py_dirs: set[str] = set()
        for p, fm in state.code_manifest.files.items():
            if not fm.written or not p.endswith(".py"):
                continue
            root = layer_root_for(p)
            if not root:
                continue
            d = os.path.dirname(p)
            root_dir = root.rstrip("/")
            while d and d != root_dir and len(d) > len(root_dir):
                py_dirs.add(d)
                d = os.path.dirname(d)
        applied: list[str] = []
        for d in sorted(py_dirs):
            init_path = f"{d}/__init__.py"
            if init_path in state.code_manifest.files:
                continue
            applied.extend(self._apply_repairs(state, [{
                "conflict_index": 0,
                "action": "fix",
                "rationale": "Missing package __init__.py (Phase A file-set)",
                "fix": {
                    "action": "create_file",
                    "filepath": init_path,
                    "replacement_content": "",
                },
            }], [{"task_id": "auto-fix"}]))
        if applied:
            logger.info("[IntegrationAgent] Auto-created %d package __init__.py file(s).", len(applied))
        return applied

    def _check_file_set(self, state: ProjectState) -> list[dict]:
        """
        A6 (FILE_SET) — every file the planner committed to (state.required_files)
        must be planned, written, and present on disk.
        """
        conflicts: list[dict] = []
        for req in (state.required_files or []):
            fm = state.code_manifest.files.get(req)
            if not fm:
                conflicts.append({
                    "type": "FILE_SET",
                    "filepath": req,
                    "task_id": "planner",
                    "expected": f"Required project file '{req}' is planned in the manifest",
                    "actual": "missing",
                    "reason": f"Required project file '{req}' is not planned in the code manifest.",
                })
            elif not fm.written:
                conflicts.append({
                    "type": "FILE_SET",
                    "filepath": req,
                    "task_id": fm.task_id,
                    "expected": f"Required project file '{req}' is written",
                    "actual": "planned but not written",
                    "reason": f"Required project file '{req}' is planned but was never written by its owning agent.",
                })
            else:
                abs_p = os.path.join(_root_dir, _normalize_output_path(req))
                if not os.path.exists(abs_p):
                    conflicts.append({
                        "type": "FILE_SET",
                        "filepath": req,
                        "task_id": fm.task_id,
                        "expected": f"Required project file '{req}' exists on disk",
                        "actual": "missing on disk",
                        "reason": f"Required project file '{req}' is marked written but missing on disk.",
                    })
        return conflicts

    def _check_file_corruption(self, state: ProjectState) -> list[dict]:
        """
        A7 — statically detect corrupted generated content (repeated-block
        duplication / JSON-escape leakage) in every in-memory file.
        """
        conflicts: list[dict] = []
        code_map = getattr(state, "generated_code", None)
        if code_map is None:
            return conflicts
        for layer in ("backend", "frontend", "database", "devops"):
            for filepath, code in (getattr(code_map, layer, {}) or {}).items():
                conflicts.extend(detect_corruption(code or "", filepath))
        return conflicts

    def _run_phase_a_checks(self, state: ProjectState) -> list[dict]:
        """Run all cross-layer semantic checks (A1–A7); never crash on a bad fixture."""
        checks = [
            ("startup-assets", self._check_startup_assets),
            ("model-schema", self._check_model_schema_consistency),
            ("layer-wiring", self._check_layer_wiring),
            ("router-prefixes", self._check_router_prefixes),
            ("duplicate-foundation", self._check_duplicate_foundation),
            ("frontend-backend", self._check_frontend_backend_contracts),
            ("contract-coverage", self._check_contract_coverage),
            ("frontend-dom", self._check_frontend_dom),
            ("file-set", self._check_file_set),
            ("file-corruption", self._check_file_corruption),
        ]
        conflicts: list[dict] = []
        for name, check in checks:
            try:
                conflicts.extend(check(state))
            except Exception as exc:
                logger.error("[IntegrationAgent] Phase A check '%s' failed: %s", name, exc)
        # Deduplicate identical findings (e.g. the same bad column used 4 times).
        seen: set[tuple] = set()
        unique: list[dict] = []
        for c in conflicts:
            key = (c.get("type"), c.get("filepath"), c.get("expected"), c.get("actual"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)
        return unique


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
                    # Only fix when EXACTLY ONE target both matches the path tokens
                    # AND can provide every imported name (promised exports / real
                    # definitions) — anything ambiguous goes to rework, never a
                    # blind rewrite.
                    best = unambiguous_import_target(import_src, current_file, manifest_files, state)
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

    # ── Phase 3: Deterministic wiring fixes (no LLM) ──────────────────────────

    def _fix_duplicate_foundations(self, state: ProjectState) -> list[str]:
        """
        Phase 3 — collapse duplicate DB-foundation definitions (engine / Base /
        SessionLocal) into ONE canonical module. Every other file's module-level
        definition line is replaced by a deterministic re-export import from the
        canonical module, so the symbol stays available in that file AND to any
        importer. Skipped (→ WIRING rework) when the definition is not a single
        line or when the canonical module already imports the duplicate file
        (a re-export would create a cycle).
        """
        applied: list[str] = []
        files = state.code_manifest.files
        by_sym: dict[str, dict[str, str]] = {s: {} for s in _FOUNDATION_SYMBOLS}
        for p in files:
            fm = files[p]
            if fm.agent != "backend" or not p.endswith(".py"):
                continue
            code = get_file_code(p, fm, state)
            if not code:
                continue
            for sym in _FOUNDATION_SYMBOLS:
                if re.search(r"^" + sym + r"\s*=", code, re.MULTILINE):
                    by_sym[sym][p] = code

        internal_prefixes = {p.split("/")[0] for p in files if p.split("/")[0]}
        for sym, owners in by_sym.items():
            if len(owners) < 2:
                continue
            canonical = self._canonical_foundation_file(state, list(owners))
            canon_code = owners.get(canonical) or get_file_code(canonical, files.get(canonical), state)
            canon_resolved = {
                resolve_import_path(i, canonical, list(files), internal_prefixes)
                for i in (parse_python_imports(canon_code) if canon_code else [])
            }
            mod = _module_path_for(canonical)
            for dup_path, code in owners.items():
                if dup_path == canonical:
                    continue
                # A re-export import between the two files would close a 2-cycle
                # if EITHER file already imports the other — leave for rework.
                dup_resolved = {
                    resolve_import_path(i, dup_path, list(files), internal_prefixes)
                    for i in parse_python_imports(code)
                }
                if canon_resolved and dup_path in canon_resolved:
                    continue  # canonical imports the duplicate — re-export would cycle
                if dup_resolved and canonical in dup_resolved:
                    continue  # duplicate imports canonical — re-export would cycle
                dup_syms = [
                    s for s in _FOUNDATION_SYMBOLS
                    if re.search(r"^" + s + r"\s*=", code, re.MULTILINE)
                ]
                if not dup_syms:
                    continue
                lines: dict[str, str] = {}
                ok = True
                for s in dup_syms:
                    m = re.search(r"^" + s + r"\s*=\s*[^\n]*", code, re.MULTILINE)
                    if not m or m.group(0).rstrip().endswith(("(", ",")):
                        ok = False  # multiline definition — do not half-rewrite
                        break
                    lines[s] = m.group(0)
                if not ok:
                    continue
                repairs = [
                    {
                        "conflict_index": 0,
                        "action": "fix",
                        "rationale": f"Collapse duplicate '{s}' foundation definition into a re-export from the canonical module",
                        "fix": {
                            "action": "replace_text",
                            "filepath": dup_path,
                            "target_content": lines[s],
                            "replacement_content": f"from {mod} import {s}",
                        },
                    }
                    for s in dup_syms
                ]
                applied_here = self._apply_repairs(state, repairs, [{}] * len(repairs))
                applied.extend(applied_here)
                if applied_here:
                    logger.info(
                        "[IntegrationAgent] Collapsed duplicate '%s' foundation definition(s) in %s → re-export from %s",
                        sym, dup_path, canonical,
                    )
        return applied

    def _fix_router_prefix_duplication(self, state: ProjectState) -> list[str]:
        """
        Phase 3 — strip a router's OWN APIRouter(prefix=...) when the layer
        mounts it via app.include_router(..., prefix=...) and the router prefix
        duplicates (or nests under) the mount prefix — otherwise every route is
        served under a doubled path (/api + /api/todos = /api/api/todos).
        Deterministic replace_text: removes only the prefix= argument (plus a
        dangling comma) from the call; nothing else in the file is touched.
        """
        applied: list[str] = []
        files = state.code_manifest.files
        include_prefixes: set[str] = set()
        for p in files:
            if files[p].agent != "backend" or not p.endswith(".py"):
                continue
            code = get_file_code(p, files.get(p), state)
            if not code:
                continue
            for m in re.finditer(r"include_router\s*\([^)]*?prefix\s*=\s*['\"]([^'\"]+)['\"]", code):
                include_prefixes.add(m.group(1).rstrip("/"))
        if not include_prefixes:
            return applied

        for p in files:
            fm = files[p]
            if fm.agent != "backend" or not p.endswith(".py"):
                continue
            code = get_file_code(p, fm, state)
            if not code or "APIRouter" not in code:
                continue
            for m in re.finditer(r"APIRouter\s*\(([^)]*)\)", code, re.DOTALL):
                call = m.group(0)
                pm = re.search(r"prefix\s*=\s*(['\"])([^'\"]+)\1", call)
                if not pm:
                    continue
                rp = pm.group(2).rstrip("/")
                if not any(rp == ip or rp.startswith(ip + "/") for ip in include_prefixes):
                    continue
                target = pm.group(0)
                span_start, span_end = pm.start(), pm.end()
                if span_end < len(call) and call[span_end] == ",":
                    span_end += 1
                    if span_end < len(call) and call[span_end] == " ":
                        span_end += 1
                elif span_start > 0 and call[span_start - 1] == ",":
                    span_start -= 1
                elif span_start > 1 and call[span_start - 1] == " " and call[span_start - 2] == ",":
                    span_start -= 2
                target = call[span_start:span_end]
                applied.extend(self._apply_repairs(state, [{
                    "conflict_index": 0,
                    "action": "fix",
                    "rationale": "Strip router's own prefix duplicated by include_router(prefix=...)",
                    "fix": {
                        "action": "replace_text",
                        "filepath": p,
                        "target_content": target,
                        "replacement_content": "",
                    },
                }], [{}]))
                logger.info("[IntegrationAgent] Stripped duplicated router prefix '%s' from %s", rp, p)
        return applied

    def _apply_deterministic_wiring_fixes(self, state: ProjectState) -> list[str]:
        """Phase 3 — run the deterministic duplicate-foundation + doubled-prefix
        fixes BEFORE the checks that flag them, so only ambiguous leftovers
        consume rework rounds."""
        applied: list[str] = []
        applied.extend(self._fix_duplicate_foundations(state))
        applied.extend(self._fix_router_prefix_duplication(state))
        return applied

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
                    written_code = safe_write(
                        filepath, content,
                        on_corrupt=lambda fp, c: record_corrupt_write(state, fp, c),
                    )
                    if written_code is None:
                        logger.warning("[IntegrationAgent] Refused corrupted create_file for %s", filepath)
                        continue
                    content = written_code

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
                        written_code = safe_write(
                            filepath, new_code,
                            on_corrupt=lambda fp, c: record_corrupt_write(state, fp, c),
                        )
                        if written_code is None:
                            logger.warning("[IntegrationAgent] Refused corrupted replace_text for %s", filepath)
                            continue
                        new_code = written_code

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
    def _looks_like_esm(code: str) -> bool:
        """True when the code uses top-level ESM syntax (import/export), which
        node's stdin `--check` only accepts with --input-type=module."""
        return bool(re.search(
            r"(^|\n)\s*(export\s+(default\s+)?(const|let|var|function|class|async\s+function|\{|\*)|"
            r"import\s+(type\s+)?['\"\w*{}])",
            code,
        ))

    @staticmethod
    def _syntax_check_js(code: str, node_bin: str) -> list[dict]:
        """`node --check` via stdin — pure syntax validation, no npm/node_modules/disk.
        ESM-style files (import/export) are checked as ES modules, plain files as CJS."""
        cmd = [node_bin, "--input-type=module", "--check"] if IntegrationAgent._looks_like_esm(code) else [node_bin, "--check"]
        try:
            proc = subprocess.run(
                cmd,
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
        # Phase 0 — deterministic package __init__.py creation (PEP-420 is fragile).
        self._ensure_package_init_files(ps)

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

        # Phase 3 — deterministic wiring fixes (duplicate DB foundation modules,
        # doubled router prefixes). Applied BEFORE the checks that flag them, so
        # only ambiguous leftovers consume rework rounds.
        if self._apply_deterministic_wiring_fixes(ps):
            logger.info("[IntegrationAgent] Re-checking after deterministic wiring fixes…")
            conflicts = self._run_deterministic_checks(ps)

        # Phase 1B — grounded syntax/compile smoke. Real parser errors are routed
        # straight to the owning coding agents; integration never guesses a parse fix.
        syntax_conflicts = self._run_syntax_smoke(ps)
        if syntax_conflicts:
            logger.info("[IntegrationAgent] Syntax smoke found %d real error(s).", len(syntax_conflicts))
        conflicts = conflicts + syntax_conflicts

        # Phase 1D — cross-layer semantic checks (A1–A7): startup assets, model/
        # schema columns, layer wiring, frontend↔backend routes, DOM/CSS and the
        # planner-committed file set. All of them are REWORK_ONLY — the owning
        # coding agents resolve them, integration never guesses a cross-file fix.
        phase_a_conflicts = self._run_phase_a_checks(ps)
        if phase_a_conflicts:
            logger.info("[IntegrationAgent] Phase A found %d cross-layer conflict(s).", len(phase_a_conflicts))
        conflicts = conflicts + phase_a_conflicts

        # Phase 1E — refusals recorded by the coding agents' safe-write guard
        # (corrupted content that never reached disk). Converted into
        # FILE_CORRUPTION conflicts that route straight back for REWORK. A
        # refusal is only meaningful while the file is still missing/corrupt:
        # once the file holds clean content, its old refusal records are stale
        # and must not keep the file in an endless rework loop.
        refused = list(getattr(ps, "corrupt_writes", []) or [])
        if refused:
            fresh_paths: set[str] = set()
            for fp, fm in (getattr(ps, "code_manifest", None) or {}).files.items():
                code = get_file_code(fp, fm, ps)
                if code and not detect_corruption(code, fp):
                    fresh_paths.add(fp)
            if fresh_paths:
                ps.corrupt_writes[:] = [
                    c for c in refused if _extract_refused_filepath(c) not in fresh_paths
                ]
                refused = list(ps.corrupt_writes)
        if refused:
            logger.warning("[IntegrationAgent] %d corrupted write(s) refused by coding agents.", len(refused))
            seen_refused: set[str] = set()
            for c_str in refused:
                key = (c_str or "").split("| Reason:")[0]
                if key in seen_refused:
                    continue
                seen_refused.add(key)
                conflicts.append({
                    "type": "FILE_CORRUPTION",
                    "filepath": _extract_refused_filepath(c_str),
                    "expected": "single clean copy of the file content",
                    "actual": "corrupted content refused at write time",
                    "reason": c_str,
                })

        # Phase 2 — batched LLM repair on static/contract conflicts only.
        # SYNTAX / IMPORT_NAME conflicts bypass the LLM (REWORK_ONLY_TYPES).
        # Files already past their rework cap are excluded so repair tokens are
        # never spent on files we have already given up on.
        rework_file_max = int(getattr(ps, "rework_file_max", 0) or 0)
        manifest_paths = set(ps.code_manifest.files.keys())
        over_limit: set[str] = set()
        if rework_file_max:
            over_limit = {
                f for f, n in (getattr(ps, "rework_failures", {}) or {}).items()
                if f in manifest_paths and n > rework_file_max
            }
        llm_repairable = [
            c for c in conflicts
            if c.get("type") not in REWORK_ONLY_TYPES and c.get("filepath") not in over_limit
        ]
        if llm_repairable:
            max_repair_rounds = 3
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
                conflicts = conflicts + self._run_phase_a_checks(ps)
                conflicts = conflicts + self._run_syntax_smoke(ps)

                llm_repairable = [
                    c for c in conflicts
                    if c.get("type") not in REWORK_ONLY_TYPES and c.get("filepath") not in over_limit
                ]
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
        # Files that have failed more than `rework_file_max` ROUNDS (counted once
        # per integration run, not once per conflict line) stop routing back (they
        # stay listed in `conflicts` for the run summary) so a single stubborn
        # file cannot consume the remaining global rounds.
        conflicts_by_agent: dict[str, list[str]] = {}
        rework_targets: dict[str, list[str]] = {}
        counted: set[str] = set()
        for c, c_str in zip(conflicts, conflict_strings):
            fpath = c.get("filepath", "")
            if fpath in manifest_paths:
                if fpath not in counted:
                    counted.add(fpath)
                    fails = int(ps.rework_failures.get(fpath, 0) or 0) + 1
                    ps.rework_failures[fpath] = fails
                    if rework_file_max and fails == rework_file_max + 1:
                        logger.warning(
                            "[IntegrationAgent] Giving up on %s after %d failed round(s) — left unresolved in the run summary.",
                            fpath, fails,
                        )
                if rework_file_max and int(ps.rework_failures.get(fpath, 0) or 0) > rework_file_max:
                    continue
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

    # \u2500\u2500 1b. Balanced call extraction: nested parens + multiline + def-skip \u2500\u2500
    nested = (
        "class S:\n  def f(self):\n    return TodoResponse(\n        id=str(db_todo.id),\n"
        "        title=todo.title,\n        description=todo.description,\n        completed=todo.completed,\n"
        "        created_at=todo.created_at.isoformat(),\n        updated_at=todo.updated_at.isoformat()\n    )\n"
        "def TodoResponse(x): return x\n"
        "TodoResponse(1, 2)\n"
    )
    args_list = extract_call_arg_strings(nested, "TodoResponse")
    assert len(args_list) == 2, args_list
    assert count_passed_arguments(args_list[0]) == 6, count_passed_arguments(args_list[0])
    assert count_passed_arguments(args_list[1]) == 2
    assert extract_call_arg_strings("s.log(fn(a, b), c);", "log") == ["fn(a, b), c"]

    # \u2500\u2500 1c. normalize_api_path: {x:path} catch-alls, param-name irrelevance \u2500\u2500
    assert normalize_api_path("/static/{file_path:path}") == "/static"
    assert normalize_api_path("/todos/{id}") == normalize_api_path("/todos/{todo_id}")
    assert normalize_api_path("/api/todos/{id}") != normalize_api_path("/todos")

    # \u2500\u2500 1c-bis. extract_actual_definitions: export async function, schema subclasses \u2500\u2500
    assert "fetchTodos" in extract_actual_definitions(
        "export async function fetchTodos() {\n  return await fetch('/todos');\n}\n", "x.js")
    assert "TodoItemCreate" in parse_pydantic_schemas(
        "class TodoBase(BaseModel):\n    title: str\nclass TodoItemCreate(TodoBase):\n    pass\n")

    # \u2500\u2500 1c-ter. layer-root-prefixed imports (`backend.schemas.todo`) resolve \u2500\u2500
    assert resolve_import_path(
        "backend.schemas.todo", "output/backend/api/routes/todo.py",
        ["output/backend/schemas/todo.py"], {"output"}) == "output/backend/schemas/todo.py"
    assert resolve_import_path(
        "backend.models.todo", "output/backend/api/routes/todo.py",
        ["output/backend/schemas/todo.py"], {"output"}) is None

    # ── 1d. Multiline parenthesized `from x import (\n a,\n b,\n)` parses ──
    ml = parse_python_imported_names(
        "from ..services.todo import (\n    create_todo_service,\n    get_all_todos_service,\n"
        "    TodoCreate as TC,\n    TodoItem,\n)\n"
    )
    assert ml == [("..services.todo", ["create_todo_service", "get_all_todos_service",
                                       "TodoCreate", "TodoItem"], False)], ml
    assert parse_python_imported_names("from x import (a, b)\n") == [("x", ["a", "b"], False)]
    print("[PASS] parse_python_imported_names — multiline parenthesized imports (no '(' junk)")

    # \u2500\u2500 1d. match_backend_route api-prefix tolerance (coverage-only) \u2500\u2500
    routes = [{"method": "GET", "path": "/todos", "prefixed": False, "static": False},
              {"method": "GET", "path": "/static", "static": True, "is_mount": True}]
    r, _ = match_backend_route("/api/todos", "GET", routes, allow_api_prefix=True)
    assert r is not None
    r, _ = match_backend_route("/api/todos", "GET", routes, allow_api_prefix=False)
    assert r is None, "strict matching must reject the api-prefixed path"
    r, _ = match_backend_route("/static", "GET", routes)
    assert r is not None and r.get("is_mount")
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
from fastapi import APIRouter
router = APIRouter()
@router.post("/api/v1/auth/login")
async def login_router(email: str):
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

    # ── 5a. Type-set consistency (Union / Optional / nullable tolerance) ────
    assert check_type_consistency("description", "string|null",
                                  "class Todo:\n    description: Union[str, None] = None\n") is None
    assert check_type_consistency("description", "string",
                                  "class Todo:\n    description: Optional[str] = None\n") is None
    assert check_type_consistency("description", "string|null",
                                  "class Todo:\n    description: str | None = None\n") is None
    assert check_type_consistency("description", "boolean",
                                  "class Todo:\n    description: str = ''\n") is not None
    assert check_type_consistency("count", "integer",
                                  "class Todo:\n    count: int = 0\n") is None
    assert check_type_consistency("content", "string",
                                  "body { justify-content: space-between; }\n") is not None, \
        "CSS property-like text must not satisfy a field check"
    assert check_type_consistency("missing_field", "string", "x = 1\n") is not None
    print("[PASS] check_type_consistency — Union/Optional/|, nullable tolerance, no CSS false matches")

    # ── 5a-bis. Fetch parser: base-member resolution, no {id}/{id} duplication ──
    fe = parse_frontend_fetches(
        "class A {\n  constructor() { this.API_BASE_URL = '/api/todos'; }\n"
        "  async del(id) { await fetch(`${this.API_BASE_URL}/${id}`, { method: 'DELETE' }); }\n"
        "  async list() { await fetch(this.API_BASE_URL, { method: 'GET' }); }\n"
        "  async patch(id) { await fetch(`/api/todos/${id}`, { method: 'PATCH', body: JSON.stringify({ title: 'x' }) }); }\n}\n"
    )
    paths = {(f["path"], f["method"]) for f in fe}
    assert ("/api/todos/{id}", "DELETE") in paths, paths
    assert ("/api/todos", "GET") in paths, paths
    assert ("/api/todos/{id}", "PATCH") in paths, paths
    assert not any("/{id}/{id}" in p for p, _ in paths), paths
    assert any("title" in f["body_keys"] for f in fe)
    fe_dyn = parse_frontend_fetches("fetch(`${this.UNKNOWN_BASE}/x`, { method: 'GET' });\n")
    assert fe_dyn and fe_dyn[0]["path"] == "/x", fe_dyn
    print("[PASS] parse_frontend_fetches — this.API_BASE_URL member, bare identifiers, no {id}/{id} junk")

    # ── 5b. Contract coverage: mounts satisfy HTML contracts; non-code files immune ──
    from core.state import APIContract as APIContractSchema
    state4 = ProjectState(project_id="int-test4", project_name="Int Test 4")
    fm_main = FileManifest(agent="backend", task_id="be-m", filepath="app/main.py",
                           purpose="Main", exports=["app"], written=True)
    fm_model = FileManifest(agent="backend", task_id="be-model", filepath="app/models/todo.py",
                            purpose="Todo", exports=["TodoCreate"], written=True,
                            classes=[ClassName(name="TodoCreate", arguments=[], docstring="")])
    fm_readme = FileManifest(agent="backend", task_id="be-r", filepath="app/README.md",
                             purpose="Docs", exports=["readme"], written=True)
    fm_fe = FileManifest(agent="frontend", task_id="fe-a", filepath="app/static/app.js",
                         purpose="UI", exports=[], written=True)
    state4.code_manifest = CodeManifest(files={
        "app/main.py": fm_main, "app/models/todo.py": fm_model,
        "app/README.md": fm_readme, "app/static/app.js": fm_fe,
    })
    state4.architecture = ArchData(
        tech_stack={"backend": "FastAPI", "frontend": "vanilla JS"},
        api_contracts=[
            APIContractSchema(method="GET", path="/", auth_required=False,
                              description="Serve the page", request_body={},
                              response_body={"content": "string (HTML)"}),
            APIContractSchema(method="POST", path="/todos", auth_required=False,
                              description="Create a todo", request_body={"description": "string|null"},
                              response_body={}),
            APIContractSchema(method="DELETE", path="/todos/{id}", auth_required=False,
                              description="Delete a todo", request_body={}, response_body={}),
        ],
    )
    code_main4 = """\
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("", StaticFiles(directory="app/static", html=True), name="frontend")
@app.post("/todos")
async def create_todo(todo: TodoCreate):
    return {"ok": True}
"""
    code_model4 = """\
from pydantic import BaseModel
class TodoCreate(BaseModel):
    description: Optional[str] = None
"""
    code_readme4 = "GET / returns the app page. content served as html.\n"
    code_fe4 = """\
fetch('/todos', { method: 'POST', body: JSON.stringify({ description: 'x' }) });
"""
    state4.generated_code = CodeMap(
        backend={"app/main.py": code_main4, "app/models/todo.py": code_model4, "app/README.md": code_readme4},
        frontend={"app/static/app.js": code_fe4},
    )
    agent4 = IntegrationAgent(provider=MockLLM())
    agent4.run(state4)
    missing = [c for c in state4.integration.conflicts if "DELETE /todos" in c]
    assert len(missing) == 1 and "not implemented in any backend file" in missing[0], \
        state4.integration.conflicts
    assert not any("README" in c or "app/static/app.js" in c for c in state4.integration.conflicts), \
        "README/css/js must not be scanned as backend route implementations"
    assert not any("Trailing slash" in c or "GET /" in c for c in state4.integration.conflicts), \
        "Static mount must satisfy the HTML-serving GET / contract"
    print("[PASS] _check_contract_coverage — mounts satisfy HTML contracts, non-code files immune, missing endpoint flagged")

    # Add the missing DELETE route → must merge cleanly.
    code_main4b = ("from app.models.todo import TodoCreate\n" + code_main4
                   + "@app.delete('/todos/{todo_id}') \nasync def delete_todo(todo_id: str):\n    return {'deleted': True}\n")
    state4.generated_code.backend["app/main.py"] = code_main4b
    agent4.run(state4)
    assert state4.integration.merged, state4.integration.conflicts
    print("[PASS] _check_contract_coverage — full contract set satisfied parses to clean merge")

    # ── 5c. Give-up counts rounds, not conflict lines ──────────
    state5 = ProjectState(project_id="int-test5", project_name="Int Test 5",
                          rework_file_max=2)
    fm_x = FileManifest(agent="backend", task_id="be-x", filepath="app/stubborn.py",
                        purpose="Stubborn", exports=["thing"], written=True)
    state5.code_manifest = CodeManifest(files={
        "app/stubborn.py": fm_x,
    })
    state5.generated_code = CodeMap(backend={
        "app/stubborn.py": "from app.models import A, B, C\n",
    })
    agent5 = IntegrationAgent(provider=MockLLM())
    agent5.run(state5)
    assert state5.rework_failures.get("app/stubborn.py", 0) == 1, \
        "One round with 3 conflict lines must count as ONE failed round"
    assert "app/stubborn.py" in state5.integration.rework_targets
    agent5.run(state5)  # round 2 — still under the cap (2)
    assert state5.rework_failures.get("app/stubborn.py", 0) == 2, state5.rework_failures
    assert "app/stubborn.py" in state5.integration.rework_targets
    agent5.run(state5)  # round 3 — past the cap: excluded from routing
    assert state5.rework_failures.get("app/stubborn.py", 0) == 3
    assert "app/stubborn.py" not in state5.integration.rework_targets, state5.integration.rework_targets
    assert not any("backend" in a for a in state5.integration.conflicts_by_agent), \
        state5.integration.conflicts_by_agent
    print("[PASS] Give-up counting — rounds not lines; excluded from rework at cap+1")

    # ── 4b. Layer-name imports (`api.routes`, `app.models`, ...) are internal ──
    state_layer = ProjectState(project_id="int-layer", project_name="Layer")
    fm_main = FileManifest(agent="backend", task_id="be-a", filepath="app/main.py",
                           purpose="main", exports=["app"], written=True)
    state_layer.code_manifest = CodeManifest(files={"app/main.py": fm_main})
    state_layer.generated_code = CodeMap(backend={"app/main.py": "from api.routes import router\n"})
    IntegrationAgent(provider=MockLLM()).run(state_layer)
    layer_conflicts = [c for c in state_layer.integration.conflicts if "api.routes" in c]
    assert layer_conflicts and all("cannot be resolved" in c for c in layer_conflicts), \
        state_layer.integration.conflicts
    # A real third-party import must NOT be flagged.
    state_layer.generated_code = CodeMap(backend={"app/main.py": "import requests\n"})
    IntegrationAgent(provider=MockLLM()).run(state_layer)
    assert not any("requests" in c for c in state_layer.integration.conflicts), \
        state_layer.integration.conflicts
    print("[PASS] Layer-name imports flagged unresolved; third-party imports left alone")

    # ── 4c. Doubled API prefix (include_router + APIRouter prefix) flagged ──
    state_pfx = ProjectState(project_id="int-pfx", project_name="Pfx")
    fm_main = FileManifest(agent="backend", task_id="be-main", filepath="output/backend/main.py",
                           purpose="main", exports=["app"], written=True)
    fm_rtr = FileManifest(agent="backend", task_id="be-rtr", filepath="output/backend/routers/todo.py",
                          purpose="todo router", exports=["router"], written=True)
    state_pfx.code_manifest = CodeManifest(files={"output/backend/main.py": fm_main,
                                                  "output/backend/routers/todo.py": fm_rtr})
    state_pfx.generated_code = CodeMap(backend={
        "output/backend/main.py": "app.include_router(todo_router, prefix='/api', tags=['todos'])\n",
        "output/backend/routers/todo.py":
            "router = APIRouter(prefix='/api/todos', tags=['todos'])\n"
            "@router.get('')\nasync def read_todos(): return []\n",
    })
    agent_pfx = IntegrationAgent(provider=MockLLM())
    pfx_conflicts = [c for c in agent_pfx._check_router_prefixes(state_pfx)
                     if "duplicate prefix" in str(c.get("actual", ""))]
    assert pfx_conflicts and pfx_conflicts[0]["filepath"] == "output/backend/routers/todo.py", \
        pfx_conflicts
    # Router WITHOUT its own prefix is clean.
    state_pfx.generated_code = CodeMap(backend={
        "output/backend/main.py": "app.include_router(todo_router, prefix='/api', tags=['todos'])\n",
        "output/backend/routers/todo.py":
            "router = APIRouter(tags=['todos'])\n@router.get('/todos')\nasync def read_todos(): return []\n",
    })
    assert not any("duplicate prefix" in str(c.get("actual", ""))
                   for c in agent_pfx._check_router_prefixes(state_pfx))
    print("[PASS] Doubled API prefix flagged on the router; clean prefix stays clean")

    # ── 4d. Phase 3: deterministic prefix strip fix ─────────────────────────
    state_pfx.generated_code = CodeMap(backend={
        "output/backend/main.py": "app.include_router(todo_router, prefix='/api', tags=['todos'])\n",
        "output/backend/routers/todo.py":
            "router = APIRouter(prefix='/api/todos', tags=['todos'])\n"
            "@router.get('/todos')\nasync def read_todos(): return []\n",
    })
    prefix_logs = agent_pfx._fix_router_prefix_duplication(state_pfx)
    assert prefix_logs, "deterministic prefix strip must apply a fix"
    fixed_code = state_pfx.generated_code.backend["output/backend/routers/todo.py"]
    assert "prefix='/api/todos'" not in fixed_code and "APIRouter(tags=['todos'])" in fixed_code, fixed_code
    assert "APIRouter" in fixed_code and "@router.get('/todos')" in fixed_code
    assert not any("duplicate prefix" in str(c.get("actual", ""))
                   for c in agent_pfx._check_router_prefixes(state_pfx)), \
        "after the mechanical strip the prefix check must be clean"
    print("[PASS] Phase 3 prefix fix — router's own prefix stripped mechanically, check clean")

    # ── 4e. Phase 3: duplicate DB foundation collapse ────────────────────────
    state_fnd = ProjectState(project_id="int-fnd", project_name="Fnd")
    fm_models = FileManifest(agent="backend", task_id="be-m", filepath="output/backend/db/models.py",
                             purpose="ORM models", exports=["TodoItem", "Base"], written=True)
    fm_init = FileManifest(agent="backend", task_id="be-i", filepath="output/backend/db/init_db.py",
                           purpose="engine bootstrap", exports=["engine", "SessionLocal", "Base"], written=True)
    state_fnd.code_manifest = CodeManifest(files={"output/backend/db/models.py": fm_models,
                                                  "output/backend/db/init_db.py": fm_init})
    code_models = (
        "from sqlalchemy.orm import declarative_base\n"
        "Base = declarative_base()\n"
        "engine = create_engine('sqlite:///./app.db')\n"
        "SessionLocal = sessionmaker(bind=engine)\n"
        "class TodoItem(Base):\n    __tablename__ = 'todos'\n    id = Column(Integer, primary_key=True)\n"
    )
    code_init = (
        "from sqlalchemy import create_engine\n"
        "from sqlalchemy.orm import sessionmaker, declarative_base\n"
        "engine = create_engine('sqlite:///./app.db')\n"
        "Base = declarative_base()\n"
        "SessionLocal = sessionmaker(bind=engine)\n"
        "Base.metadata.create_all(engine)\n"
    )
    state_fnd.generated_code = CodeMap(backend={
        "output/backend/db/models.py": code_models,
        "output/backend/db/init_db.py": code_init,
    })
    agent_fnd = IntegrationAgent(provider=MockLLM())
    dup_conflicts = agent_fnd._check_duplicate_foundation(state_fnd)
    assert len(dup_conflicts) >= 3, dup_conflicts  # engine/Base/SessionLocal each flagged
    canonical_targets = {c["filepath"] for c in dup_conflicts}
    assert canonical_targets == {"output/backend/db/models.py"}, \
        "models.py (non-canonical) must be the flagged file; init_db.py owns the symbols"
    # Deterministic collapse: models.py re-exports from init_db.py, no cycle.
    fnd_logs = agent_fnd._fix_duplicate_foundations(state_fnd)
    assert fnd_logs, "duplicate foundation fix must apply"
    fixed_models = state_fnd.generated_code.backend["output/backend/db/models.py"]
    assert "from db.init_db import Base" in fixed_models, fixed_models
    assert "from db.init_db import engine" in fixed_models, fixed_models
    assert "from db.init_db import SessionLocal" in fixed_models, fixed_models
    assert "class TodoItem(Base):" in fixed_models
    assert not any(
        c.get("filepath", "").startswith("output/backend/db/models.py")
        for c in agent_fnd._check_duplicate_foundation(state_fnd)
    ), "after the collapse, models.py must not be flagged anymore"
    print("[PASS] Phase 3 foundation fix — duplicates collapsed into re-exports, canonical owns symbols")

    # ── 5. Syntax smoke helpers ──────────────────────────────────────────────
    assert IntegrationAgent._syntax_check_python("def f(:\n    pass\n"), "expected a SyntaxError from broken python"
    errs = IntegrationAgent._syntax_check_python("def f(:\n    pass\n")
    assert errs and errs[0]["line"] == 1 and errs[0]["col"], errs
    assert IntegrationAgent._syntax_check_python("x = 1\n") == []
    node_bin = shutil.which("node")
    if node_bin:
        assert IntegrationAgent._syntax_check_js("const x = 1;\n", node_bin) == []
        assert IntegrationAgent._syntax_check_js("const x = ;\n", node_bin), "expected a SyntaxError from broken js"
        assert IntegrationAgent._syntax_check_js("export const x = 1;\n", node_bin) == [], \
            "ESM module syntax must parse under --input-type=module"
        assert IntegrationAgent._syntax_check_js("export const x = ;\n", node_bin), \
            "broken ESM must still be a SyntaxError"
        assert IntegrationAgent._looks_like_esm("import { x } from './x';\n")
        assert not IntegrationAgent._looks_like_esm("const x = require('x');\n")
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
