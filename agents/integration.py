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

import os
import sys
import re
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any, Optional, Union

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent
except ImportError:
    from base import BaseAgent

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


# ── Phase 2: LLM Prompt & System Prompt ────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert integration and build engineer.
Your job is to analyze a list of integration conflicts in the codebase and classify each as either "AUTO_FIX" or "HUMAN".

Classify a conflict as "AUTO_FIX" ONLY if the fix is mechanical, safe, reversible, and has zero ambiguity about intent:
  ✓ Missing __init__.py file (action: create_file, replacement_content: "")
  ✓ Import path typo (action: replace_text)
  ✓ Casing-only rename (action: replace_text)
  ✓ Trailing slash mismatch in API paths (action: replace_text)
  ✓ Export exists in code but manifest missing it (action: update_manifest)

Classify a conflict as "HUMAN" if it requires deciding business logic, behavior, or has ambiguity:
  ✗ Function signature mismatch (different number of arguments — which one is correct?)
  ✗ API contract mismatch (Backend and Frontend disagree, or request/response field names/types mismatch)
  ✗ Circular imports (fixing requires restructuring, not just path changes)
  ✗ Missing implementation/file entirely (which agent should write it?)
  ✗ Anything requiring behavioral decisions

Return a single JSON object. No explanations outside the JSON.
Schema:
{
  "classifications": [
    {
      "conflict_index": <int: index of conflict in the list>,
      "classification": "AUTO_FIX" | "HUMAN",
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

Please classify each conflict as either "AUTO_FIX" or "HUMAN". Propose a precise fix if it is AUTO_FIX.
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


def check_js_ts_extensions(base_path: str, manifest_files: list[str]) -> Optional[str]:
    for ext in ['.tsx', '.ts', '.jsx', '.js', '.css', '/index.tsx', '/index.ts', '/index.jsx', '/index.js']:
        path_with_ext = base_path + ext
        if path_with_ext in manifest_files:
            return path_with_ext
    if base_path in manifest_files:
        return base_path
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
    elif filepath.endswith(('.js', '.jsx', '.ts', '.tsx')):
        for match in re.finditer(r"\bexport\s+(?:default\s+)?(?:class|function|const|let|var)\s+([a-zA-Z0-9_]+)", code):
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  IntegrationAgent                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╗

class IntegrationAgent(BaseAgent):
    """
    Validation and mechanical repairing agent for project code.
    """

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_prompt(self, state: ProjectState) -> str:
        raise NotImplementedError("_build_prompt unused; run() is overridden.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        raise NotImplementedError("_parse_response unused; run() is overridden.")


    # ── Phase 1: Deterministic Static Checks ──────────────────────────────────

    def _check_single_file(self, filepath: str, fm: FileManifest, state: ProjectState, manifest_files: list[str], internal_prefixes: set[str]) -> list[dict]:
        conflicts = []
        code = get_file_code(filepath, fm, state)

        if not code:
            # If not written, check if it's already declared as a known gap.
            # Planner/Agents mark unwritten dependencies, but if it is completely unwritten and has exports, we don't fail here.
            # Rule 9: "Do not report missing_dependencies or suggested_additions if already declared in the owning manifest entry."
            return conflicts

        # 1. Verify actual exports vs promised exports
        actual_defs = extract_actual_definitions(code, filepath)
        for promised in fm.exports:
            if promised not in actual_defs:
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

        setattr(fm, "_resolved_imports", resolved_paths)
        return conflicts

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

        # 1. Parallel verification per-file
        lock = Lock()
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            futures: list[Future] = []
            for path, fm in manifest.files.items():
                if fm.written:
                    futures.append(pool.submit(self._check_single_file, path, fm, state, manifest_files, internal_prefixes))

            for future in futures:
                try:
                    res = future.result()
                    if res:
                        conflicts.extend(res)
                except Exception as exc:
                    logger.error("[IntegrationAgent] Verification task exception: %s", exc)

        # 2. Cycle Detection (report only, never auto-fix)
        graph = {}
        for path, fm in manifest.files.items():
            if fm.written:
                graph[path] = getattr(fm, "_resolved_imports", [])

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


    # ── Phase 2: LLM Classification & Proposal ──────────────────────────────────

    def _get_llm_classifications(self, conflicts: list[dict]) -> list[dict]:
        try:
            prompt = USER_PROMPT_TEMPLATE.format(conflicts_json=json.dumps(conflicts, indent=2))
            resp = self.llm_call(prompt)
            from agents.schemas.integration import IntegrationSchema
            data = self.extract_json_and_validate(resp.text, IntegrationSchema)
            return data.get("classifications", [])
        except Exception as exc:
            logger.error("[IntegrationAgent] Phase 2 LLM call failed: %s", exc)
            return []

    def _apply_fixes(self, state: ProjectState, classifications: list[dict], conflicts: list[dict]) -> list[str]:
        applied_logs = []
        for item in classifications:
            idx = item.get("conflict_index")
            if idx is None or idx < 0 or idx >= len(conflicts):
                continue

            conflict = conflicts[idx]
            if item.get("classification") != "AUTO_FIX":
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

    # ── Main Execution ────────────────────────────────────────────────────────

    def run(self, state: dict | ProjectState) -> dict[str, Any]:
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)
        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        logger.info("[IntegrationAgent] Starting Phase 1 checks…")
        conflicts = self._run_deterministic_checks(ps)

        if conflicts:
            logger.info("[IntegrationAgent] Phase 1 found %d conflicts. Running Phase 2 LLM call…", len(conflicts))
            classifications = self._get_llm_classifications(conflicts)

            if classifications:
                logger.info("[IntegrationAgent] Applying auto-fixes…")
                applied = self._apply_fixes(ps, classifications, conflicts)
                if applied:
                    logger.info("[IntegrationAgent] Re-running deterministic checks post-fixes…")
                    for fm in ps.code_manifest.files.values():
                        if hasattr(fm, "_resolved_imports"):
                            delattr(fm, "_resolved_imports")
                    conflicts = self._run_deterministic_checks(ps)

        for fm in ps.code_manifest.files.values():
            if hasattr(fm, "_resolved_imports"):
                delattr(fm, "_resolved_imports")

        merged = len(conflicts) == 0
        conflict_strings = [
            f"[{c.get('type', 'MISMATCH').upper()}] File: {c.get('filepath')} (task: {c.get('task_id')}) | Expected: {c.get('expected')} | Actual: {c.get('actual')} | Reason: {c.get('reason')}"
            for c in conflicts
        ]

        ps.integration = IntegrationStatus(
            merged=merged,
            conflicts=conflict_strings,
            unified_entry_point="output/frontend/index.html" if "output/frontend/index.html" in ps.code_manifest.files else ""
        )

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)

        logger.info(
            "[IntegrationAgent] Done in %.2fs. Merged: %s. Remaining conflicts: %d.",
            duration, merged, len(conflicts)
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
                "classifications": [
                    {
                        "conflict_index": 0,
                        "classification": "AUTO_FIX",
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

    assert result.integration.merged, f"Expected clean integration merge, got conflicts: {result.integration.conflicts}"
    assert "models.user" in result.generated_code.backend["backend/app/routes/auth.py"]
    print("[PASS] IntegrationAgent mock run — auto-fixed typo and re-verified successfully")
    print("=" * 60 + "\n")
