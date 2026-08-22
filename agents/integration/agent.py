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
from typing import Any, Optional

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF, deep_unescape, record_corrupt_write
except ImportError:
    from base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF, deep_unescape, record_corrupt_write

from agents.utils import detect_corruption, normalize_output_path, safe_write

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

# ── Sub-module imports (Phase 1 extraction) ────────────────────────────────────

from agents.integration.constants import (
    MAX_PARALLEL_CALLS,
    MAX_REPAIR_ROUNDS,
    OUTPUT_DIR,
    REWORK_ONLY_TYPES,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    _FASTAPI_SKIP_TYPES,
    _FOUNDATION_SYMBOLS,
    _HTTP_VERBS,
    _JS_MEMBER_SAFE,
    _LAYER_DIR_NAMES,
    _PYDANTIC_BASES,
    _PRIMITIVE_TYPE_SETS,
    _PY_KEYWORDS,
    _REFUSED_FILEPATH_RE,
)

from agents.integration.parsers import (
    parse_python_imports,
    parse_js_ts_imports,
    parse_python_imported_names,
    parse_js_ts_imported_names,
    extract_actual_definitions,
    count_passed_arguments,
    extract_call_arg_strings,
    parse_sqlalchemy_models,
    parse_pydantic_schemas,
    _class_body_lines,
    parse_backend_routes,
    parse_frontend_fetches,
    _append_fetch,
    _fetch_base_urls,
    _resolve_fetch_segment,
    parse_html_ids,
    parse_js_used_ids,
    parse_html_classes,
    parse_js_classes,
    parse_css_classes,
    parse_js_response_field_usage,
)

from agents.integration.types import (
    normalize_api_path,
    strip_leading_api,
    _split_top_level_commas,
    _type_sets,
    _capture_annotation,
    check_type_consistency,
)

from agents.integration.resolvers import (
    check_js_ts_extensions,
    layer_root_for,
    resolve_import_path,
    match_backend_route,
    best_manifest_match,
    build_internal_prefixes,
    dirname_matches,
    word_match_score,
    _tokenize,
)

from agents.integration.checks import (
    _module_path_for,
    _set_generated_code,
    canonical_foundation_file,
    get_file_code,
    run_phase_a_checks,
)

logger = logging.getLogger(__name__)


def _extract_refused_filepath(conflict_str: str) -> str:
    """Pull the filepath out of a safe-write refusal string."""
    m = _REFUSED_FILEPATH_RE.search(conflict_str or "")
    return m.group(1) if m else ""


# ── Extracted to sub-modules ──────────────────────────────────────────────────
# Parsers       → agents.integration.parsers
# Type checks   → agents.integration.types
# Resolvers     → agents.integration.resolvers
# Constants     → agents.integration.constants


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
        internal_prefixes = build_internal_prefixes(manifest_files)

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

    def _run_phase_a_checks(self, state: ProjectState) -> list[dict]:
        """Run all cross-layer semantic checks (A1–A7); never crash on a bad fixture."""
        return run_phase_a_checks(state)

    def _resolve_by_word_match(self, state: ProjectState, conflicts: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Deterministically repair mechanical conflicts before invoking the LLM.

        Returns (remaining_conflicts, applied_logs). Conflicts it can fix with
        zero ambiguity are resolved inline and removed from the returned list.
        """
        manifest = state.code_manifest
        manifest_files = list(manifest.files.keys())
        internal_prefixes = build_internal_prefixes(manifest_files)
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

        internal_prefixes = build_internal_prefixes(list(files))
        for sym, owners in by_sym.items():
            if len(owners) < 2:
                continue
            canonical = canonical_foundation_file(state, list(owners))
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

            abs_path = os.path.join(_root_dir, normalize_output_path(filepath))

            if action == "create_file":
                log = self._apply_create_file(state, filepath, agent, conflict, fix)
            elif action == "replace_text":
                log = self._apply_replace_text(state, filepath, agent, abs_path, fix)
            elif action == "update_manifest":
                log = self._apply_update_manifest(state, filepath, conflict)
            else:
                log = None

            if log:
                applied_logs.append(log)

        return applied_logs

    def _apply_create_file(
        self, state: ProjectState, filepath: str, agent: str, conflict: dict, fix: dict
    ) -> Optional[str]:
        """Mechanical repair action: write a whole new file (corruption-guarded)."""
        content = fix.get("replacement_content", "")
        try:
            written_code = safe_write(
                filepath, content,
                on_corrupt=lambda fp, c: record_corrupt_write(state, fp, c),
            )
            if written_code is None:
                logger.warning("[IntegrationAgent] Refused corrupted create_file for %s", filepath)
                return None
            content = written_code
            _set_generated_code(state, filepath, agent, content)

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

            return f"Created file: {filepath}"
        except Exception as e:
            logger.warning("[IntegrationAgent] Failed to create file %s: %s", filepath, e)
            return None

    def _apply_replace_text(
        self, state: ProjectState, filepath: str, agent: str, abs_path: str, fix: dict
    ) -> Optional[str]:
        """Mechanical repair action: replace an exact snippet inside one file."""
        target = fix.get("target_content")
        replacement = fix.get("replacement_content")
        if target is None or replacement is None:
            return None

        fm = state.code_manifest.files.get(filepath)
        code = get_file_code(filepath, fm, state) if fm else ""
        if not code and os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception:
                pass

        if target not in code:
            return None

        new_code = code.replace(target, replacement)
        try:
            written_code = safe_write(
                filepath, new_code,
                on_corrupt=lambda fp, c: record_corrupt_write(state, fp, c),
            )
            if written_code is None:
                logger.warning("[IntegrationAgent] Refused corrupted replace_text for %s", filepath)
                return None
            _set_generated_code(state, filepath, agent, written_code)
            return f"Replaced text in {filepath}"
        except Exception as e:
            logger.warning("[IntegrationAgent] Failed to write file %s: %s", filepath, e)
            return None

    def _apply_update_manifest(self, state: ProjectState, filepath: str, conflict: dict) -> Optional[str]:
        """Mechanical repair action: synchronize the manifest's promised exports."""
        expected_export = conflict.get("expected")
        fm = state.code_manifest.files.get(filepath)
        if expected_export and fm and expected_export not in fm.exports:
            fm.exports.append(expected_export)
            return f"Synchronized manifest exports for {filepath} with '{expected_export}'"
        return None

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
        self._run_phase_0_init(ps)
        conflicts = self._run_phase_1_deterministic(ps)
        conflicts = conflicts + self._run_phase_1b_syntax(ps)
        phase_a_conflicts = self._run_phase_a_checks(ps)
        if phase_a_conflicts:
            logger.info("[IntegrationAgent] Phase A found %d cross-layer conflict(s).", len(phase_a_conflicts))
        conflicts = conflicts + phase_a_conflicts
        conflicts = conflicts + self._convert_stale_refusals(ps)

        # Phase 2 — batched LLM repair on static/contract conflicts only.
        conflicts = self._run_phase_2_llm_repair(ps, conflicts)

        self._build_integration_result(ps, conflicts)

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)

        logger.info(
            "[IntegrationAgent] Done in %.2fs. Merged: %s. Remaining conflicts: %d. Rework rounds: %d/%d.",
            duration, ps.integration.merged, len(conflicts),
            ps.integration_rework_rounds, ps.max_integration_rounds,
        )
        return self._compute_diff(before, ps)

    def _run_phase_0_init(self, ps: ProjectState) -> None:
        """Phase 0 — deterministic package __init__.py creation (PEP-420 is fragile)."""
        self._ensure_package_init_files(ps)

    def _run_phase_1_deterministic(self, ps: ProjectState) -> list[dict]:
        """
        Phase 1 — deterministic checks, word-match repair, workspace-import
        normalization, and mechanical wiring fixes. Every applied fix group
        triggers a full deterministic re-check.
        """
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

        return conflicts

    def _run_phase_1b_syntax(self, ps: ProjectState) -> list[dict]:
        """Phase 1B — grounded syntax/compile smoke. Real parser errors are routed
        straight to the owning coding agents; integration never guesses a parse fix."""
        syntax_conflicts = self._run_syntax_smoke(ps)
        if syntax_conflicts:
            logger.info("[IntegrationAgent] Syntax smoke found %d real error(s).", len(syntax_conflicts))
        return syntax_conflicts

    def _convert_stale_refusals(self, ps: ProjectState) -> list[dict]:
        """
        Phase 1E — refusals recorded by the coding agents' safe-write guard
        (corrupted content that never reached disk). Converted into
        FILE_CORRUPTION conflicts that route straight back for REWORK. A
        refusal is only meaningful while the file is still missing/corrupt:
        once the file holds clean content, its old refusal records are stale
        and must not keep the file in an endless rework loop.
        """
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
        if not refused:
            return []

        logger.warning("[IntegrationAgent] %d corrupted write(s) refused by coding agents.", len(refused))
        conflicts: list[dict] = []
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
        return conflicts

    def _run_phase_2_llm_repair(self, ps: ProjectState, conflicts: list[dict]) -> list[dict]:
        """
        Phase 2 — batched LLM repair on static/contract conflicts only.
        SYNTAX / IMPORT_NAME conflicts bypass the LLM (REWORK_ONLY_TYPES).
        Files already past their rework cap are excluded so repair tokens are
        never spent on files we have already given up on.
        """
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
        for round_idx in range(MAX_REPAIR_ROUNDS):
            if not llm_repairable:
                break
            logger.info(
                "[IntegrationAgent] Repair round %d/%d — %d conflicts. Running batched LLM repair…",
                round_idx + 1, MAX_REPAIR_ROUNDS, len(llm_repairable),
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

        return conflicts

    def _build_integration_result(self, ps: ProjectState, conflicts: list[dict]) -> None:
        """Format conflict strings, group unresolved items by owning agent, and
        store the IntegrationStatus on the project state."""
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
        rework_file_max = int(getattr(ps, "rework_file_max", 0) or 0)
        manifest_paths = set(ps.code_manifest.files.keys())
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

