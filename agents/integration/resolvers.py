"""
agents/integration/resolvers.py
─────────────────────────────────
Import path resolution and word-match helpers extracted from
agents/integration.py (Phase 1.5).
"""

import os
import re
from typing import Optional

from agents.integration.constants import (
    OUTPUT_DIR,
    _LAYER_DIR_NAMES,
)


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
    from agents.integration.types import normalize_api_path, strip_leading_api

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


# ── Word-match deterministic resolver (Phase 1A) ───────────────────────────────

def build_internal_prefixes(manifest_files: list[str]) -> set[str]:
    """Top-level directory names from manifest paths — treated as internal import roots."""
    return {p.split("/")[0] for p in manifest_files if p.split("/")[0]}


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
        path_score = word_match_score(import_src, path)
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
