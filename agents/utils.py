"""
Utility functions for agents.
"""
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

OUTPUT_DIR = "output"
_root_dir = Path(__file__).resolve().parent.parent

#: Code-file extensions that are scanned for corruption (JSON/lock/data files skipped).
_CORRUPT_SCAN_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".htm", ".css"}
_CORRUPT_SKIP_NAMES = {"package-lock.json", "package.json", "yarn.lock", "poetry.lock",
                       "Pipfile.lock", "requirements.txt"}

#: Signature of JSON-escape leakage: a closing bracket glued to a quote that is
#: immediately followed by source code (e.g. `func.now())"from sqlalchemy import`).
_QUOTE_BEFORE_CODE_RE = re.compile(r"[\)\]\}]\s*\"\s*(from|import|class|def|const|var|let|function)\b")
_BACKSLASH_N_BEFORE_CODE_RE = re.compile(r"\\n\s*(from|import|class|def|const|var|let|function)\b")

_ESCAPED_CHAR_RE = re.compile(r"\\(?:[\"\\/bfnrt]|u[0-9a-fA-F]{4})")


def as_list(value: Any) -> list[str]:
    """Coerce to list[str] — handles None and non-list values gracefully."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def normalize_output_path(filepath: str) -> str:
    """Anchor a manifest filepath under the output/ directory root."""
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith("output/") and not rel.startswith("output\\") and rel != "output":
        return f"output/{rel}"
    return rel


def _is_corrupt_scannable(filepath: str) -> bool:
    """True when a file should be scanned by the corruption detector."""
    basename = os.path.basename(filepath or "").lower()
    if basename in _CORRUPT_SKIP_NAMES:
        return False
    ext = os.path.splitext(basename)[1].lower()
    return ext in _CORRUPT_SCAN_EXTENSIONS


def detect_corruption(code: str, filepath: str) -> list[dict]:
    """
    Statically detect corrupted generated content (concatenated JSON-escaped
    blobs, repeated-block duplication from echo-and-append repair behavior).

    Returns a list of conflict dicts (type FILE_CORRUPTION); empty when clean.
    Only scans code files — data/JSON/lockfiles are skipped to avoid false hits.
    """
    if not code or not _is_corrupt_scannable(filepath):
        return []
    lines = [ln.rstrip() for ln in code.splitlines()]
    if not lines:
        return []

    findings: list[dict] = []

    # 1. Repeated-block duplication: a 6-line normalized block occurring >=5
    #    times and covering >=40% of the file is a corrupted concatenation.
    window = 6
    if len(lines) >= 2 * window:
        seen: dict[str, list[int]] = {}
        for i in range(0, len(lines) - window + 1):
            block = tuple(lines[i:i + window])
            key = "\n".join(block)
            seen.setdefault(key, []).append(i)
        most_repeated = max(seen.values(), key=len)
        count = len(most_repeated)
        if count >= 5 and (count * window) >= 0.4 * len(lines):
            findings.append({
                "type": "FILE_CORRUPTION",
                "filepath": filepath,
                "expected": "single clean copy of the file content",
                "actual": f"{count} copies of the same {window}-line block (corrupted concatenation)",
                "reason": (
                    f"'{filepath}' repeats an identical {window}-line block {count} times — "
                    "content looks like concatenated/duplicated repair output, not generated code."
                ),
            })

    # 2. JSON-escape leakage: closing bracket glued to a quote before source keywords.
    for m in _QUOTE_BEFORE_CODE_RE.finditer(code):
        findings.append({
            "type": "FILE_CORRUPTION",
            "filepath": filepath,
            "expected": "plain source text",
            "actual": f"stray JSON quote before '{m.group(1)}' at char {m.start()}",
            "reason": (
                f"'{filepath}' contains a literal JSON quote glued to source code "
                f"(`...{code[max(0, m.start() - 12):m.start()]}\"...{m.group(1)}...`) — "
                "an escaped JSON string was written verbatim."
            ),
        })
        break

    # 3. Literal backslash-n before source keywords (escaped JSON written raw).
    for m in _BACKSLASH_N_BEFORE_CODE_RE.finditer(code):
        findings.append({
            "type": "FILE_CORRUPTION",
            "filepath": filepath,
            "expected": "plain source text",
            "actual": f"literal '\\n' escape before '{m.group(1)}' at char {m.start()}",
            "reason": (
                f"'{filepath}' contains a literal '\\n' escape sequence glued to source code — "
                "a JSON-escaped string was written verbatim."
            ),
        })
        break

    # 4. Truncated output: Python must parse; JS/TS must be structurally balanced.
    ext = os.path.splitext(os.path.basename(filepath or "").lower())[1]
    if ext == ".py":
        import ast
        try:
            ast.parse(code)
        except SyntaxError as exc:
            findings.append({
                "type": "FILE_CORRUPTION",
                "filepath": filepath,
                "expected": "complete, parseable Python module",
                "actual": f"truncated or invalid Python (syntax error: {exc.msg}",
                "reason": (
                    f"'{filepath}' does not parse as Python — line {exc.lineno}: {exc.msg}. "
                    "Generated content was truncated mid-definition or otherwise broken; "
                    "it is refused so the owning agent rewrites it whole."
                ),
            })
    elif ext in (".js", ".ts", ".tsx", ".jsx"):
        imbalance = _balanced_js(code)
        if imbalance:
            findings.append({
                "type": "FILE_CORRUPTION",
                "filepath": filepath,
                "expected": "complete, structurally balanced JS/TS module",
                "actual": f"truncated or unbalanced output ({imbalance})",
                "reason": (
                    f"'{filepath}' is structurally incomplete — {imbalance}. Generated content "
                    "was likely truncated mid-expression; refused so the owning agent rewrites it whole."
                ),
            })

    return findings


def _strip_trailing_junk(code: str) -> tuple[str, bool]:
    """
    Remove unambiguous trailing JSON artifacts (lone quotes glued to the end of
    otherwise-clean code). Preserves a single trailing newline when present.
    Returns (cleaned_code, changed).
    """
    cleaned = code
    changed = False
    while True:
        ws_stripped = cleaned.rstrip(" \t\r\n")
        if ws_stripped.endswith('"') and len(ws_stripped) < len(cleaned):
            cleaned = ws_stripped[:-1].rstrip(" \t")
            changed = True
            continue
        if ws_stripped.endswith('"'):
            cleaned = ws_stripped[:-1]
            changed = True
            continue
        break
    return cleaned, changed


def _balanced_js(code: str) -> str | None:
    """String-aware brace/paren/bracket balance check — catches truncated JS output."""
    stack: list[str] = []
    pairs = {"}": "{", ")": "(", "]": "["}
    i, n = 0, len(code)
    in_str: str | None = None
    in_tmpl = False
    while i < n:
        ch = code[i]
        if in_tmpl:
            if ch == "`":
                in_tmpl = False
            elif ch == "\\":
                i += 2
                continue
        elif in_str:
            if ch == in_str:
                in_str = None
            elif ch == "\\":
                i += 2
                continue
        elif ch in "\"'":
            in_str = ch
        elif ch == "`":
            in_tmpl = True
        elif ch in "{[(":
            stack.append(ch)
        elif ch in "}])":
            if not stack or stack[-1] != pairs[ch]:
                return f"unbalanced '{ch}'"
            stack.pop()
        elif ch == "/" and i + 1 < n:
            if code[i + 1] == "/":
                nl = code.find("\n", i)
                i = n if nl == -1 else nl
            elif code[i + 1] == "*":
                end = code.find("*/", i + 2)
                if end == -1:
                    return "unterminated block comment"
                i = end + 1
        i += 1
    if in_tmpl:
        return "unterminated template literal"
    if in_str:
        return "unterminated string literal"
    if stack:
        return f"unclosed '{stack[-1]}'"
    return None


def _detect_echo_duplication(abs_path: str, code: str, filepath: str) -> list[dict]:
    """
    Detect echo-and-append repair output: the previous on-disk file content
    appearing 2+ times inside the new full-file rewrite (the model pasted the
    current file and appended its "fix"). The 6-line repeated-block rule misses
    a single echo because only 2 copies exist.
    """
    if not os.path.exists(abs_path):
        return []
    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            previous = fh.read()
    except OSError:
        return []
    prev_lines = [ln.strip() for ln in previous.splitlines() if ln.strip()]
    code_lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if len(prev_lines) < 3 or len(code_lines) < 2 * len(prev_lines):
        return []
    prev_norm = "\n".join(prev_lines)
    code_norm = "\n".join(code_lines)
    count = code_norm.count(prev_norm)
    if count >= 2:
        return [{
            "type": "FILE_CORRUPTION",
            "filepath": filepath,
            "expected": "single clean copy of the file content",
            "actual": f"{count} copies of the previous file content (echo-and-append)",
            "reason": (
                f"'{filepath}' contains the previous file content {count} times — the repair "
                "output pasted the current file and appended to it instead of rewriting it once."
            ),
        }]
    return []


def safe_write(
    filepath: str,
    code: str,
    output_dir: str = OUTPUT_DIR,
    on_corrupt: Optional[Callable[[str, dict], None]] = None,
) -> Optional[str]:
    """
    Write generated code under *output_dir* through a corruption guard.

    - Unambiguous trailing JSON junk is stripped (best-effort clean).
    - Structural corruption (block duplication / escape leakage / truncation /
      echo-and-append) REFUSES the write: the previous good version on disk is
      preserved and *on_corrupt* (filepath, conflict) is invoked so the caller
      can register a FILE_CORRUPTION conflict that drives a REWORK round.

    Returns the cleaned content that was written, or None when refused.
    """
    code, _ = _strip_trailing_junk(code)

    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith(output_dir + "/") and not rel.startswith(output_dir + "\\") and rel != output_dir:
        rel = os.path.join(output_dir, rel)
    abs_path = os.path.join(_root_dir, rel)

    conflicts = detect_corruption(code, filepath)
    if not conflicts:
        conflicts = _detect_echo_duplication(abs_path, code, filepath)

    if conflicts:
        if on_corrupt is not None:
            try:
                on_corrupt(filepath, conflicts[0])
            except Exception:
                pass
        return None

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return code


def sha256_of(code: str) -> str:
    """Stable content fingerprint for run summaries."""
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def write_run_summary(state: Any) -> str:
    """
    Persist a run summary next to the generated project (`output/run_summary.json`)
    so the deliverable on disk can be diffed against the plan. Never raises.
    """
    try:
        import json
        from datetime import datetime, timezone

        files: dict[str, dict] = {}
        code_map = getattr(state, "generated_code", None)
        buckets = {}
        if code_map is not None:
            for layer in ("backend", "frontend", "database", "devops"):
                buckets.update(getattr(code_map, layer, {}) or {})

        manifest = getattr(state, "code_manifest", None)
        for fp, content in buckets.items():
            fm = (manifest.files.get(fp) if manifest is not None else None)
            entry = {
                "agent": fm.agent if fm else "",
                "task_id": fm.task_id if fm else "",
                "exports": list(fm.exports or []) if fm else [],
                "written": bool(fm.written if fm else True),
                "sha256": sha256_of(content),
                "lines": len((content or "").splitlines()),
            }
            files[fp] = entry

        integration = getattr(state, "integration", None)
        summary = {
            "project_id": getattr(state, "project_id", ""),
            "project_name": getattr(state, "project_name", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": getattr(state, "status", ""),
            "files": files,
            "integration": {
                "merged": bool(integration.merged if integration else False),
                "conflicts": list(integration.conflicts if integration else []),
                "rework_rounds": getattr(state, "integration_rework_rounds", 0),
                "max_rounds": getattr(state, "max_integration_rounds", 0),
                "rework_failures": dict(getattr(state, "rework_failures", {}) or {}),
            },
            "errors": [
                {"agent": e.get("agent", ""), "message": e.get("message", ""), "ts": str(e.get("ts", ""))}
                for e in (getattr(state, "errors", None) or [])
            ],
            "metrics": {
                "total_llm_calls": getattr(getattr(state, "metrics", None), "total_llm_calls", 0),
                "total_tokens_used": getattr(getattr(state, "metrics", None), "total_tokens_used", 0),
            },
        }

        rel = os.path.join(OUTPUT_DIR, "run_summary.json")
        abs_path = os.path.join(_root_dir, rel)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return rel
    except Exception as exc:  # pragma: no cover — summary must never break a run
        import logging
        logging.getLogger(__name__).warning("write_run_summary failed: %s", exc)
        return ""


def write_to_disk(filepath: str, code: str, output_dir: str = OUTPUT_DIR) -> None:
    """
    Safely write generated code under output_dir, preserving sub-path structure.
    """
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith(output_dir + "/") and not rel.startswith(output_dir + "\\") and rel != output_dir:
        rel = os.path.join(output_dir, rel)
    abs_path = os.path.join(_root_dir, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(code)


def parse_metadata(
    data: dict[str, Any], default_return_type: str = "None"
) -> tuple[list[Any], list[Any]]:
    """
    Extract structured ClassName and FunctionName objects from an LLM response dict.
    Handles general classes/functions as well as DB manifest column schemas.
    """
    from core.state import Argument, ClassName, FunctionName

    classes: list[ClassName] = []
    # 1. Check for database manifest schema format
    manifest_data: dict = data.get("manifest") or {}
    if isinstance(manifest_data, dict) and isinstance(manifest_data.get("table_name"), str):
        table_name = manifest_data["table_name"]
        columns = manifest_data.get("columns", [])
        args = []
        for col in columns:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name", "")
            col_type = col.get("type", "unknown")
            if not isinstance(col_name, str):
                col_name = str(col_name) if col_name is not None else ""
            if not isinstance(col_type, str):
                col_type = "unknown"
            default_val = ""
            if col.get("primary_key"):
                col_type += " (PK)"
            if col.get("foreign_key"):
                col_type += f" (FK -> {col['foreign_key']})"
            if col.get("nullable"):
                default_val = "NULL"
            args.append(Argument(name=col_name, type=col_type, default=default_val))
        classes.append(
            ClassName(
                name=table_name,
                arguments=args,
                docstring=f"Table schema for {table_name}",
            )
        )

    # 2. Standard class list
    for c in data.get("classes", []):
        try:
            args = [Argument(**a) for a in c.get("arguments", [])]
            classes.append(
                ClassName(
                    name=c["name"],
                    arguments=args,
                    docstring=c.get("docstring", ""),
                )
            )
        except Exception:
            pass

    # 3. Standard function list
    functions: list[FunctionName] = []
    for f in data.get("functions", []):
        try:
            args = [Argument(**a) for a in f.get("arguments", [])]
            functions.append(
                FunctionName(
                    name=f["name"],
                    arguments=args,
                    return_type=f.get("return_type", default_return_type),
                    docstring=f.get("docstring", ""),
                )
            )
        except Exception:
            pass

    return classes, functions


def resolve_stages(tasks: list[Any], use_priority: bool = False) -> list[list[Any]]:
    """
    Groups tasks into dependency-resolved execution stages for parallel processing.
    Tasks within the same stage share no intra-agent dependencies and are safe to run concurrently.
    If use_priority is True, tasks are grouped by priority ascending before topological sorting.
    """
    if not tasks:
        return []

    agent_ids = {t.id for t in tasks}

    if use_priority:
        priority_groups: dict[int, list[Any]] = {}
        for t in tasks:
            p = getattr(t, "priority", 0)
            priority_groups.setdefault(p, []).append(t)
        priority_keys = sorted(priority_groups.keys())
    else:
        priority_groups = {0: list(tasks)}
        priority_keys = [0]

    stages: list[list[Any]] = []
    visited: set[str] = set()

    for priority in priority_keys:
        remaining = list(priority_groups[priority])
        while remaining:
            ready: list[Any] = []
            for task in remaining:
                deps = [d for d in task.depends_on if d in agent_ids]
                if all(d in visited for d in deps):
                    ready.append(task)

            if not ready:
                # Cycle or unreachable dependencies — bundle remaining tasks to avoid deadlock
                stages.append(remaining)
                for t in remaining:
                    visited.add(t.id)
                break

            stages.append(ready)
            for t in ready:
                visited.add(t.id)
                remaining.remove(t)

    return stages


# Aliases for backward compatibility across agents
_as_list = as_list
_write_to_disk = write_to_disk
write_code_to_disk = write_to_disk
_parse_metadata = parse_metadata
