"""
The workspace module: where generated files live and how they get there.

One home for:
  - the fixed Layer roots (`BACKEND_ROOT`, `FRONTEND_ROOT`), re-rooting a planned
    path under its owning coding agent's Layer root (`reroot`), and the
    connection module's workspace path from the Persistence Contract
    (`connection_module_path`),
  - the single "put this path under `output/`" normalisation (`workspace_path`),
  - conversion between a workspace path and a dotted import (`dotted_import`,
    `module_workspace_path`) and JS/TS relative imports (`relative_import`),
  - writing generated files under an injected output root (`Workspace`),
    through the write guard that refuses corrupt code.

Agents receive a `Workspace` when they are constructed; it defaults to the real
project directory, so a run without overrides writes to `<project>/output/`.
"""
import os
import re
from pathlib import Path
from typing import Callable, Optional

#: Project directory that holds the real `output/` directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent

#: Workspace directory every generated file lives under.
OUTPUT_DIR = "output"

#: The fixed Layer roots. Every import inside a layer is relative to its Layer
#: root; database code lives under the backend's.
BACKEND_ROOT = f"{OUTPUT_DIR}/backend/"
FRONTEND_ROOT = f"{OUTPUT_DIR}/frontend/"

#: Layer root of each coding agent's files.
LAYER_ROOTS: dict[str, str] = {
    "database": BACKEND_ROOT,
    "backend": BACKEND_ROOT,
    "frontend": FRONTEND_ROOT,
}


# ── Workspace paths ──────────────────────────────────────────────────────────

def workspace_path(path: str) -> str:
    """
    Normalise a planned or generated path to its workspace form:
    forward slashes, no leading slash, anchored under `output/`.
    'backend\api\todo.py' → 'output/backend/api/todo.py'.
    """
    rel = str(path).strip().replace("\\", "/").lstrip("/")
    if rel != OUTPUT_DIR and not rel.startswith(OUTPUT_DIR + "/"):
        rel = f"{OUTPUT_DIR}/{rel}"
    return rel


def layer_root_for(path: str) -> str | None:
    """Layer root of a workspace path: its first directory under `output/`, e.g. 'output/backend/'."""
    parts = workspace_path(path).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}/"
    return None


def dotted_import(path: str) -> str:
    """
    Layer-root-relative dotted Python module for a workspace path.
    'output/backend/api/todo.py' → 'api.todo'; 'output/backend/db/__init__.py' → 'db'.
    """
    path = workspace_path(path)
    root = layer_root_for(path)
    rel = path[len(root):] if root else path
    if rel.endswith(".py"):
        rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[: -len("/__init__")].rstrip("/")
    return rel.replace("/", ".")


def module_workspace_path(dotted: str, layer_root: str) -> str:
    """
    Workspace path of a dotted Python module under *layer_root* — the inverse
    of `dotted_import` for module files. ('db.session', 'output/backend/')
    → 'output/backend/db/session.py'.
    """
    root = workspace_path(layer_root).rstrip("/") + "/"
    return f"{root}{dotted.strip().replace('.', '/')}.py"


def layer_root(agent: str) -> str:
    """Layer root of a coding agent's files: 'output/backend/' or 'output/frontend/'."""
    return LAYER_ROOTS[agent]


def reroot(path: str, agent: str) -> tuple[str, bool]:
    """
    Place a planned path under its owning coding agent's Layer root.

    - Already under the owner's Layer root (with or without `output/`): kept.
    - Starts with the other layer's directory: that directory is dropped and the
      path re-rooted under the owner; the second value is True (misplaced).
    - Otherwise the owner's Layer root is prepended.

    ('app/main.py', 'backend') → ('output/backend/app/main.py', False);
    ('frontend/db/x.py', 'database') → ('output/backend/db/x.py', True).
    """
    rel = workspace_path(path)[len(OUTPUT_DIR) + 1:]
    root = layer_root(agent)
    own_dir = root[len(OUTPUT_DIR) + 1:]
    if rel.startswith(own_dir):
        return f"{root}{rel[len(own_dir):]}", False
    for other in set(LAYER_ROOTS.values()) - {root}:
        other_dir = other[len(OUTPUT_DIR) + 1:]
        if rel.startswith(other_dir):
            return f"{root}{rel[len(other_dir):]}", True
    return f"{root}{rel}", False


def connection_module_path(contract) -> str:
    """Workspace path of the Persistence Contract's connection module, e.g.
    'output/backend/db/session.py' — it lives under the backend Layer root."""
    return f"{BACKEND_ROOT}{contract.connection_module.strip().lstrip('/')}"


def relative_import(from_file: str, to_file: str) -> str:
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


# ── Write guard (refuses corrupt generated code) ─────────────────────────────
#: Code-file extensions that are scanned for corruption (JSON/lock/data files skipped).
_CORRUPT_SCAN_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".htm", ".css"}
_CORRUPT_SKIP_NAMES = {"package-lock.json", "package.json", "yarn.lock", "poetry.lock",
                       "Pipfile.lock", "requirements.txt"}

#: Signature of JSON-escape leakage: a closing bracket glued to a quote that is
#: immediately followed by source code (e.g. `func.now())"from sqlalchemy import`).
_QUOTE_BEFORE_CODE_RE = re.compile(r"[\)\]\}]\s*\"\s*(from|import|class|def|const|var|let|function)\b")
_BACKSLASH_N_BEFORE_CODE_RE = re.compile(r"\\n\s*(from|import|class|def|const|var|let|function)\b")


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


# ── Workspace ────────────────────────────────────────────────────────────────

class Workspace:
    """
    The output root generated files are written under. Defaults to the real
    project directory; tests pass a temporary directory instead.
    """

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root is not None else PROJECT_DIR

    @property
    def output_dir(self) -> Path:
        """The workspace's `output/` directory."""
        return self.root / OUTPUT_DIR

    def abs_path(self, path: str) -> Path:
        """Absolute on-disk location of a workspace path."""
        return self.root / workspace_path(path)

    def write(
        self,
        filepath: str,
        code: str,
        on_corrupt: Optional[Callable[[str, dict], None]] = None,
    ) -> Optional[str]:
        """
        Write generated code under the output root through the corruption guard.

        - Unambiguous trailing JSON junk is stripped (best-effort clean).
        - Structural corruption (block duplication / escape leakage / truncation /
          echo-and-append) REFUSES the write: the previous good version on disk is
          preserved and *on_corrupt* (filepath, conflict) is invoked so the caller
          can register a FILE_CORRUPTION conflict that drives a REWORK round.

        Returns the cleaned content that was written, or None when refused.
        """
        code, _ = _strip_trailing_junk(code)
        abs_path = str(self.abs_path(filepath))

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

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"
