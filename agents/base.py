from providers.llm.base import LLMResponse
import time
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from providers.llm.base import BaseLLMProvider
from providers.llm.llmFactory import LLMFactory as llmFactory

from core.state import ProjectState
import os
import sys
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


from core.config import settings

DEFAULT_MODEL = settings.model_name
DEFAULT_MAX_TOKENS = settings.default_max_tokens or 8192
DEFAULT_RETRIES    = settings.default_retries or 3
DEFAULT_BACKOFF    = settings.default_backoff or 2.0

# Prompt size caps — keep the request body under typical gateway payload limits.
MAX_TREE_ENTRIES    = 80   # project file tree entries shown in coding prompts
MAX_MANIFEST_ENTRIES = 25  # dependency/manifest entries + written summaries per prompt 

_ESCAPED_CHAR_RE = re.compile(r"\\(?:[\"\\/bfnrt]|u[0-9a-fA-F]{4})")


def deep_unescape(data: Any) -> Any:
    """
    Recursively unescape values that are themselves JSON-encoded strings
    (double-encoded LLM output, e.g. a `code` field containing
    `"from sqlalchemy import ...\\n..."` written as a JSON string).

    Two shapes are decoded:
      1. values still carrying escape sequences (`\\n`, `\\"`, ...) — inner
         JSON string, decoded with json.loads;
      2. values that are quote-wrapped with raw newlines inside — the outer
         parse already turned `\\n` into real newlines, leaving a string that
         is no longer valid JSON but clearly was a wrapped code blob.

    Plain code strings such as `"use strict"` are left untouched.
    """
    if isinstance(data, dict):
        return {k: deep_unescape(v) for k, v in data.items()}
    if isinstance(data, list):
        return [deep_unescape(v) for v in data]
    if isinstance(data, str):
        s = data
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            inner = s[1:-1]
            if _ESCAPED_CHAR_RE.search(s):
                try:
                    value = json.loads(s)
                    if isinstance(value, str):
                        return deep_unescape(value)
                except Exception:
                    pass
            elif "\n" in inner or re.search(r"=|;|[{}()]", inner):
                # Already-outer-parsed wrapped string: raw newlines inside, or a
                # single line that clearly holds code (`=`, braces, semicolons).
                return deep_unescape(inner)
        return s
    return data


def record_corrupt_write(state: Any, filepath: str, conflict: dict) -> None:
    """Register a refused safe-write as a FILE_CORRUPTION conflict for the rework loop."""
    c_str = (
        f"[FILE_CORRUPTION] File: {filepath} | Expected: {conflict.get('expected')} "
        f"| Actual: {conflict.get('actual')} | Reason: {conflict.get('reason')}"
    )
    if hasattr(state, "corrupt_writes"):
        state.corrupt_writes.append(c_str)
    logger.warning("[safe-write] REFUSED corrupted content for %s: %s", filepath, conflict.get("reason"))


def agent_refuse_corrupt(state: Any, lock: Any):
    """Build a lock-safe on_corrupt callback for the safe-write guard."""
    def _refuse(filepath: str, conflict: dict) -> None:
        with lock:
            record_corrupt_write(state, filepath, conflict)
    return _refuse


####################Filters####################

def filter_manifest_by_dependencies(
    manifest: Any,
    filepath: str,
    task: Any = None,
) -> dict[str, Any]:
    """
    Filter CodeManifest to only include files that the given file/task actually depends on.
    """
    if not hasattr(manifest, "files") or not manifest.files:
        return {}

    fm = manifest.files.get(filepath)
    if not fm:
        return {}

    dep_task_ids = set(task.depends_on) if (task and hasattr(task, "depends_on") and task.depends_on) else set()

    filtered: dict[str, Any] = {}

    if dep_task_ids:
        for path, file_m in manifest.files.items():
            if path == filepath:
                continue
            if getattr(file_m, "task_id", None) in dep_task_ids or path in dep_task_ids:
                filtered[path] = file_m

    if not filtered:
        target_token = os.path.basename(filepath).split('.')[0].lower()
        for path, file_m in manifest.files.items():
            if path == filepath:
                continue
            path_token = os.path.basename(path).split('.')[0].lower()
            if getattr(file_m, "written", False) or (path_token and (path_token in target_token or target_token in path_token)):
                filtered[path] = file_m

    return filtered


def filter_api_contracts_by_relevance(
    contracts: list[Any],
    filepath: str,
    purpose: str = "",
    task_title: str = "",
    top_k: int = 3,
) -> list[Any]:
    """Filter and rank API contracts by relevance score using Jaccard keyword similarity."""
    
    if not contracts:
        return []
    
    if len(contracts) <= top_k:
        return contracts
    
    search_words = set(re.findall(r"[a-zA-Z0-9]+", f"{filepath} {purpose} {task_title}".lower()))
    if not search_words:
        return contracts[:top_k]
    
    def score_contract(contract: Any) -> float:
        if isinstance(contract, dict):
            c_text = f"{contract.get('path', '')} {contract.get('description', '')}"
        else:
            c_text = f"{getattr(contract, 'path', '')} {getattr(contract, 'description', '')}"
        
        c_words = set(re.findall(r"[a-zA-Z0-9]+", c_text.lower()))
        if not c_words:
            return 0.0
        
        intersection = len(search_words & c_words)
        if not intersection:
            return 0.0
        return intersection / len(search_words | c_words)
    
    # Compute score once per contract
    scored = [(score_contract(c), c) for c in contracts]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Return top K with score > threshold
    threshold = 0.05
    filtered = [c for score, c in scored if score >= threshold]
    
    return filtered[:top_k] if filtered else [c for _, c in scored[:top_k]]


# ── Trivial-file classifier ───────────────────────────────────────────────────

#: Filename patterns that are almost always trivial (< 20 lines, no bespoke logic)
_TRIVIAL_NAMES = frozenset({
    "__init__.py",
    "index.ts", "index.tsx", "index.js", "index.jsx",  # barrel / re-export files
    "types.ts", "types.tsx",                            # pure type declarations
    "constants.ts", "constants.js",                     # constant modules
    "config.py", "config.ts", "config.js",              # app config stubs
    "settings.py",
    "exceptions.py", "errors.py", "errors.ts",          # exception/error stubs
    "utils.py",                                         # small utility stubs
})

_TRIVIAL_PATTERNS = (
    "/__init__.py",
    "/index.",          # barrel files under any directory
    "/types.",
    "/constants.",
    "/exceptions.",
    "/errors.",
    "/config.",
    "/settings.",
)

TRIVIAL_BATCH_SIZE = 3  # max files per batch LLM call

#: Entrypoint/scaffold files get a dedicated single-file prompt (Problem 2) —
#: they must NEVER be batched as trivial because their wiring context is special.
_SCAFFOLD_NAMES = frozenset({"main.py", "app.py", "server.py", "index.html", "index.htm"})


def is_scaffold_file(filepath: str | None) -> bool:
    return os.path.basename(filepath or "").lower() in _SCAFFOLD_NAMES


_BARREL_NAMES = frozenset({
    "__init__.py",
    "index.ts", "index.tsx", "index.js", "index.jsx",  # barrel / re-export files
})


def is_barrel_file(filepath: str | None) -> bool:
    """
    True for package/barrel re-export files (`__init__.py`, `index.ts|js|tsx|jsx`).

    These files do the actual WIRING between modules (re-exports), so they are
    synthesized deterministically from the manifest — never sent to the LLM.
    """
    base = os.path.basename(filepath or "").lower()
    return base in _BARREL_NAMES


def _layer_root_for(filepath: str) -> str | None:
    """Layer root dir of a workspace file, e.g. 'output/backend/' (mirrors integration.layer_root_for)."""
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


def build_import_contract(manifest: Any, filepath: str) -> str:
    """
    Deterministic IMPORT CONTRACT for *filepath*: the exact import lines it may
    use, computed from the Planner's import graph (imports_from) + manifest
    exports. Never guesses — every source must exist in the manifest.

    Returns an empty string when the file declares no internal imports.
    """
    if not hasattr(manifest, "files"):
        return ""
    fm = manifest.files.get(filepath)
    if fm is None:
        return ""
    imports_from = getattr(fm, "imports_from", {}) or {}

    lines: list[str] = []
    missing: list[str] = []
    for src_raw, names in imports_from.items():
        src = _normalized_src(src_raw)
        if src not in manifest.files or src == filepath:
            if src not in manifest.files:
                missing.append(src)
            continue
        names = [str(n) for n in names if str(n).strip()]
        if filepath.endswith(".py"):
            mod = _module_path_for(src)
            if names:
                lines.append(f"from {mod} import {', '.join(names)}")
            else:
                lines.append(f"import {mod}")
        else:
            rel = _rel_import_path(filepath, src)
            if names:
                lines.append(f"import {{ {', '.join(names)} }} from '{rel}'")
            else:
                lines.append(f"import '{rel}'")

    if not lines:
        return ""

    header = (
        "## IMPORT CONTRACT (deterministic — use EXACTLY these imports)\n"
        "Every line below is derived from the planner's import graph. Do NOT invent "
        "other internal imports, modules, or paths.\n"
    )
    body = "\n".join(f"- `{l}`" for l in lines)
    note = "\nUse exactly these import statements in your code (same modules, same names)."
    if missing:
        note += (
            "\nNote: the following planned sources do not exist in the manifest and "
            "were omitted: " + ", ".join(missing)
        )
    return header + body + note


def build_barrel_content(filepath: str, manifest: Any) -> str:
    """
    Deterministic re-export content for a barrel file (`__init__.py` / `index.*`).
    No LLM call — wiring-critical files must not be guessed.

    Re-exports come from the Planner's import graph first; when the planner gave
    no graph, falls back to re-exporting same-directory sibling exports.
    """
    if not hasattr(manifest, "files"):
        return ""
    fm = manifest.files.get(filepath)
    if fm is None:
        return ""
    imports_from = getattr(fm, "imports_from", {}) or {}

    sources: list[tuple[str, list[str]]] = []
    for src_raw, names in imports_from.items():
        src = _normalized_src(src_raw)
        if src not in manifest.files or src == filepath:
            continue
        sources.append((src, [str(n) for n in names if str(n).strip()]))

    if not sources:
        sibling_dir = os.path.dirname(filepath)
        for src, sfm in sorted(manifest.files.items()):
            if src == filepath or os.path.dirname(src) != sibling_dir:
                continue
            ex = [e for e in (getattr(sfm, "exports", None) or []) if str(e).strip()]
            if ex:
                sources.append((src, list(ex)))

    if not sources:
        return ""

    if filepath.endswith("__init__.py"):
        lines: list[str] = []
        for src, names in sources:
            rel = _module_path_for(src)
            if names:
                lines.append(f"from {rel} import {', '.join(names)}")
        return "\n".join(lines) + ("\n" if lines else "")

    lines = []
    for src, names in sources:
        rel = _rel_import_path(filepath, src)
        if names:
            lines.append(f"export {{ {', '.join(names)} }} from '{rel}'")
        else:
            lines.append(f"export * from '{rel}'")
    return "\n".join(lines) + "\n"


def align_batch_items(items: list[dict], tasks: list[Any], manifest: Any) -> list[tuple[dict, Any]] | None:
    """
    Pair batch LLM response items with their tasks.

    Prefers filepath-based matching (the LLM echoes each filepath), falls back to
    order-based pairing. Returns None on a count mismatch so the caller can fall
    back to per-file generation instead of silently misassigning code.
    """
    if not items or len(items) != len(tasks):
        return None

    by_path: dict[str, dict] = {}
    for it in items:
        fp = str(it.get("filepath", "") or "").strip()
        if fp:
            by_path[fp] = it

    if len(by_path) == len(items) and hasattr(manifest, "by_task"):
        aligned: list[tuple[dict, Any]] = []
        for t in tasks:
            fm = manifest.by_task(t.id)
            fp = getattr(fm, "filepath", None)
            it = by_path.get(fp)
            if it is None:
                return None
            aligned.append((it, t))
        return aligned

    return list(zip(items, tasks))


def build_rework_instructions(integration: Any, filepath: str) -> str:
    """
    Build a prompt section for *filepath* describing integration conflicts that
    a coding agent must resolve when it regenerates the file (REWORK loop).

    Returns an empty string when no rework is required for this file.
    """
    targets = {}
    if integration is not None:
        targets = getattr(integration, "rework_targets", {}) or {}
        if isinstance(targets, dict) is False and hasattr(targets, "model_dump"):
            targets = targets.model_dump()

    notes = targets.get(filepath) if isinstance(targets, dict) else None
    if not notes:
        return ""

    lines = "\n".join(f"- {n}" for n in notes)
    return (
        "\n## INTEGRATION REWORK REQUIRED\n"
        "The previous version of this file failed integration checks. "
        "Fix the following issue(s) in this rewrite:\n\n"
        f"{lines}\n"
    )


def build_rework_prompt(
    filepath: str,
    current_code: str,
    rework_instructions: str,
    output_key: str = "code",
) -> str:
    """
    Build a lean, targeted prompt for a REWORK round.

    Unlike the full generation prompt, this sends ONLY the conflict notes and the
    current file content, and asks the LLM to return the full corrected file. It is
    cheaper and minimizes the risk of the LLM regressing unrelated parts of the file.
    """
    return (
        f"{rework_instructions.strip()}\n\n"
        f"Here is the CURRENT content of `{filepath}`:\n\n"
        f"```\n{current_code}\n```\n\n"
        "Rewrite ONLY what is necessary to fix the issue(s) above. Keep everything "
        "else identical. Return a single valid JSON object with the full corrected "
        f"file content under the \"{output_key}\" key. No prose outside the JSON.\n\n"
        "CRITICAL — the \"code\" value MUST contain the COMPLETE corrected file as "
        "plain source text, exactly one copy. Never quote or JSON-escape it (no \\n "
        "sequences, no wrapped quotes). Do NOT paste the CURRENT content shown above "
        "into the value as a prefix and append changes — the value REPLACES the entire "
        "file. If no change is actually needed, return the current content as-is, "
        "exactly one copy."
    )


def build_rework_batch_prompt(
    entries: list[tuple[str, str, str]],
    output_key: str = "code",
) -> str:
    """
    Build a lean, targeted prompt for a REWORK batch of multiple files.

    *entries* is a list of (filepath, current_code, rework_instructions) tuples.
    Returns a prompt asking for a JSON ARRAY of corrected files (one per entry,
    same order). Used to batch small/trivial rework files into one LLM call.
    """
    blocks = []
    for i, (fp, current_code, instr) in enumerate(entries, start=1):
        blocks.append(
            f"### File {i}: {fp}\n"
            f"{instr.strip()}\n\n"
            f"CURRENT content of `{fp}`:\n"
            f"```\n{current_code}\n```"
        )
    joined = "\n\n".join(blocks)
    return (
        f"{joined}\n\n"
        "For EACH file above, fix ONLY the issue(s) listed and return the full "
        "corrected file content. Do not add new files or reorder. "
        f"Respond with a single JSON ARRAY (one object per file, in the same order):\n"
        f'[{{"filepath":"<path>","{output_key}":"<full corrected source>"}}, ...]\n\n'
        "CRITICAL — each \"code\" value MUST contain the COMPLETE corrected file as "
        "plain source text, exactly one copy. Never quote or JSON-escape it (no \\n "
        "sequences, no wrapped quotes). Do NOT paste the CURRENT content shown above "
        "into a value as a prefix and append changes — each value REPLACES the entire "
        "file. If no change is needed for a file, return its content as-is, exactly once."
    )


def is_trivial_filepath(filepath: str, exports: list | None = None) -> bool:
    """True if a file is 'trivial' boilerplate (init, barrel, config, types, constants, errors, <3 exports)."""
    basename = os.path.basename(filepath or "").lower()
    if basename in _TRIVIAL_NAMES:
        return True
    for pattern in _TRIVIAL_PATTERNS:
        if pattern in (filepath or "").lower():
            return True
    if exports is not None and len(exports) < 3:
        return True
    return False


def is_trivial_file(task: Any, manifest: Any) -> bool:
    """
    Returns True if a task maps to a file that is "trivial":
      - A known boilerplate pattern (init, barrel, config, types, constants, errors)
      - OR fewer than 3 planned exports in the manifest
    Trivial files will be batched together into a single LLM call.
    """
    if not hasattr(manifest, "files"):
        return False
    fm = getattr(manifest, "by_task", lambda _: None)(task.id)
    if fm is None:
        return False

    filepath = getattr(fm, "filepath", "")
    exports  = getattr(fm, "exports", []) or []
    return is_trivial_filepath(filepath, exports)


def _shares_resource(path_a: str, path_b: str) -> bool:
    """True when two paths share a directory or an identifier token (e.g. both relate to 'todo')."""
    if not path_a or not path_b:
        return False
    if os.path.dirname(path_a).rstrip("/") == os.path.dirname(path_b).rstrip("/"):
        return True
    a_tokens = {t for t in os.path.basename(path_a).split(".")[0].lower().split("_") if t}
    b_tokens = {t for t in os.path.basename(path_b).split(".")[0].lower().split("_") if t}
    return bool(a_tokens & b_tokens)


def build_project_file_tree(
    manifest: Any,
    highlight_path: str = "",
    task: Any = None,
    agent: str | None = None,
    include_trivial: bool = True,
) -> str:
    """
    Build a CAPPED project file tree for injection into coding prompts.

    Capping rule (keeps ~95% of the full-tree benefit at small token cost):
      include a file iff it is one of:
        - a declared dependency of `task` (task.depends_on)
        - already written
        - a trivial/boilerplate file
        - a same-agent + same-resource file
    The current file is marked with '★'.
    """
    if not hasattr(manifest, "files") or not manifest.files:
        return "(no files planned yet)"
    if highlight_path not in manifest.files:
        # highlight may be an unregistered scaffold path — find by filepath anyway
        pass

    dep_ids = set(task.depends_on) if (task and getattr(task, "depends_on", None)) else set()
    all_filepaths = sorted(
        manifest.files.keys(),
        key=lambda p: (p.rstrip("/").count("/"), p),
    )
    entries: list[str] = []
    for path in all_filepaths:
        fm = manifest.files[path]
        written = getattr(fm, "written", False)
        included = (
            path == highlight_path
            or getattr(fm, "task_id", None) in dep_ids
            or written
            or (include_trivial and is_trivial_filepath(path, getattr(fm, "exports", [])))
            or (fm.agent == agent and _shares_resource(path, highlight_path))
        )
        if not included:
            continue
        marker = " ★ (generating)" if path == highlight_path else ""
        status = "written" if written else "pending"
        exports = ", ".join(getattr(fm, "exports", []) or []) or "(none)"
        depth = path.rstrip("/").count("/")
        entries.append(
            f"{'  ' * depth}- `{path}`  [{status}] {fm.agent} — exports: {exports}{marker}"
        )
    if not entries:
        return "(no relevant project files planned yet)"
    if len(entries) > MAX_TREE_ENTRIES:
        omitted = len(entries) - MAX_TREE_ENTRIES
        entries = entries[:MAX_TREE_ENTRIES]
        entries.append(f"  … and {omitted} more files omitted (tree capped)")
    return "\n".join(entries)


def build_scaffold_context(manifest: Any, filepath: str) -> str:
    """
    Deterministic wiring instructions for entrypoint/scaffold files .
    Backend entrypoint (main.py) gets the exact router list; index.html gets the
    exact stylesheet/script list — no LLM guessing about what to load.
    """
    if not hasattr(manifest, "files"):
        return ""
    base = os.path.basename(filepath or "").lower()
    if base in ("main.py", "app.py", "server.py"):
        routers = []
        for path, fm in manifest.files.items():
            if path == filepath:
                continue
            if fm.agent != "backend":
                continue
            exports = getattr(fm, "exports", []) or []
            if "router" in exports or "routes" in path.lower() or "controllers" in path.lower():
                routers.append(f"- `{path}` → `from {_module_path_for(path)} import router`")
        header = "## Scaffold Wiring — You MUST register these routers (their files already exist):"
        note = "\n**IMPORT RULE**: Import each router with the EXACT import line shown (layer-root-relative), NEVER with the `output.backend.` / `output.` workspace prefix."
        return header + note + ("\n" + "\n".join(routers) if routers else "\n(none found)")
    if base == "index.html":
        styles = sorted(
            p for p in manifest.files if p != filepath and p.endswith((".css", ".scss", ".sass"))
        )
        scripts = sorted(
            p for p in manifest.files
            if p != filepath and p.endswith((".js", ".ts", ".tsx", ".jsx"))
        )
        parts = ["## Scaffold Wiring — index.html MUST load the following files in this order:"]
        if styles:
            parts.append("\n".join([f"- stylesheet: `{p}`" for p in styles]))
        if scripts:
            parts.append("\n".join([f"- script: `{p}`" for p in scripts]))
        if not styles and not scripts:
            parts.append("(no stylesheets/scripts planned yet)")
        if scripts:
            parts.append(
                "**NO INLINE LOGIC RULE**: All application behavior MUST live in the script "
                "files listed above and be loaded with <script src>. Do NOT duplicate "
                "fetch/render logic inside inline <script> tags in index.html — inline "
                "duplicates shadow the real script files and break the app."
            )
        return "\n".join(parts)
    return ""


####################BaseAgent##########################
class BaseAgent(ABC):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRIES,
        retries: int | None = None,
        backoff: float = DEFAULT_BACKOFF,
        temperature: float = 0.2,
        provider: BaseLLMProvider | str | None = None,
    ):
        """Initialize the base agent."""
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = retries if retries is not None else max_retries
        self.backoff = backoff
        self.temperature = temperature
        
        if isinstance(provider, str):
            provider_name = provider.lower().strip()
            self.provider = llmFactory.get(provider_name)
        elif provider is not None:
            self.provider = provider
        else:
            provider_name = self.model.split("/")[0] if "/" in self.model else settings.default_provider
            if provider_name not in llmFactory.available():
                provider_name = settings.default_provider
            self.provider = llmFactory.get(provider_name)

    @property
    def name(self)->str:
        """Get the name of the agent."""
        return self.__class__.__name__

    @property
    def code_bucket(self) -> str:
        """Name of the dictionary attribute in ProjectState.generated_code (e.g. 'backend', 'frontend', 'database')."""
        name = self.name.lower()
        if name.endswith("agent"):
            name = name[:-5]
        if name in ("data_base", "db"):
            name = "database"
        return name

    @property
    @abstractmethod
    def system_prompt(self)->str:
        """Get the system prompt of the agent."""
        ...

    def _build_prompt(self, state: ProjectState) -> str:
        """Build the prompt for single-call agents using default run(). Override in subclass if using default run()."""
        raise NotImplementedError(f"[{self.name}] _build_prompt() is not implemented.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        """Parse response for single-call agents using default run(). Override in subclass if using default run()."""
        raise NotImplementedError(f"[{self.name}] _parse_response() is not implemented.")
    


    ##LLM CALL
    def llm_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> LLMResponse:
        

        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        prefix = agent_name if agent_name is not None else self.name

        printer = LineBufferedPrinter(prefix)
        try:
            response = self.provider.complete(
                system_prompt = sys_prompt,
                user_message  = prompt,
                model         = self.model,
                max_tokens    = self.max_tokens,
                temperature   = self.temperature,
                stream        = True,
                chunk_callback = printer,
            )
        except TypeError as e:
            if "unexpected keyword argument" in str(e) or "complete() got" in str(e):
                response = self.provider.complete(
                    system_prompt = sys_prompt,
                    user_message  = prompt,
                    model         = self.model,
                    max_tokens    = self.max_tokens,
                    temperature   = self.temperature,
                )
            else:
                raise
        printer.flush()
        return response

    def _call_with_system(
        self,
        system: str,
        user: str,
        agent_name: str = "",
        state: Optional[ProjectState] = None,
        lock: Optional[Any] = None,
    ) -> str:
        """
        Variant of llm_call() that accepts a dedicated system prompt.
        Retries up to self.max_retries with exponential back-off.
        Returns raw text and thread-safely updates state metrics if state is provided.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(
                    prompt=user,
                    system_prompt=system,
                    agent_name=agent_name or self.name,
                )
                if state is not None:
                    if lock is not None:
                        with lock:
                            state.metrics.total_tokens_used += response.total_tokens
                            state.metrics.total_llm_calls += 1
                    else:
                        state.metrics.total_tokens_used += response.total_tokens
                        state.metrics.total_llm_calls += 1
                return response.text
            except Exception as exc:
                logger.warning(
                    "[%s] LLM attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)
        raise RuntimeError("Unreachable")

    def _parse_batch_response(self, raw: str) -> list[dict]:
        """
        Parse raw text response from a batch LLM call into a list of dictionaries.
        Strips markdown code fences and extracts JSON array or object fallback.
        """
        text = self._strip_fences(raw).strip()

        # 1. Try parsing JSON array
        try:
            raw_arr = self._extract_raw_json(text, "[")
            items = self._try_parse(raw_arr, list)
            if isinstance(items, list):
                return deep_unescape(items)
        except Exception:
            pass

        # 2. Try parsing single JSON object and wrapping in list
        try:
            raw_obj = self._extract_raw_json(text, "{")
            items = self._try_parse(raw_obj, dict)
            if isinstance(items, dict):
                return deep_unescape([items])
        except Exception:
            pass

        # 3. Fallback using json_repair
        import json_repair
        try:
            res = json_repair.loads(text)
            if isinstance(res, list):
                return deep_unescape(res)
            elif isinstance(res, dict):
                return deep_unescape([res])
        except Exception:
            pass

        raise ValueError(f"Batch JSON parse failed: {raw[:300]}")


    @staticmethod
    def _compute_diff(before: ProjectState, after: ProjectState) -> dict[str, Any]:
        """
        Compare two ProjectState snapshots and return only the fields that changed as a dict.

        For Annotated/reducer fields (metrics, errors, current_agent), returns
        only the delta so LangGraph reducers merge correctly.
        """
        from core.state import PerformanceMetrics

        before_dict = before.model_dump(mode="json")
        after_dict = after.model_dump(mode="json")

        diff: dict[str, Any] = {}
        for key in after_dict:
            if key == "metrics":
                bm = before_dict["metrics"]
                am = after_dict["metrics"]
                dt = am["total_tokens_used"] - bm["total_tokens_used"]
                dc = am["total_llm_calls"] - bm["total_llm_calls"]
                nd = {k: v for k, v in am["agent_durations"].items()
                      if bm["agent_durations"].get(k) != v}
                if dt or dc or nd:
                    diff["metrics"] = PerformanceMetrics(
                        total_tokens_used=dt,
                        total_llm_calls=dc,
                        agent_durations=nd,
                    ).model_dump(mode="json")
                continue

            if key == "errors":
                new_errors = after_dict["errors"][len(before_dict["errors"]):]
                if new_errors:
                    diff["errors"] = new_errors
                continue

            if key == "current_agent":
                if after_dict["current_agent"] != before_dict["current_agent"]:
                    diff["current_agent"] = after_dict["current_agent"]
                continue

            if key == "completed_task_ids":
                b_ids = set(before_dict.get("completed_task_ids") or [])
                a_ids = set(after_dict.get("completed_task_ids") or [])
                new_ids = a_ids - b_ids
                if new_ids:
                    diff["completed_task_ids"] = list(new_ids)
                continue

            if before_dict[key] != after_dict[key]:
                diff[key] = after_dict[key]

        return diff


    def run(self, state: dict | ProjectState) -> dict[str, Any]:
        """
        Run the agent.
        Accepts the full state (dict or ProjectState) as input.
        Returns a dict containing ONLY the changed/wanted fields.
        """
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)

        ps.mark_agent_start(self.name)
        prompt = self._build_prompt(ps)
        t0 = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(prompt)
                ps.metrics.total_tokens_used += response.total_tokens
                ps.metrics.total_llm_calls   += 1

                ps = self._parse_response(ps, response.text)
                break

            except Exception as exc:
                logger.warning(
                    "[%s] attempt %d/%d failed: %s",
                    self.name, attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    logger.error("[%s] all retries exhausted.", self.name)
                    ps.log_error(self.name, str(exc))
                else:
                    time.sleep(self.backoff * attempt)

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)
        logger.info("[%s] finished in %.2fs via %s", self.name, duration, self.provider.provider_name)
        return self._compute_diff(before, ps)


    # ── JSON helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) and return inner content."""
        fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        return fenced.group(1) if fenced else text

    @staticmethod
    def _extract_raw_json(text: str, bracket: str = "{") -> str:
        """
        Extract the first balanced JSON object or array from *text*.

        Uses a bracket-counting scan instead of a greedy regex so that
        `{"a":1} some trailing text {"b":2}` correctly returns `{"a":1}`.
        """
        open_b  = bracket
        close_b = "}" if bracket == "{" else "]"
        start   = text.find(open_b)
        if start == -1:
            kind = "object" if bracket == "{" else "array"
            raise ValueError(f"No JSON {kind} found in response:\n{text[:300]}")

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_b:
                depth += 1
            elif ch == close_b:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # Unbalanced — return from start to end and let the parser try
        return text[start:]

    @staticmethod
    def _try_parse(raw: str, expect_type: type):
        """
        Try json.loads first, then json_repair.loads.
        Returns the parsed object or raises ValueError.
        """
        import json_repair

        # 1. Strict parse
        try:
            result = json.loads(raw)
            if isinstance(result, expect_type):
                return result
        except Exception:
            pass

        # 2. json_repair
        try:
            result = json_repair.loads(raw)
            if isinstance(result, expect_type):
                return result
        except Exception:
            pass

        type_name = "dict" if expect_type is dict else "list"
        raise ValueError(f"Failed to parse JSON as {type_name}:\n{raw[:300]}")

    # ── Public extraction methods ─────────────────────────────────────

    @classmethod
    def extract_json(cls, text: str) -> dict[str, Any]:
        """
        Safely extract the first JSON object from a model response.
        Handles markdown code fences (```json ... ```) gracefully.
        """
        text = cls._strip_fences(text)
        raw  = cls._extract_raw_json(text, "{")
        return deep_unescape(cls._try_parse(raw, dict))

    @classmethod
    def extract_json_list(cls, text: str) -> list[Any]:
        """
        Safely extract the first JSON array from a model response.
        """
        text = cls._strip_fences(text)
        raw  = cls._extract_raw_json(text, "[")
        return deep_unescape(cls._try_parse(raw, list))

    @staticmethod
    def extract_section(text: str, tag: str) -> str:
        """
        Extract content between <tag> ... </tag> in the response.
        Useful when the model is prompted to use XML-style delimiters.
        """
        pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
        match   = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError(f"<{tag}> section not found in response.")
        return match.group(1).strip()

    # ── Validation helpers ────────────────────────────────────────────

    @staticmethod
    def _validate_data(data: Any, schema_cls: Any, is_list: bool) -> Any:
        """
        Run Pydantic validation on parsed data.
        Returns the validated/coerced Python object(s).
        """
        if is_list:
            if not isinstance(data, list):
                raise ValueError("Expected a list, got " + str(type(data)))
            validated = []
            for item in data:
                if hasattr(schema_cls, "model_validate"):
                    obj = schema_cls.model_validate(item)
                else:
                    obj = schema_cls(**item)
                validated.append(obj.model_dump() if hasattr(obj, "model_dump") else item)
            return validated
        else:
            if not isinstance(data, dict):
                raise ValueError("Expected a dictionary, got " + str(type(data)))
            if hasattr(schema_cls, "model_validate"):
                obj = schema_cls.model_validate(data)
            else:
                obj = schema_cls(**data)
            return obj.model_dump() if hasattr(obj, "model_dump") else data

    def _repair_json_via_llm(
        self,
        json_fragment: str,
        schema_cls: Any,
        state: Optional[ProjectState] = None,
        lock: Optional[Any] = None,
    ) -> str:
        """
        Send **only** the JSON fragment (not the full LLM prose) to the
        model for repair, together with the expected schema so the model
        knows what structure to preserve.
        """
        schema_hint = ""
        if hasattr(schema_cls, "model_json_schema"):
            try:
                schema_hint = (
                    "\n\nExpected JSON schema:\n"
                    + json.dumps(schema_cls.model_json_schema(), indent=2)
                )
            except Exception:
                pass

        repair_prompt = (
            "The following JSON is invalid or does not match the expected schema.\n\n"
            "Repair it so it becomes valid JSON matching the schema.\n\n"
            "Rules:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT add new information.\n"
            "- Only fix syntax errors and missing/misnamed fields.\n"
            "- Return ONLY the corrected JSON, no explanation.\n"
            f"{schema_hint}\n\n"
            f"{json_fragment}"
        )

        logger.info("[%s] Sending JSON fragment to LLM for repair…", self.name)
        try:
            raw_text = self._call_with_system(
                system="You are a helpful assistant that repairs JSON syntax.",
                user=repair_prompt,
                agent_name=f"{self.name}-repair",
                state=state,
                lock=lock,
            )
            return raw_text
        except Exception as exc:
            logger.error("[%s] LLM JSON repair call failed: %s", self.name, exc)
            return json_fragment

    def extract_json_and_validate(
        self,
        text: str,
        schema_cls: Any,
        is_list: bool = False,
        state: Optional[ProjectState] = None,
        lock: Optional[Any] = None,
    ) -> Any:
        """
        Three-stage extraction pipeline:

        1. **Parse + validate** — json.loads / json_repair on the raw text.
        2. **Local repair + validate** — re-run json_repair on just the
           extracted substring (handles truncated brackets, trailing commas).
        3. **LLM repair + validate** — send the JSON fragment (not the full
           response) to the model with the target schema for correction.

        Returns the validated (and pydantic-coerced) data.
        """
        bracket   = "[" if is_list else "{"
        extractor = self.extract_json_list if is_list else self.extract_json

        # ── Stage 1: direct parse ────────────────────────────────────
        try:
            parsed = extractor(text)
            return self._validate_data(parsed, schema_cls, is_list)
        except Exception as stage1_exc:
            logger.warning(
                "[%s] Stage 1 (direct parse) failed: %s", self.name, stage1_exc
            )

        # ── Stage 2: local repair on raw substring ───────────────────
        raw_fragment = ""
        try:
            stripped = self._strip_fences(text)
            raw_fragment = self._extract_raw_json(stripped, bracket)
        except Exception:
            raw_fragment = text  # fallback — let json_repair try the whole thing

        import json_repair
        try:
            repaired_local = json_repair.loads(raw_fragment)
            if not isinstance(repaired_local, (dict, list)):
                raise ValueError("json_repair returned unexpected type")
            return self._validate_data(deep_unescape(repaired_local), schema_cls, is_list)
        except Exception as stage2_exc:
            logger.warning(
                "[%s] Stage 2 (local repair) failed: %s", self.name, stage2_exc
            )

        # ── Stage 3: LLM repair (send only the JSON fragment) ───────
        repaired_text = self._repair_json_via_llm(raw_fragment, schema_cls, state=state, lock=lock)
        try:
            parsed_repaired = extractor(repaired_text)
            result = self._validate_data(parsed_repaired, schema_cls, is_list)
            logger.info("[%s] Stage 3 (LLM repair) succeeded.", self.name)
            return result
        except Exception as stage3_exc:
            logger.error(
                "[%s] All 3 repair stages failed. Final error: %s",
                self.name, stage3_exc,
            )
            raise stage3_exc

########stream printer with thread safety########   
class LineBufferedPrinter:
            _print_lock = Lock()
            def __init__(self, prefix: str):
                self.prefix = prefix
                self.buffer = ""
            def __call__(self, chunk: str):
                self.buffer += chunk
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    with LineBufferedPrinter._print_lock:
                        sys.stdout.write(f"[{self.prefix}] {line}\n")
                        sys.stdout.flush()
            def flush(self):
                if self.buffer:
                    with LineBufferedPrinter._print_lock:
                        sys.stdout.write(f"[{self.prefix}] {self.buffer}\n")
                        sys.stdout.flush()
                    self.buffer = ""
