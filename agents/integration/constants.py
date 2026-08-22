"""
agents/integration/constants.py
─────────────────────────────────
Module-level constants extracted from agents/integration.py (Phase 1.2).

All values are frozen / read-only. Prompt templates are kept here too so the
main agent module stays focused on logic.
"""

import re
from agents.utils import OUTPUT_DIR  # single source — no redefinition

# ── Parallelism & retry ───────────────────────────────────────────────────────
MAX_PARALLEL_CALLS = 3

#: Maximum number of LLM repair rounds the agent will attempt before giving up
#: (replaces the magic number '3' that was inlined in the repair loop).
MAX_REPAIR_ROUNDS = 3

#: Character window after a route decorator in which to look for the
#: `response_model=...` kwarg (replaces the magic number '600').
RESPONSE_MODEL_WINDOW = 600

# ── Conflict routing ─────────────────────────────────────────────────────────

#: Conflict types that integration must NEVER guess a fix for — they are routed
#: straight to the owning coding agent via the REWORK loop with grounded error text.
REWORK_ONLY_TYPES = frozenset({
    "SYNTAX", "IMPORT_NAME", "STARTUP", "SCHEMA_COLUMN", "WIRING",
    "FRONTEND_HTTP", "FRONTEND_DOM", "FILE_SET", "FILE_CORRUPTION",
})

# ── Parsing / resolution helpers ─────────────────────────────────────────────

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

#: FastAPI route-handler parameter types to skip when identifying request models.
_FASTAPI_SKIP_TYPES = frozenset({
    "request", "response", "backgroundtasks", "str", "int", "bool", "float",
    "list", "dict", "any", "optional", "depends", "session", "none", "status",
    "tuple", "set", "bytes", "none",
})

#: Fetch-response methods that are NOT entity fields (frontend usage analysis).
_JS_MEMBER_SAFE = frozenset({
    "json", "text", "ok", "status", "statusText", "headers", "then", "catch",
    "finally", "push", "pop", "shift", "unshift", "splice", "concat", "slice",
    "map", "filter", "reduce", "forEach", "find", "includes", "indexOf",
    "length", "trim", "toLowerCase", "toUpperCase", "replace", "split", "join",
    "appendChild", "removeChild", "addEventListener", "classList", "dataset",
    "setAttribute", "getAttribute", "focus", "innerHTML", "textContent", "value",
})

#: Python keywords — used by extract_actual_definitions to skip keyword-starting
#: lines when detecting module-level constants.
_PY_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
    "or", "pass", "raise", "return", "try", "while", "with", "yield",
}

#: Primitive type normalisation mapping for contract field checks.
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

_REFUSED_FILEPATH_RE = re.compile(r"File:\s*([^\s|]+)")


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
