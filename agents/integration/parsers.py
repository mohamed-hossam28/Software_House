"""
agents/integration/parsers.py
──────────────────────────────
Pure parsing functions extracted from agents/integration.py (Phase 1.3).

Every function here is side-effect-free: it takes a code string (and sometimes
a filepath for language dispatch) and returns structured data. No state, no I/O.
"""

import re
from typing import Optional

from agents.integration.constants import (
    RESPONSE_MODEL_WINDOW,
    _FASTAPI_SKIP_TYPES,
    _HTTP_VERBS,
    _JS_MEMBER_SAFE,
    _PYDANTIC_BASES,
    _PY_KEYWORDS,
)


# ── Import parsers ─────────────────────────────────────────────────────────────

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


# ── Definition / export extraction ────────────────────────────────────────────

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


# ── Call-site argument analysis ───────────────────────────────────────────────

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


# ── SQLAlchemy / Pydantic model parsers ───────────────────────────────────────

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


# ── Backend route parsers ─────────────────────────────────────────────────────

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
        rm = re.search(
            r"response_model\s*=\s*(?:List\[(\w+)\]|(\w+))",
            code[dm.end():dm.end() + RESPONSE_MODEL_WINDOW],
        )
        if rm:
            response_model = rm.group(1) or rm.group(2)

        routes.append({
            "method": method,
            "path": full_path,
            "request_model": request_model,
            "response_model": response_model,
        })
    return routes


# ── Frontend fetch parsers ────────────────────────────────────────────────────

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


# ── DOM / CSS parsers ─────────────────────────────────────────────────────────

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


# ── Response-field usage analysis ─────────────────────────────────────────────

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
