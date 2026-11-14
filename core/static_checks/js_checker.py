"""
JavaScript static checks: syntax parsing and frontend-to-contract URL matching.

Syntax — every generated .js/.mjs file is parsed with the pure-Python esprima
parser (no JSX). A parse failure is an error finding; script vs module mode is
picked by a lightweight import/export heuristic with cross-try fallback.

URL matching — fetch(...) and axios(...) calls are extracted, their URL
expression resolved to a normalized path (base-URL/env prefixes and full URLs
stripped to the path, query strings and trailing slashes removed, `${id}` /
`:id` / `{id}` parameters unified), and then matched by method and path shape
against the Architecture Agent's API contracts. A call that matches no contract
— or that hits a contract path with the wrong method — is an error finding.

Calls whose URL cannot be statically resolved (a bare variable, a function
call) are skipped rather than guessed.
"""
import re
from typing import Any

import esprima

from core.state import FindingSeverity, StaticFinding

HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
_AXIOS_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "request"}
_URL_PLACEHOLDER = "{}"
_ABSOLUTE_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://[^/]*")


def check_javascript(
    files: dict[str, str],
    manifest: Any,
    architecture: Any = None,
    dialect: str = "",
) -> list[StaticFinding]:
    """Run all JavaScript checks over *files* and return the findings.

    manifest and dialect are part of the uniform checker signature; they are
    unused by the JavaScript checks.
    """
    findings: list[StaticFinding] = []
    js_files = {
        filepath: code
        for filepath, code in sorted(files.items())
        if filepath.endswith((".js", ".mjs")) and code
    }

    contracts = _contract_index(architecture)

    for filepath, code in js_files.items():
        tree, error = _parse(code)
        if tree is None:
            findings.append(StaticFinding(
                file=filepath,
                check="js_syntax",
                severity=FindingSeverity.ERROR,
                line=error.get("line", 1),
                message=f"SyntaxError: {error.get('message', 'could not be parsed')}",
            ))
            continue
        if contracts:
            findings.extend(_check_urls(filepath, tree, contracts))
    return findings


# ── Parsing ──────────────────────────────────────────────────────────────────


def _looks_like_module(code: str) -> bool:
    return re.search(r"^\s*(import\s|export\s)", code, re.MULTILINE) is not None


def _parse(code: str) -> tuple[Any, dict[str, Any] | None]:
    """Parse *code* in the mode its shape suggests, falling back to the other.

    esprima cannot parse `import.meta`, so a final attempt rewrites it to a
    placeholder identifier before giving up.
    Returns (ast, None) on success or (None, {"message", "line"}) on failure.
    """
    attempts = (
        (esprima.parseModule, esprima.parseScript)
        if _looks_like_module(code)
        else (esprima.parseScript, esprima.parseModule)
    )
    primary_error: dict[str, Any] | None = None
    parsers = [*attempts, _sanitized_parser(attempts[0])]
    for parser in parsers:
        try:
            return parser(code), None
        except Exception as exc:  # esprima.error_handler.Error and defensive
            error = {
                "message": getattr(exc, "message", None) or str(exc),
                "line": getattr(exc, "lineNumber", None) or 1,
            }
            if primary_error is None:
                primary_error = error
    return None, primary_error


_IMPORT_META_RE = re.compile(r"\bimport\s*\.\s*meta\b")


def _sanitized_parser(parser):
    """Wrap *parser* to rewrite `import.meta` into a plain identifier first."""

    def parse(code: str):
        return parser(_IMPORT_META_RE.sub("__import_meta__", code))

    return parse


def _walk(node: Any):
    """Yield every AST node (self first) reachable from *node*."""
    if not hasattr(node, "type"):
        return
    yield node
    for value in vars(node).values():
        if value is node:
            continue
        for child in (value if isinstance(value, list) else [value]):
            yield from _walk(child)


# ── Contract matching ────────────────────────────────────────────────────────


def _contract_index(architecture: Any) -> dict[tuple[tuple[str, ...], frozenset[str]]]:
    """path-shape → set of HTTP methods; empty when there is nothing to match against."""
    contracts = getattr(architecture, "api_contracts", None) or []
    index: dict[tuple[tuple[str, ...], frozenset[str]]] = {}
    for contract in contracts:
        method = str(getattr(contract, "method", "") or "").upper()
        path = str(getattr(contract, "path", "") or "")
        if not method or not path:
            continue
        shape = _path_shape(_normalize_path(path))
        methods = index.setdefault(shape, frozenset()) | {method}
        index[shape] = methods
    return index


def _shapes_match(call: tuple[str, ...], contract: tuple[str, ...]) -> bool:
    """True when the two path shapes align, with parameter slots matching any value."""
    return len(call) == len(contract) and all(
        a == b or _URL_PLACEHOLDER in (a, b) for a, b in zip(call, contract)
    )


def _check_urls(
    filepath: str,
    tree: Any,
    contracts: dict[tuple[tuple[str, ...], frozenset[str]]],
) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    for node in _walk(tree):
        if node.type != "CallExpression":
            continue
        url_expr, method = _call_info(node)
        path = _path_from_expr(url_expr)
        if not path or not method:
            continue
        shape = _path_shape(path)
        line = getattr(getattr(node, "loc", None), "start", None)
        line = getattr(line, "line", 1) or 1
        methods_on_path = sorted(
            m for known, methods in contracts.items() if _shapes_match(shape, known)
            for m in methods
        )
        if method in methods_on_path:
            continue
        if methods_on_path:
            message = (
                f"Frontend call to {method} {path} hits a contract path but no "
                f"contract defines {method} there (defined: {', '.join(methods_on_path)})"
            )
        else:
            message = f"Frontend call to {method} {path} does not match any API contract"
        findings.append(StaticFinding(
            file=filepath,
            check="js_urls",
            severity=FindingSeverity.ERROR,
            line=line,
            message=message,
        ))
    return findings


def _call_info(call: Any) -> tuple[Any, str | None]:
    """(URL expression, HTTP method) for a fetch/axios call; (None, None) otherwise."""
    callee = call.callee
    args = list(call.arguments or [])

    if callee.type == "Identifier":
        if callee.name == "fetch" and args:
            return args[0], _method_from_config(args[1] if len(args) > 1 else None) or "GET"
        if callee.name == "axios" and args:
            if args[0].type == "ObjectExpression":
                return _object_property(args[0], "url"), _method_from_config(args[0]) or "GET"
            return args[0], "GET"
    elif callee.type == "MemberExpression":
        obj, prop = callee.object, callee.property
        if getattr(obj, "name", None) == "axios" and getattr(prop, "name", None) in _AXIOS_VERBS:
            verb = prop.name
            if verb == "request" and args and args[0].type == "ObjectExpression":
                return _object_property(args[0], "url"), _method_from_config(args[0]) or "GET"
            if not args:
                return None, None
            return args[0], "GET" if verb == "request" else verb.upper()
    return None, None


def _method_from_config(arg: Any) -> str | None:
    """Uppercased 'method' from an options object literal, when statically known."""
    if arg is None or arg.type != "ObjectExpression":
        return None
    value = _object_property(arg, "method")
    if value is not None and value.type == "Literal" and isinstance(value.value, str):
        method = value.value.upper()
        return method if method in HTTP_METHODS else None
    return None


def _object_property(obj: Any, key: str) -> Any | None:
    """Value node of an object literal property *key*; None when absent/unknown."""
    for prop in getattr(obj, "properties", []) or []:
        prop_key = getattr(prop, "key", None)
        if prop_key is None:
            continue
        name = prop_key.name if prop_key.type == "Identifier" else prop_key.value
        if name == key:
            return prop.value
    return None


# ── URL normalization ────────────────────────────────────────────────────────


def _path_from_expr(expr: Any) -> str | None:
    """Normalized path string of a URL expression; None when unresolvable."""
    if expr is None:
        return None
    if expr.type == "Literal" and isinstance(expr.value, str):
        return _normalize_path(expr.value)
    if expr.type == "TemplateLiteral":
        return _normalize_path(_template_raw(expr))
    if expr.type == "BinaryExpression" and getattr(expr.operator, "operator", expr.operator) == "+":
        left = _path_from_expr(expr.left) or ""
        right = _path_from_expr(expr.right) or ""
        return _normalize_path(left + right)
    return None


def _template_raw(template: Any) -> str:
    """Raw text of a template literal; interpolations become {P} parameters.

    A leading interpolation is treated as a base-URL prefix and dropped, so
    `${API_BASE}/todos` and `${import.meta.env.X}/todos` resolve to /todos.
    """
    pieces: list[str] = []
    expressions = list(getattr(template, "expressions", []) or [])
    for index, quasi in enumerate(getattr(template, "quasis", []) or []):
        pieces.append(quasi.value.raw or "")
        if index < len(expressions):
            pieces.append("{P}" if (pieces and any(p for p in pieces)) else "")
    return "".join(pieces)


def _normalize_path(raw: str) -> str | None:
    """Path portion of *raw*: scheme/host stripped, query and trailing slash removed."""
    path = _ABSOLUTE_URL_RE.sub("", raw or "")
    path = path.split("?", 1)[0]
    if not path.startswith("/"):
        start = path.find("/")
        if start < 0:
            return None
        path = path[start:]
    path = path.rstrip("/") or "/"
    return path


def _path_shape(path: str) -> tuple[str, ...]:
    """Path reduced to comparable segments; parameters become a shared placeholder."""
    segments: list[str] = []
    for segment in path.split("/"):
        if not segment:
            continue
        if (
            (segment.startswith("{") and segment.endswith("}"))
            or segment.startswith(":")
            or segment == "{P}"
        ):
            segments.append(_URL_PLACEHOLDER)
        else:
            segments.append(segment.lower())
    return tuple(segments)
