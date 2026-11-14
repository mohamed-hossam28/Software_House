"""
JavaScript static checks: syntax parsing and frontend-to-contract URL matching.

Syntax — every generated .js/.mjs file is parsed with the pure-Python esprima
parser (no JSX). A parse failure is an error finding; script vs module mode is
picked by a lightweight import/export heuristic with cross-try fallback.

URL matching — fetch(...) and axios(...) calls are extracted, their URL
expression resolved to normalized path candidates (base-URL/env prefixes and
full URLs stripped to the path, query strings and trailing slashes removed,
`${id}` / `:id` / `{id}` parameters unified), and then matched by method and
path shape against the Architecture Agent's API contracts together with the
backend's implemented FastAPI routes (extracted by
core.static_checks.route_checker — the two-way HTTP contract story). A call is
clean when any candidate matches a contract or an implemented route; a call
whose every candidate matches neither — or that hits a known path only with a
different method — is an error finding. Candidates exist because a template
literal's leading interpolation is ambiguous: `${BASE}/todos` (base-URL
prefix) and `/${id}/todos` (leading path parameter) are structurally
identical, so both readings are tried and any-match wins — no false positives
at the cost of missing an error only when one reading matches.

Calls whose URL cannot be statically resolved (a bare variable, a function
call) are skipped rather than guessed. The URL check runs when there is
anything to match against — API contracts or implemented backend routes
(see core.static_checks.route_checker); when both are empty it is skipped,
because flagging every frontend call would be noise.

The manifest and dialect parameters are part of the uniform checker signature
and are unused by the JavaScript checks.
"""
import re
from typing import Any

import esprima

from core.state import FindingSeverity, StaticFinding
from core.static_checks.http_paths import (
    HTTPIndex,
    PATH_SLOT,
    PathShape,
    contract_index,
    normalize_path,
    path_shape,
    shapes_match,
)
from core.static_checks.route_checker import extract_backend_routes

HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
_AXIOS_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "request"}


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

    contracts = contract_index(architecture)
    routes = extract_backend_routes(files)
    targets = _merge_indexes(contracts, routes)

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
        if targets:
            findings.extend(_check_urls(filepath, tree, targets))
    return findings


def _merge_indexes(contracts: HTTPIndex, routes: HTTPIndex) -> HTTPIndex:
    """Contracts and implemented backend routes as one path shape → methods map."""
    merged: HTTPIndex = {}
    for index in (contracts, routes):
        for shape, methods in index.items():
            merged[shape] = merged.get(shape, frozenset()) | methods
    return merged


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


def _node_line(node: Any) -> int:
    """Source line of *node* as a plain integer (1 when location is unknown)."""
    line = getattr(getattr(getattr(node, "loc", None), "start", None), "line", None)
    return line or 1


# ── Contract matching ────────────────────────────────────────────────────────


def _check_urls(
    filepath: str,
    tree: Any,
    targets: HTTPIndex,
) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    for node in _walk(tree):
        if node.type != "CallExpression":
            continue
        url_expr, method = _call_info(node)
        candidates = _path_candidates(url_expr)
        if not candidates or not method:
            continue
        if _any_candidate_matches(candidates, method, targets):
            continue
        methods_on_path = _methods_on_any_candidate(candidates, targets)
        line = _node_line(node)
        if methods_on_path:
            message = (
                f"Frontend call to {method} {candidates[0]} hits a contract path but no "
                f"contract or implemented route defines {method} there "
                f"(defined: {', '.join(methods_on_path)})"
            )
        else:
            message = (
                f"Frontend call to {method} {candidates[0]} matches no API contract "
                f"or implemented backend route"
            )
        findings.append(StaticFinding(
            file=filepath,
            check="js_urls",
            severity=FindingSeverity.ERROR,
            line=line,
            message=message,
        ))
    return findings


def _any_candidate_matches(
    candidates: list[str],
    method: str,
    targets: HTTPIndex,
) -> bool:
    """True when any path candidate matches a contract or implemented route."""
    for path in candidates:
        shape = path_shape(path)
        if method in targets.get(shape, frozenset()):
            return True
        for known, methods in targets.items():
            if method in methods and shapes_match(shape, known):
                return True
    return False


def _methods_on_any_candidate(
    candidates: list[str],
    targets: HTTPIndex,
) -> list[str]:
    """Methods contracts or routes define on any path shape a candidate could be."""
    methods: set[str] = set()
    for path in candidates:
        shape = path_shape(path)
        for known, known_methods in targets.items():
            if shapes_match(shape, known):
                methods.update(known_methods)
    return sorted(methods)


def _call_info(call: Any) -> tuple[Any, str | None]:
    """(URL expression, HTTP method) for a fetch/axios call; (None, None) otherwise."""
    callee = call.callee
    args = list(call.arguments or [])

    if callee.type == "Identifier":
        if callee.name == "fetch" and args:
            return args[0], _method_from_config(args[1] if len(args) > 1 else None) or "GET"
        if callee.name == "axios" and args:
            if args[0].type == "ObjectExpression":
                return _axios_config_call(args[0])
            return args[0], "GET"
    elif callee.type == "MemberExpression":
        obj, prop = callee.object, callee.property
        if getattr(obj, "name", None) == "axios" and getattr(prop, "name", None) in _AXIOS_VERBS:
            verb = prop.name
            if not args:
                return None, None
            if verb == "request" and args[0].type == "ObjectExpression":
                return _axios_config_call(args[0])
            return args[0], "GET" if verb == "request" else verb.upper()
    return None, None


def _axios_config_call(config: Any) -> tuple[Any | None, str | None]:
    """(url, method) from an axios-style config object literal; method defaults to GET."""
    return _object_property(config, "url"), _method_from_config(config) or "GET"


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


def _path_candidates(expr: Any) -> list[str]:
    """Normalized path readings of a URL expression, best first.

    Empty list = statically unresolvable as a standalone URL. A template
    literal yields one reading per interpretation of its leading
    interpolation (base-URL prefix, or path parameter); other expressions
    yield at most one.
    """
    if expr is None:
        return []
    if expr.type == "Literal" and isinstance(expr.value, str):
        return _dedupe_paths([normalize_path(expr.value)])
    if expr.type == "TemplateLiteral":
        readings = [
            _template_raw(expr, leading_is_parameter=leading)
            for leading in (False, True)
        ]
        return _dedupe_paths(normalize_path(raw) for raw in readings)
    if expr.type == "BinaryExpression" and getattr(expr.operator, "operator", expr.operator) == "+":
        lefts = _path_candidates(expr.left) or [""]
        rights = _path_candidates(expr.right) or [""]
        return _dedupe_paths(normalize_path(left + right) for left in lefts for right in rights)
    return []


def _dedupe_paths(paths) -> list[str]:
    """Order-preserving dedupe; drops None (normalization failed) entries."""
    seen: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.append(path)
    return seen


def _template_raw(template: Any, leading_is_parameter: bool) -> str:
    """Raw text of a template literal; interpolations become {P} parameters.

    The first interpolation is ambiguous — it is a base-URL prefix in
    `${API_BASE}/todos` and a path parameter in `/${id}/todos`, which are
    structurally identical — so *leading_is_parameter* selects the reading:
    False drops it, True keeps it as a parameter. Later interpolations are
    always parameters.
    """
    pieces: list[str] = []
    expressions = list(getattr(template, "expressions", []) or [])
    for index, quasi in enumerate(getattr(template, "quasis", []) or []):
        pieces.append(quasi.value.raw or "")
        if index < len(expressions):
            if "".join(pieces):                    # path text precedes → mid-path parameter
                pieces.append(PATH_SLOT)
            elif leading_is_parameter:             # leading interpolation read as a parameter
                pieces.append(PATH_SLOT)
            # else: leading interpolation read as a base-URL prefix → dropped
    return "".join(pieces)
