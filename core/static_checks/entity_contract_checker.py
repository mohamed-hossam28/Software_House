"""
Entity-contract check: generated code agrees with the architecture's Entity
Contract (`ArchData.entities`), the source of truth. Each file is checked
against the contract, never model-vs-schema.

Checked (v1: SQLAlchemy + Pydantic + FastAPI only; any other backend stack is
skipped with no findings):
- ORM column types of a SQLAlchemy model (matched by `__tablename__`), from the
  column type argument or, when absent, the `Mapped[...]` annotation;
- Pydantic response schema field annotations: a BaseModel subclass a route
  returns (`response_model=`, in any file) or named as a response schema
  (`TodoRead`, `TodoOut`, ...), matched to its entity by name. Request schemas
  are not judged;
- a FastAPI by-id route's path parameter annotation against the primary key's
  route-parameter type (`todo_id`, or `id` after `/todos/`).

Names are matched to entities with `core.entity_contract.match_entity`, the
same matching the planner uses for field contracts.

Types are compared through `core.entity_contract.TYPE_MAPPINGS`, by the last
dotted component of an annotation (`uuid.UUID` and `UUID` are the same type).
An annotation that names none of the contract's types (e.g. `EmailStr`) is not
judged. A mismatch is an ERROR on the disagreeing file so integration repair
regenerates that file. Files that do not parse are skipped (the Python checker
reports them).
"""
import ast
import re
from typing import Any, Optional

from core.entity_contract import TYPE_MAPPINGS, Entity, is_response_schema, match_entity
from core.state import FindingSeverity, StaticFinding

CHECK = "entity_contract"
_ROUTE_DECORATORS = {"get", "post", "put", "delete", "patch"}


def check_entity_contract(
    files: dict[str, str],
    manifest: Any,
    architecture: Any = None,
) -> list[StaticFinding]:
    """Error for every ORM column, schema field or route parameter that disagrees with the Entity Contract."""
    entities: list[Entity] = list(getattr(architecture, "entities", None) or [])
    if not entities or not _is_fastapi(architecture):
        return []
    trees: dict[str, ast.Module] = {}
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        try:
            trees[path] = ast.parse(source)
        except SyntaxError:
            continue
    response_models = _response_models(trees)
    findings: list[StaticFinding] = []
    for path, tree in trees.items():
        findings.extend(_check_orm(path, tree, entities))
        findings.extend(_check_schemas(path, tree, entities, response_models))
        findings.extend(_check_routes(path, tree, entities))
    return findings


def _is_fastapi(architecture: Any) -> bool:
    stack = getattr(architecture, "tech_stack", None) or {}
    return "fastapi" in str(stack.get("backend", "")).lower()


# ── type tokens ──────────────────────────────────────────────────────────────

def _token(code_type: Optional[str]) -> Optional[str]:
    """Last dotted component of a code type: `uuid.UUID` -> `UUID`."""
    return code_type.rsplit(".", 1)[-1] if code_type else None


def _annotation_token(node: Optional[ast.expr]) -> Optional[str]:
    """Type token of an annotation, unwrapping Optional[X], X | None, Union[X, None]."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return _annotation_token(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        parts = [p for p in (node.left, node.right) if not _is_none(p)]
        return _annotation_token(parts[0]) if len(parts) == 1 else None
    if isinstance(node, ast.Subscript):
        outer = _name(node.value)
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if outer == "Optional":
            return _annotation_token(args[0])
        if outer == "Union":
            parts = [a for a in args if not _is_none(a)]
            return _annotation_token(parts[0]) if len(parts) == 1 else None
        if outer == "Annotated":
            return _annotation_token(args[0])
        return outer
    return _name(node)


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _disagrees(found: Optional[str], expected: set[str], vocabulary: set[str]) -> bool:
    """True when `found` is one of the contract's code types (*vocabulary*) but not an
    expected one. A type outside the vocabulary (e.g. `EmailStr`) is never judged."""
    return found is not None and found in vocabulary and found not in expected


_ROUTE_TOKENS = {_token(m.route_param) for m in TYPE_MAPPINGS.values() if m.route_param}


_PYDANTIC_TOKENS = {_token(m.pydantic) for m in TYPE_MAPPINGS.values()}

#: Column types accepted for a field type beyond its canonical TYPE_MAPPINGS one.
#: str and text share their string types: a String-vs-Text choice breaks nothing.
_SQLALCHEMY_ALIASES: dict[str, set[str]] = {
    "int": {"BigInteger", "SmallInteger"},
    "uuid": {"UUID"},
    "str": {"Text", "Unicode", "UnicodeText", "VARCHAR"},
    "text": {"String", "Unicode", "UnicodeText", "VARCHAR"},
    "float": {"Numeric", "REAL"},
    "datetime": {"TIMESTAMP"},
    "json": {"JSONB"},
}


def _sqlalchemy_types(field_type: str) -> set[str]:
    return {TYPE_MAPPINGS[field_type].sqlalchemy} | _SQLALCHEMY_ALIASES.get(field_type, set())


_SQLALCHEMY_TOKENS = set().union(*(_sqlalchemy_types(t) for t in TYPE_MAPPINGS))


# ── ORM columns ──────────────────────────────────────────────────────────────

def _class_assigns(cls: ast.ClassDef):
    """(name, annotation, value, line) for each simple attribute assignment in a class body."""
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            yield stmt.targets[0].id, None, stmt.value, stmt.lineno
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            yield stmt.target.id, stmt.annotation, stmt.value, stmt.lineno


def _orm_entity(cls: ast.ClassDef, entities: list[Entity]) -> Optional[Entity]:
    """The entity an ORM class maps (by `__tablename__`); None for a non-ORM class."""
    for name, _, value, _ in _class_assigns(cls):
        if name == "__tablename__" and isinstance(value, ast.Constant):
            table = value.value
            return next((e for e in entities if e.table == table or e.name == cls.name), None)
    return None


def _column_type(call: ast.Call) -> Optional[str]:
    """The column type token of `Column(...)` / `mapped_column(...)`; None when not given."""
    args = list(call.args)
    if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str):
        args = args[1:]  # Column("name", Type, ...)
    if not args:
        return None
    first = args[0]
    return _name(first.func) if isinstance(first, ast.Call) else _name(first)


def _mapped_token(annotation: Optional[ast.expr]) -> Optional[str]:
    """Type token inside `Mapped[X]`."""
    if isinstance(annotation, ast.Subscript) and _name(annotation.value) == "Mapped":
        return _annotation_token(annotation.slice)
    return None


def _check_orm(path: str, tree: ast.Module, entities: list[Entity]) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        entity = _orm_entity(cls, entities)
        if entity is None:
            continue
        fields = {f.name: f for f in entity.fields}
        for name, annotation, value, line in _class_assigns(cls):
            field = fields.get(name)
            if field is None or not (isinstance(value, ast.Call) and _name(value.func) in ("Column", "mapped_column")):
                continue
            column = _column_type(value)
            if column is not None:
                bad = _disagrees(column, _sqlalchemy_types(field.type), _SQLALCHEMY_TOKENS)
                found = column
            else:
                found = _mapped_token(annotation)
                bad = _disagrees(found, {_token(TYPE_MAPPINGS[field.type].pydantic)}, _PYDANTIC_TOKENS)
            if bad:
                findings.append(StaticFinding(
                    file=path, check=CHECK, severity=FindingSeverity.ERROR, line=line,
                    message=(
                        f"ORM column `{cls.name}.{name}` is `{found}` but the Entity Contract "
                        f"says {entity.name}.{name} is {field.type} "
                        f"(SQLAlchemy `{TYPE_MAPPINGS[field.type].sqlalchemy}`)"
                    ),
                ))
    return findings


# ── Pydantic response schemas ────────────────────────────────────────────────

def _schema_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """Pydantic model classes: subclasses of BaseModel, directly or via a schema in the same file."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    schema_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for cls in classes:
            if cls.name not in schema_names and any(
                _name(b) == "BaseModel" or _name(b) in schema_names for b in cls.bases
            ):
                schema_names.add(cls.name)
                changed = True
    return [c for c in classes if c.name in schema_names]


def _response_models(trees: dict[str, ast.Module]) -> set[str]:
    """Class names any route returns: its `response_model=` (unwrapping `list[X]` etc.)."""
    names: set[str] = set()
    for tree in trees.values():
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in func.decorator_list:
                if _route_path(decorator) is None:
                    continue
                for kw in decorator.keywords:
                    if kw.arg == "response_model":
                        names.update(_name(n) for n in ast.walk(kw.value) if _name(n))
    return names


def _response_entity(
    cls: ast.ClassDef, response_models: set[str], entities: dict[str, Entity]
) -> Optional[Entity]:
    """The entity a response schema returns; None for a class no route returns that is not
    named as a response schema (`TodoRead`, `TodoOut`, ...), or that matches no one entity."""
    if cls.name not in response_models and not is_response_schema(cls.name):
        return None
    name = match_entity(cls.name, entities)
    return entities[name] if name else None


def _check_schemas(
    path: str, tree: ast.Module, entities: list[Entity], response_models: set[str]
) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    by_name = {e.name: e for e in entities}
    for cls in _schema_classes(tree):
        entity = _response_entity(cls, response_models, by_name)
        if entity is None:
            continue
        fields = {f.name: f for f in entity.fields}
        for name, annotation, _, line in _class_assigns(cls):
            field = fields.get(name)
            if field is None or annotation is None:
                continue
            expected = TYPE_MAPPINGS[field.type].pydantic
            found = _annotation_token(annotation)
            if _disagrees(found, {_token(expected)}, _PYDANTIC_TOKENS):
                findings.append(StaticFinding(
                    file=path, check=CHECK, severity=FindingSeverity.ERROR, line=line,
                    message=(
                        f"Schema field `{cls.name}.{name}: {found}` disagrees with the Entity "
                        f"Contract: {entity.name}.{name} is {field.type} (annotate it `{expected}`)"
                    ),
                ))
    return findings


# ── route parameters ─────────────────────────────────────────────────────────

def _primary_key(entity: Entity):
    return next((f for f in entity.fields if f.primary_key), None)


def _entity_named(name: str, entities: list[Entity]) -> Optional[Entity]:
    """The one entity *name* matches by entity name, else by table name."""
    for key in ("name", "table"):
        by_key = {getattr(e, key): e for e in entities}
        match = match_entity(name, by_key)
        if match is not None:
            return by_key[match]
    return None


def _entity_for_param(param: str, path: str, entities: list[Entity]) -> Optional[Entity]:
    """The entity whose primary key a path parameter names: `todo_id`, or `id` after `/todos/`."""
    if param.endswith("_id"):
        return _entity_named(param[: -len("_id")], entities)
    if param in ("id", "pk"):
        segments = [s for s in path.split("/") if s]
        index = segments.index("{" + param + "}") if "{" + param + "}" in segments else -1
        previous = segments[index - 1] if index > 0 else ""
        return _entity_named(previous, entities) if previous else None
    return None


def _route_path(decorator: ast.expr) -> Optional[str]:
    if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
        return None
    if decorator.func.attr not in _ROUTE_DECORATORS or not decorator.args:
        return None
    first = decorator.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def _check_routes(path: str, tree: ast.Module, entities: list[Entity]) -> list[StaticFinding]:
    findings: list[StaticFinding] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        route_paths = [p for p in map(_route_path, func.decorator_list) if p is not None]
        if not route_paths:
            continue
        args = {a.arg: a for a in func.args.args + func.args.kwonlyargs}
        for route_path in route_paths:
            for param in re.findall(r"{(\w+)(?::\w+)?}", route_path):
                entity = _entity_for_param(param, route_path, entities)
                pk = _primary_key(entity) if entity else None
                arg = args.get(param)
                if pk is None or arg is None:
                    continue
                expected = TYPE_MAPPINGS[pk.type].route_param
                found = _annotation_token(arg.annotation)
                if _disagrees(found, {_token(expected)}, _ROUTE_TOKENS):
                    findings.append(StaticFinding(
                        file=path, check=CHECK, severity=FindingSeverity.ERROR,
                        line=arg.lineno,
                        message=(
                            f"Route parameter `{param}: {found}` disagrees with the Entity "
                            f"Contract: {entity.name}.{pk.name} is {pk.type} "
                            f"(annotate it `{expected}`)"
                        ),
                    ))
    return findings
