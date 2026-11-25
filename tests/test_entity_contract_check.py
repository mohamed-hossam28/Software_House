"""Entity-contract check: generated ORM models, Pydantic schemas and by-id route
parameters agree with the architecture's Entity Contract (the source of truth).

Seam: core.static_checks.run_all_checks(files, manifest, architecture).
"""
from core.entity_contract import Entity, EntityField
from core.state import ArchData, CodeManifest, FindingSeverity
from core.static_checks import run_all_checks

ROUTER = "output/backend/app/routers/todos.py"
MODEL = "output/backend/app/models/todo.py"
SCHEMA = "output/backend/app/schemas/todo.py"


def _arch(pk_type: str = "uuid", backend: str = "FastAPI (Python 3.12)") -> ArchData:
    return ArchData(
        tech_stack={"backend": backend, "database": "SQLite 3"},
        entities=[Entity(name="Todo", table="todos", fields=[
            EntityField(name="id", type=pk_type, primary_key=True),
            EntityField(name="title", type="str", max_length=200),
            EntityField(name="done", type="bool"),
        ])],
    )


def _findings(files: dict[str, str], arch: ArchData):
    return [f for f in run_all_checks(files, CodeManifest(), arch) if f.check == "entity_contract"]


def _router(param: str) -> str:
    return (
        "import uuid\n"
        "from fastapi import APIRouter\n"
        "router = APIRouter(prefix='/todos')\n"
        "@router.get('/{todo_id}')\n"
        f"def get_todo(todo_id: {param}):\n"
        "    return {}\n"
    )


def test_route_param_disagreeing_with_uuid_primary_key_is_an_error_on_the_router():
    findings = _findings({ROUTER: _router("int")}, _arch("uuid"))
    assert len(findings) == 1
    assert findings[0].file == ROUTER
    assert findings[0].severity == FindingSeverity.ERROR
    assert findings[0].line == 5
    assert "todo_id" in findings[0].message and "uuid" in findings[0].message


def test_route_param_matching_the_primary_key_is_clean():
    assert _findings({ROUTER: _router("uuid.UUID")}, _arch("uuid")) == []
    assert _findings({ROUTER: _router("int")}, _arch("int")) == []


def test_non_fastapi_stack_has_no_findings():
    assert _findings({ROUTER: _router("int")}, _arch("uuid", backend="Django (Python 3.12)")) == []


def _model(id_column: str) -> str:
    return (
        "import uuid\n"
        "from sqlalchemy import Boolean, Column, Integer, String, Uuid\n"
        "from sqlalchemy.orm import Mapped, mapped_column\n"
        "from app.db import Base\n"
        "class Todo(Base):\n"
        "    __tablename__ = 'todos'\n"
        f"    {id_column}\n"
        "    title = Column(String(200), nullable=False)\n"
        "    done: Mapped[bool] = mapped_column(Boolean, default=False)\n"
    )


def test_orm_column_type_disagreeing_with_the_contract_is_an_error_on_the_model():
    findings = _findings({MODEL: _model("id = Column(Integer, primary_key=True)")}, _arch("uuid"))
    assert [(f.file, f.line, f.severity) for f in findings] == [(MODEL, 7, FindingSeverity.ERROR)]
    assert "Todo.id" in findings[0].message and "Uuid" in findings[0].message


def test_orm_mapped_annotation_without_column_type_is_checked():
    findings = _findings({MODEL: _model("id: Mapped[int] = mapped_column(primary_key=True)")}, _arch("uuid"))
    assert [(f.file, f.line) for f in findings] == [(MODEL, 7)]


def test_orm_columns_matching_the_contract_are_clean():
    assert _findings({MODEL: _model("id = Column(Uuid, primary_key=True, default=uuid.uuid4)")}, _arch("uuid")) == []
    assert _findings({MODEL: _model("id: Mapped[uuid.UUID] = mapped_column(primary_key=True)")}, _arch("uuid")) == []
    assert _findings({MODEL: _model("id = Column(Integer, primary_key=True)")}, _arch("int")) == []


def _schema(id_annotation: str) -> str:
    return (
        "import uuid\n"
        "from typing import Optional\n"
        "from pydantic import BaseModel, ConfigDict\n"
        "class TodoBase(BaseModel):\n"
        "    title: str\n"
        "    done: bool = False\n"
        "class TodoRead(TodoBase):\n"
        f"    id: {id_annotation}\n"
        "    model_config = ConfigDict(from_attributes=True)\n"
    )


def test_schema_field_disagreeing_with_the_contract_is_an_error_on_the_schema():
    findings = _findings({SCHEMA: _schema("int")}, _arch("uuid"))
    assert [(f.file, f.line, f.severity) for f in findings] == [(SCHEMA, 8, FindingSeverity.ERROR)]
    assert "TodoRead.id" in findings[0].message and "uuid.UUID" in findings[0].message


def test_schema_fields_matching_the_contract_are_clean():
    assert _findings({SCHEMA: _schema("uuid.UUID")}, _arch("uuid")) == []
    assert _findings({SCHEMA: _schema("Optional[int]")}, _arch("int")) == []


def test_request_schema_is_not_judged_against_the_contract():
    source = (
        "from pydantic import BaseModel\n"
        "class TodoFilter(BaseModel):\n"
        "    id: int\n"
    )
    assert _findings({SCHEMA: source}, _arch("uuid")) == []


def test_class_used_as_a_route_response_model_is_checked():
    schema = (
        "from pydantic import BaseModel\n"
        "class TodoRecord(BaseModel):\n"
        "    id: int\n"
    )
    router = (
        "import uuid\n"
        "from fastapi import APIRouter\n"
        "from app.schemas.todo import TodoRecord\n"
        "router = APIRouter(prefix='/todos')\n"
        "@router.get('/', response_model=list[TodoRecord])\n"
        "def list_todos():\n"
        "    return []\n"
    )
    findings = _findings({SCHEMA: schema, ROUTER: router}, _arch("uuid"))
    assert [(f.file, f.line) for f in findings] == [(SCHEMA, 3)]


def test_route_param_is_matched_to_the_entity_by_its_name_forms():
    arch = _arch("uuid")
    arch.entities[0].name, arch.entities[0].table = "TodoItem", "todo_items"
    findings = _findings({ROUTER: _router("int")}, arch)
    assert [(f.file, f.line) for f in findings] == [(ROUTER, 5)]


def test_matching_model_schema_and_router_together_are_clean():
    files = {
        MODEL: _model("id = Column(Uuid, primary_key=True, default=uuid.uuid4)"),
        SCHEMA: _schema("uuid.UUID"),
        ROUTER: _router("uuid.UUID"),
    }
    assert _findings(files, _arch("uuid")) == []
