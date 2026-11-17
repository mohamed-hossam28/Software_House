"""
Schema ↔ Agent compatibility tests.
Validates that Pydantic schemas accept the data shapes LLMs actually return.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 1. ArchitectureSchema ─────────────────────────────────────────────────────
from agents.schemas.architecture import ArchitectureSchema, APIContractSchema

# Test: nested response_body (the bug that was breaking production)
data = {
    "tech_stack": {"backend": "FastAPI", "frontend": "React"},
    "system_components": ["API Server", "Frontend SPA"],
    "api_contracts": [
        {
            "method": "GET",
            "path": "/api/tasks",
            "auth_required": True,
            "description": "List tasks",
            "request_body": {},
            "response_body": {
                "tasks": [{"id": "string", "title": "string", "status": "string"}],
                "total": "integer",
            },
        },
        {
            "method": "GET",
            "path": "/api/data",
            "auth_required": True,
            "description": "Get data",
            "request_body": {},
            "response_body": {
                "data": {"id": "string", "title": "string", "user_id": "string"},
            },
        },
    ],
    "database_schema_overview": "users -> posts",
    "data_flow": "Client -> API -> DB",
    "deployment_strategy": "Docker Compose",
    "security_requirements": ["JWT"],
}
obj = ArchitectureSchema.model_validate(data)
assert obj.api_contracts[0].response_body["tasks"] == [{"id": "string", "title": "string", "status": "string"}]
print("[PASS] ArchitectureSchema — nested response_body (list value) accepted")

# Test: missing optional fields (LLM omits them)
minimal = {
    "tech_stack": {},
    "system_components": [],
    "api_contracts": [],
}
obj2 = ArchitectureSchema.model_validate(minimal)
assert obj2.database_schema_overview == ""
assert obj2.data_flow == ""
assert obj2.deployment_strategy == ""
print("[PASS] ArchitectureSchema — missing optional fields default to empty string")


# ── 2. TaskSchema ─────────────────────────────────────────────────────────────
from agents.schemas.planner import TaskSchema

# Test: LLM omits priority (agent overwrites it anyway)
task_data = {
    "id": "be-001",
    "agent": "backend",
    "title": "Create User model",
    "description": "Build the SQLAlchemy User model.",
    "depends_on": ["db-001"],
    "status": "pending",
    "file": "backend/app/models/user.py",
    "exports": ["User"],
}
task_obj = TaskSchema.model_validate(task_data)
assert task_obj.priority == 1  # default
print("[PASS] TaskSchema — missing priority defaults to 1")


# ── 3. DatabaseSchema ────────────────────────────────────────────────────────
from agents.schemas.database import DatabaseSchema, ColumnSchema

# Test: column without nullable/primary_key (LLM omits them)
col = ColumnSchema.model_validate({"name": "email", "type": "VARCHAR(255)"})
assert col.nullable is True
assert col.primary_key is False
print("[PASS] ColumnSchema — missing nullable/primary_key get safe defaults")

# Test: manifest is Optional
db_data = {
    "migration_code": "CREATE TABLE users (id UUID PRIMARY KEY);",
    "functions": [],
}
db_obj = DatabaseSchema.model_validate(db_data)
assert db_obj.manifest is None
assert db_obj.migration_code.startswith("CREATE TABLE")
print("[PASS] DatabaseSchema — missing manifest accepted (Optional)")

# Test: manifest with full data still works
db_data_full = {
    "migration_code": "CREATE TABLE users (...);",
    "manifest": {
        "table_name": "users",
        "columns": [
            {"name": "id", "type": "UUID", "nullable": False, "primary_key": True},
            {"name": "email", "type": "VARCHAR(255)"},  # omits nullable, primary_key
        ],
        "indexes": ["email"],
        "exports": ["users"],
        "suggested_additions": [],
    },
    "functions": [],
}
db_obj2 = DatabaseSchema.model_validate(db_data_full)
assert db_obj2.manifest.table_name == "users"
assert db_obj2.manifest.columns[1].nullable is True      # default
assert db_obj2.manifest.columns[1].primary_key is False   # default
print("[PASS] DatabaseSchema — full manifest with column defaults works")


# ── 4. BackendSchema ─────────────────────────────────────────────────────────
from agents.schemas.backend import BackendSchema

be_data = {
    "code": "class User:\n    pass\n",
    "classes": [
        {"name": "User", "arguments": [{"name": "id", "type": "int"}], "docstring": "User model"}
    ],
    "functions": [
        {"name": "get_user", "arguments": [], "return_type": "User", "docstring": "Fetch user"}
    ],
}
be_obj = BackendSchema.model_validate(be_data)
assert be_obj.code.startswith("class User")
assert be_obj.classes[0].arguments[0].default == ""  # default
print("[PASS] BackendSchema — full validation with argument defaults")


# ── 5. FrontendSchema ────────────────────────────────────────────────────────
from agents.schemas.frontend import FrontendSchema

fe_data = {
    "code": "export function LoginPage() { return <div>Login</div>; }",
    "classes": [],
    "functions": [
        {"name": "LoginPage", "arguments": [], "return_type": "JSX.Element", "docstring": "Login page"}
    ],
}
fe_obj = FrontendSchema.model_validate(fe_data)
assert fe_obj.functions[0].name == "LoginPage"
print("[PASS] FrontendSchema — function-only response accepted")


# ── 6. RequirementsSchema ────────────────────────────────────────────────────
from agents.schemas.requirements import RequirementsSchema

req_data = {
    "functional_requirements": ["User can register"],
    "non_functional_requirements": ["Response < 200ms"],
    "constraints": ["Must use PostgreSQL"],
    "assumptions": ["Single tenant"],
    "out_of_scope": ["Mobile app"],
    "clarifying_questions": [],
}
req_obj = RequirementsSchema.model_validate(req_data)
assert len(req_obj.functional_requirements) == 1
print("[PASS] RequirementsSchema — full happy path")

# Test: all fields optional (LLM returns partial)
req_partial = RequirementsSchema.model_validate({})
assert req_partial.functional_requirements == []
print("[PASS] RequirementsSchema — empty response accepted (all defaults)")


print("\n" + "=" * 60)
print("All schema validation tests PASSED")
print("=" * 60)
