"""
Scripts for the scripted fake LLM (providers/llm/scripted.py): the todo
skeleton the Planner is answered with, and helpers that build one scripted
fake per agent.
"""
from __future__ import annotations

from typing import Any, Iterable

from core.state import CodeManifest
from providers.llm.scripted import ScriptedLLMProvider

#: A fixed, self-consistent todo skeleton for the Planner's single call. It has
#: real import edges (the routes import the model and the schemas), a
#: frontend-to-frontend edge (the entry imports the component), and ONE phantom
#: import (the service imports a file not in the skeleton).
TODO_SKELETON: dict[str, Any] = {
    "skeleton": [
        {
            "id": "db-001", "agent": "database", "title": "Todo model",
            "description": "SQLAlchemy model for todo_items table.",
            "file": "app/models/todo.py",
            "exports": ["TodoItem", "Base"], "imports_from": {},
        },
        {
            "id": "be-001", "agent": "backend", "title": "Todo schemas",
            "description": "Pydantic request/response schemas.",
            "file": "app/schemas/todo.py",
            "exports": ["TodoItemCreate", "TodoItemResponse"], "imports_from": {},
        },
        {
            "id": "be-002", "agent": "backend", "title": "Todo routes",
            "description": "FastAPI CRUD router.",
            "file": "app/api/routes/todo.py",
            "exports": ["router"],
            "imports_from": {
                "app/models/todo.py": ["TodoItem"],
                "app/schemas/todo.py": ["TodoItemCreate", "TodoItemResponse"],
            },
        },
        {
            "id": "be-003", "agent": "backend", "title": "Todo service",
            "description": "Business logic (imports a file that does NOT exist).",
            "file": "app/services/todo.py",
            "exports": ["TodoService"],
            "imports_from": {"app/models/nonexistent.py": ["Missing"]},
        },
        {
            "id": "fe-001", "agent": "frontend", "title": "Todo list component",
            "description": "Renders the todo list.",
            "file": "assets/js/todoList.js",
            "exports": ["TodoList"], "imports_from": {},
        },
        {
            "id": "fe-002", "agent": "frontend", "title": "App entry",
            "description": "Frontend entry point.",
            "file": "assets/js/main.js",
            "exports": ["App"],
            "imports_from": {"assets/js/todoList.js": ["TodoList"]},
        },
    ]
}

#: The files the Planner's todo plan sends to the LLM (no Planned Content), by
#: owning Coding agent. Pinned project files and scaffolds included.
TODO_LLM_FILES: dict[str, list[str]] = {
    "database": ["output/backend/app/models/todo.py"],
    "backend": [
        "output/backend/app/api/routes/todo.py",
        "output/backend/app/schemas/todo.py",
        "output/backend/app/services/todo.py",
        "output/backend/main.py",
        "output/backend/requirements.txt",
    ],
    "frontend": [
        "output/frontend/assets/js/main.js",
        "output/frontend/assets/js/todoList.js",
        "output/frontend/index.html",
    ],
}

#: A stub reply per Coding agent, valid against that agent's response schema.
STUB_REPLIES: dict[str, dict[str, Any]] = {
    "database": {"migration_code": "LLM_WROTE_THIS = 1\n", "functions": []},
    "backend": {"code": "LLM_WROTE_THIS = 1\n", "classes": [], "functions": []},
    "frontend": {"code": "export const LLM_WROTE_THIS = 1;\n", "classes": [], "functions": []},
}


def scripted_planner(skeleton: dict[str, Any] = TODO_SKELETON) -> ScriptedLLMProvider:
    """The Planner's fake: its one call is answered with *skeleton*."""
    return ScriptedLLMProvider("planner", calls=[skeleton])


def scripted_coding_agent(agent: str, files: Iterable[str], reply: Any = None) -> ScriptedLLMProvider:
    """A Coding agent's fake answering each of *files* with *reply* (default: the agent's stub)."""
    reply = STUB_REPLIES[agent] if reply is None else reply
    return ScriptedLLMProvider(agent, files={path: reply for path in files})


def llm_files(manifest: CodeManifest, agent: str) -> list[str]:
    """The files a Code Manifest sends to the LLM for *agent*: its File Manifests without Planned Content."""
    return sorted(
        path for path, fm in manifest.files.items()
        if fm.agent == agent and fm.planned_content is None
    )
