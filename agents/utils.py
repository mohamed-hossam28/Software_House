"""
Utility functions for agents.
"""
import os
from pathlib import Path
from typing import Any

OUTPUT_DIR = "output"
_root_dir = Path(__file__).resolve().parent.parent


def as_list(value: Any) -> list[str]:
    """Coerce to list[str] — handles None and non-list values gracefully."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def write_to_disk(filepath: str, code: str, output_dir: str = OUTPUT_DIR) -> None:
    """
    Safely write generated code under output_dir, preserving sub-path structure.
    """
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith(output_dir + "/") and not rel.startswith(output_dir + "\\") and rel != output_dir:
        rel = os.path.join(output_dir, rel)
    abs_path = os.path.join(_root_dir, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(code)


def parse_metadata(
    data: dict[str, Any], default_return_type: str = "None"
) -> tuple[list[Any], list[Any]]:
    """
    Extract structured ClassName and FunctionName objects from an LLM response dict.
    Handles general classes/functions as well as DB manifest column schemas.
    """
    from core.state import Argument, ClassName, FunctionName

    classes: list[ClassName] = []
    # 1. Check for database manifest schema format
    manifest_data: dict = data.get("manifest") or {}
    if manifest_data and "table_name" in manifest_data:
        table_name = manifest_data["table_name"]
        columns = manifest_data.get("columns", [])
        args = []
        for col in columns:
            col_name = col.get("name", "")
            col_type = col.get("type", "unknown")
            default_val = ""
            if col.get("primary_key"):
                col_type += " (PK)"
            if col.get("foreign_key"):
                col_type += f" (FK -> {col['foreign_key']})"
            if col.get("nullable"):
                default_val = "NULL"
            args.append(Argument(name=col_name, type=col_type, default=default_val))
        classes.append(
            ClassName(
                name=table_name,
                arguments=args,
                docstring=f"Table schema for {table_name}",
            )
        )

    # 2. Standard class list
    for c in data.get("classes", []):
        try:
            args = [Argument(**a) for a in c.get("arguments", [])]
            classes.append(
                ClassName(
                    name=c["name"],
                    arguments=args,
                    docstring=c.get("docstring", ""),
                )
            )
        except Exception:
            pass

    # 3. Standard function list
    functions: list[FunctionName] = []
    for f in data.get("functions", []):
        try:
            args = [Argument(**a) for a in f.get("arguments", [])]
            functions.append(
                FunctionName(
                    name=f["name"],
                    arguments=args,
                    return_type=f.get("return_type", default_return_type),
                    docstring=f.get("docstring", ""),
                )
            )
        except Exception:
            pass

    return classes, functions


def resolve_stages(tasks: list[Any], use_priority: bool = False) -> list[list[Any]]:
    """
    Groups tasks into dependency-resolved execution stages for parallel processing.
    Tasks within the same stage share no intra-agent dependencies and are safe to run concurrently.
    If use_priority is True, tasks are grouped by priority ascending before topological sorting.
    """
    if not tasks:
        return []

    agent_ids = {t.id for t in tasks}

    if use_priority:
        priority_groups: dict[int, list[Any]] = {}
        for t in tasks:
            p = getattr(t, "priority", 0)
            priority_groups.setdefault(p, []).append(t)
        priority_keys = sorted(priority_groups.keys())
    else:
        priority_groups = {0: list(tasks)}
        priority_keys = [0]

    stages: list[list[Any]] = []
    visited: set[str] = set()

    for priority in priority_keys:
        remaining = list(priority_groups[priority])
        while remaining:
            ready: list[Any] = []
            for task in remaining:
                deps = [d for d in task.depends_on if d in agent_ids]
                if all(d in visited for d in deps):
                    ready.append(task)

            if not ready:
                # Cycle or unreachable dependencies — bundle remaining tasks to avoid deadlock
                stages.append(remaining)
                for t in remaining:
                    visited.add(t.id)
                break

            stages.append(ready)
            for t in ready:
                visited.add(t.id)
                remaining.remove(t)

    return stages


# Aliases for backward compatibility across agents
_as_list = as_list
_write_to_disk = write_to_disk
write_code_to_disk = write_to_disk
_parse_metadata = parse_metadata
