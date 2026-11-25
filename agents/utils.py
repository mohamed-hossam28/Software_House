"""
Utility functions for agents.
"""
import hashlib
import os
from typing import Any

from core.workspace import Workspace


def as_list(value: Any) -> list[str]:
    """Coerce to list[str] — handles None and non-list values gracefully."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def sha256_of(code: str) -> str:
    """Stable content fingerprint for run summaries."""
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def static_checks_snapshot(state: Any) -> dict[str, Any]:
    """Canonical serializable snapshot used by reports and human summaries."""
    static_checks = getattr(state, "static_checks", None)
    findings = list(getattr(static_checks, "findings", None) or [])

    def enum_str(value: Any) -> str:
        return getattr(value, "value", str(value or "")) or ""

    serialized = [
        {
            "file": getattr(finding, "file", ""),
            "check": getattr(finding, "check", ""),
            "severity": enum_str(getattr(finding, "severity", "")),
            "line": getattr(finding, "line", 0),
            "message": getattr(finding, "message", ""),
        }
        for finding in findings
    ]
    error_count = sum(1 for finding in serialized if finding["severity"] == "error")
    warn_count = sum(1 for finding in serialized if finding["severity"] == "warn")
    repaired_files = list(getattr(static_checks, "repaired_files", None) or [])
    return {
        "verdict": enum_str(getattr(static_checks, "verdict", "")),
        "findings_count": len(serialized),
        "errors": error_count,
        "warnings": warn_count,
        "repair_rounds": int(getattr(static_checks, "rounds", 0) or 0),
        "repaired_count": len(repaired_files),
        "repaired_files": repaired_files,
        "findings": serialized,
    }


def format_static_checks_summary(state: Any) -> str:
    """One-line final verdict with finding and repair counts."""
    checks = static_checks_snapshot(state)
    return (
        f"Integration verdict: {checks['verdict']} "
        f"(findings: {checks['findings_count']}, errors: {checks['errors']}, "
        f"warnings: {checks['warnings']}, repair rounds: {checks['repair_rounds']}, "
        f"repaired files: {checks['repaired_count']})"
    )


def write_static_checks_report(state: Any, workspace: Workspace | None = None) -> str:
    """Write every final static finding to `output/static_checks_report.json` in *workspace*."""
    try:
        import json
        from datetime import datetime, timezone

        checks = static_checks_snapshot(state)

        report = {
            "project_id": getattr(state, "project_id", ""),
            "project_name": getattr(state, "project_name", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "verdict": checks["verdict"],
            "counts": {
                "findings": checks["findings_count"],
                "errors": checks["errors"],
                "warnings": checks["warnings"],
                "repair_rounds": checks["repair_rounds"],
                "repaired_files": checks["repaired_count"],
            },
            "repaired_files": checks["repaired_files"],
            "findings": checks["findings"],
        }

        rel = "output/static_checks_report.json"   # workspace-relative, forward slashes on every OS
        abs_path = (workspace or Workspace()).abs_path(rel)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return rel
    except Exception as exc:  # pragma: no cover — reporting must never break a run
        import logging
        logging.getLogger(__name__).warning("write_static_checks_report failed: %s", exc)
        return ""


def write_run_summary(
    state: Any, static_report_path: str = "", workspace: Workspace | None = None
) -> str:
    """
    Persist a run summary next to the generated project (`output/run_summary.json`)
    so the deliverable on disk can be diffed against the plan. Never raises.
    """
    try:
        import json
        from datetime import datetime, timezone

        files: dict[str, dict] = {}
        code_map = getattr(state, "generated_code", None)
        buckets = code_map.all_files() if code_map is not None else {}

        manifest = getattr(state, "code_manifest", None)
        for fp, content in buckets.items():
            fm = (manifest.files.get(fp) if manifest is not None else None)
            entry = {
                "agent": fm.agent if fm else "",
                "task_id": fm.task_id if fm else "",
                "exports": list(fm.exports or []) if fm else [],
                "written": bool(fm.written if fm else True),
                "sha256": sha256_of(content),
                "lines": len((content or "").splitlines()),
            }
            files[fp] = entry

        checks = static_checks_snapshot(state)

        summary = {
            "project_id": getattr(state, "project_id", ""),
            "project_name": getattr(state, "project_name", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": getattr(state, "status", ""),
            "static_checks": {**checks, "report_path": static_report_path},
            "files": files,
            "errors": [
                {"agent": e.get("agent", ""), "message": e.get("message", ""), "ts": str(e.get("ts", ""))}
                for e in (getattr(state, "errors", None) or [])
            ],
            "metrics": {
                "total_llm_calls": getattr(getattr(state, "metrics", None), "total_llm_calls", 0),
                "total_tokens_used": getattr(getattr(state, "metrics", None), "total_tokens_used", 0),
            },
        }

        rel = "output/run_summary.json"   # workspace-relative, forward slashes on every OS
        abs_path = (workspace or Workspace()).abs_path(rel)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return rel
    except Exception as exc:  # pragma: no cover — summary must never break a run
        import logging
        logging.getLogger(__name__).warning("write_run_summary failed: %s", exc)
        return ""



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
    if isinstance(manifest_data, dict) and isinstance(manifest_data.get("table_name"), str):
        table_name = manifest_data["table_name"]
        columns = manifest_data.get("columns", [])
        args = []
        for col in columns:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name", "")
            col_type = col.get("type", "unknown")
            if not isinstance(col_name, str):
                col_name = str(col_name) if col_name is not None else ""
            if not isinstance(col_type, str):
                col_type = "unknown"
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

