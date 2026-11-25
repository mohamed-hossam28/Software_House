"""
agents/frontend.py
──────────────────
Frontend Agent — a Coding agent: a name plus its profile. The coding loop
(agents/coding_loop.py) does the work; this module holds what is frontend-only:
the system prompt (frontend tech) and the frontend sections of a file's
request (stylesheet note, API contracts).
"""
from __future__ import annotations

from typing import Any

from agents.coding_loop import CodingAgent, CodingProfile, FileContext, filter_api_contracts_by_relevance
from agents.schemas.frontend import FrontendSchema

DEFAULT_FRONTEND_TECH = "React 18 + TypeScript"

_STYLE_EXTENSIONS = (".css", ".scss", ".sass", ".less")


def _frontend_tech(arch: Any) -> str:
    stack = getattr(arch, "tech_stack", None) or {}
    return stack.get("frontend", DEFAULT_FRONTEND_TECH)


def _build_system_prompt(frontend_tech: str) -> str:
    return f"""\
You are an expert frontend engineer specialising in {frontend_tech}.
Generate the COMPLETE source code for a single frontend file plus its manifest metadata.

## Non-negotiable rules
1. Return ONE valid JSON object — no prose outside the JSON.
2. The "code" field must be production-ready, complete, and syntax-valid. No placeholders, no TODOs.
3. Export every name listed in "Required Exports" with the exact signature implied by the manifest.
4. Only import from file paths listed in the Project Code Manifest.
5. Use the exact API endpoint paths, HTTP methods, request_body, and response_body fields from
   the API Contracts section — do NOT invent or rename fields.
6. Generate real, complete CSS/styling (not inline placeholders). If the file is a stylesheet,
   write a full stylesheet with variables, reset, typography, layout, and component classes.
   If it is a component, embed a <style> block or import the stylesheet — never leave styles empty.
7. Use {frontend_tech} idioms and best practices throughout.
8. Do not hardcode secrets; read from environment variables or config files.
9. The backend serves the frontend directory at `/`. HTML links its local assets with relative
   paths to the files in the manifest (href="style.css", src="app.js") — never `/static/...`.

Return JSON: {{"code":"<full source>","classes":[{{"name":"<class name>","arguments":[{{"name":"<param>","type":"<type>","default":""}}],"docstring":"<desc>"}}],"functions":[{{"name":"<func name>","arguments":[{{"name":"<param>","type":"<type>","default":""}}],"return_type":"<type>","docstring":"<desc>"}}]}}
"""


def _api_contracts(ctx: FileContext) -> str:
    contracts = filter_api_contracts_by_relevance(
        getattr(ctx.arch, "api_contracts", None) or [],
        filepath=ctx.filepath,
        purpose=ctx.fm.purpose,
        task_title=ctx.task.title if ctx.task else "",
    )
    if not contracts:
        return "  (none)"
    lines = []
    for c in contracts:
        lock = "🔒" if c.auth_required else "🔓"
        lines.append(
            f"  {lock} {c.method:7} {c.path}\n"
            f"     Description : {c.description}\n"
            f"     request_body : {c.request_body}\n"
            f"     response_body: {c.response_body}"
        )
    return "\n".join(lines)


def _sections(ctx: FileContext) -> list[str]:
    sections = [f"## Framework\n{_frontend_tech(ctx.arch)}"]
    if ctx.filepath.endswith(_STYLE_EXTENSIONS):
        sections.append(
            "⚠️  This is a STYLESHEET file. Write a complete, production-quality stylesheet:\n"
            "   - CSS custom properties (variables) for colours, spacing, typography\n"
            "   - Normalise / reset block\n"
            "   - Typography scale\n"
            "   - Layout helpers (flex/grid utilities)\n"
            "   - Component classes that match the component files in the manifest\n"
            "   - Responsive breakpoints\n"
            "   No empty rule-sets, no placeholder comments."
        )
    sections.append("## API Contracts  ← USE THESE EXACT PATHS / FIELD NAMES\n" + _api_contracts(ctx))
    return sections


FRONTEND_PROFILE = CodingProfile(
    agent="frontend",
    code_key="code",
    response_schema=FrontendSchema,
    system_prompt=lambda arch: _build_system_prompt(_frontend_tech(arch)),
    sections=_sections,
    tech=_frontend_tech,
    default_return_type="void",
)


class FrontendAgent(CodingAgent):
    """The frontend Coding agent."""
    profile = FRONTEND_PROFILE
