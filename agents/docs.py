"""
agents/docs.py
──────────────
Docs Agent — writes the project README after Integration, from the FINAL
files (the coding agents' output with every Integration repair applied), so
the README documents the code that actually ships: its routes, health values
and run steps.

It is the backend Coding agent run on one file: the backend profile's system
prompt, the coding request header, the backend reply schema, and the loop's
write guard + record step. Only the request body differs: instead of planned
contracts it carries the content of every final file.
"""
from __future__ import annotations

import logging
from threading import Lock

from agents.backend import BACKEND_PROFILE
from agents.coding_loop import CodingAgent, request_header
from core.state import ArchData, FileManifest, ProjectState

logger = logging.getLogger(__name__)

README_PATH = "output/README.md"
README_TASK_ID = "docs-readme"
README_PURPOSE = (
    "Project README: what the project does, the API it exposes, and run instructions: "
    "install dependencies, start the backend with `uvicorn main:app` from backend/, and "
    "browse to the URL it serves (the backend serves the frontend at /). uvicorn only: "
    "no `python -m http.server`, no opening index.html directly."
)


def final_files(ps: ProjectState) -> dict[str, str]:
    """Every generated file's final content (Integration repairs applied), README excluded, by path."""
    files: dict[str, str] = {}
    for layer in ("database", "backend", "frontend"):
        files.update(getattr(ps.generated_code, layer, {}) or {})
    files.pop(README_PATH, None)
    return dict(sorted(files.items()))


def build_readme_request(arch: ArchData, files: dict[str, str]) -> str:
    """The README's coding request: header, tech stack, the final files' content, reply format."""
    stack = getattr(arch, "tech_stack", None) or {}
    tech = "\n".join(f"  {k}: {v}" for k, v in stack.items()) or "  (none)"
    blocks = "\n\n".join(f"### `{path}`\n```\n{code}\n```" for path, code in files.items())
    keys = ", ".join(BACKEND_PROFILE.reply_keys)
    parts = [
        "## Tech Stack\n" + tech,
        "## File to Generate\n"
        f"- **File Path**: `{README_PATH}`\n"
        f"- **Purpose**: {README_PURPOSE}",
        "## README Rules (MUST follow)\n"
        "The README documents the code below EXACTLY as written — never the plan, never "
        "assumptions. Route paths and prefixes, response values (e.g. health status "
        "strings), environment variables, ports and run commands must match these files "
        "character for character. Frontend run steps must keep the asset paths the "
        "files reference working.",
        "## Final Project Files\n" + (blocks or "(no files were generated)"),
        "---\n"
        f"Write the complete Markdown for `{README_PATH}` in the \"code\" key.\n"
        f"Respond with a single valid JSON object with keys: {keys}. "
        "No prose outside the JSON.\n```json\n",
    ]
    return request_header([README_PATH]) + "\n\n".join(parts)


class DocsAgent(CodingAgent):
    """Writes the README from the final, repaired files, through the backend profile."""

    profile = BACKEND_PROFILE

    def _work(self, ps: ProjectState) -> None:
        lock = Lock()
        ps.code_manifest.files[README_PATH] = FileManifest(
            agent=self.profile.agent, task_id=README_TASK_ID,
            filepath=README_PATH, purpose=README_PURPOSE,
        )
        system = self.profile.system_prompt(ps.architecture)
        request = build_readme_request(ps.architecture, final_files(ps))
        raw = self._ask(system, request, README_PATH, ps, lock)
        if raw is not None:
            self._accept(raw, README_PATH, ps, lock)
