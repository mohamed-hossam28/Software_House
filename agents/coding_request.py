"""
agents/coding_request.py
────────────────────────
The coding request header: the fixed first lines of every request a Coding
agent sends to the LLM, single file or batch. It names the file asked for, one
line per file in a batch, so a reader (the scripted fake LLM) knows which
File Manifests a request is about without reading the prose that follows.

    <<<coding-request>>>
    file: output/backend/app/api/routes/todo.py
    <<<end-coding-request>>>

The format and its parser live here, and only here; the coding loop owns them.
"""
from __future__ import annotations

from typing import Iterable

HEADER_OPEN = "<<<coding-request>>>"
HEADER_CLOSE = "<<<end-coding-request>>>"
FILE_PREFIX = "file: "


def request_header(filepaths: Iterable[str]) -> str:
    """The header naming *filepaths*, followed by a blank line, to prepend to a coding request."""
    paths = list(filepaths)
    if not paths:
        raise ValueError("a coding request header names at least one file")
    for path in paths:
        if not path or path != path.strip() or "\n" in path:
            raise ValueError(f"not a header-safe file path: {path!r}")
    lines = [HEADER_OPEN, *(FILE_PREFIX + path for path in paths), HEADER_CLOSE]
    return "\n".join(lines) + "\n\n"


def parse_request_header(request: str) -> list[str]:
    """
    The file paths named by the header *request* starts with, in order.

    Returns [] when *request* does not start with a header (it is not a coding
    request). Raises ValueError when it starts with a malformed header.
    """
    lines = request.split("\n")
    if lines[0] != HEADER_OPEN:
        return []
    paths: list[str] = []
    for line in lines[1:]:
        if line == HEADER_CLOSE:
            if not paths:
                raise ValueError("coding request header names no file")
            return paths
        if not line.startswith(FILE_PREFIX):
            raise ValueError(f"malformed coding request header line: {line!r}")
        paths.append(line[len(FILE_PREFIX):])
    raise ValueError("coding request header is never closed")
