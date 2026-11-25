"""
HTML asset-link check: every local href/src in a written HTML file must resolve
to a written frontend file.

The backend serves the frontend directory (the frontend Layer root) at `/`, so
the served root is `output/frontend/`: a root-absolute link (`/app.js`) resolves
against it, and a relative link (`style.css`, `./app.js`, `../x.css`) resolves
against the HTML file's own directory. External URLs (any scheme or `//host`),
fragments, `mailto:`/`data:`/`javascript:` links and template placeholders are
ignored; query strings and fragments are stripped before resolving.

A link that resolves to no written file is an error finding on the HTML file,
so the repair loop routes it to the file's manifest owner (the frontend agent).

The manifest and architecture parameters are part of the uniform checker
signature and are unused.
"""
import posixpath
from html.parser import HTMLParser
from typing import Any

from core.state import FindingSeverity, StaticFinding
from core.workspace import FRONTEND_ROOT, workspace_path

_LINK_ATTRS = {"href", "src"}


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in _LINK_ATTRS and value:
                self.links.append((self.getpos()[0], value.strip()))

    handle_startendtag = handle_starttag


def _is_local(link: str) -> bool:
    if not link or link.startswith(("#", "//")):
        return False
    if "{" in link or "<" in link:  # template placeholders ({{ x }}, ${x}, <%= x %>)
        return False
    scheme, sep, _ = link.partition(":")
    if sep and "/" not in scheme:  # http:, https:, mailto:, data:, javascript:, ...
        return False
    return True


def _resolve(html_path: str, link: str) -> str:
    target = link.split("#", 1)[0].split("?", 1)[0]
    if target.startswith("/"):
        joined = FRONTEND_ROOT + target.lstrip("/")
    else:
        joined = posixpath.join(posixpath.dirname(html_path), target)
    return posixpath.normpath(joined)


def check_html_asset_links(
    files: dict[str, str],
    manifest: Any = None,
    architecture: Any = None,
) -> list[StaticFinding]:
    """Flag every local href/src in written HTML that resolves to no written file."""
    written = {posixpath.normpath(workspace_path(p)) for p in files}
    findings: list[StaticFinding] = []
    for filepath, code in sorted(files.items()):
        if not filepath.endswith((".html", ".htm")) or not code:
            continue
        html_path = workspace_path(filepath)
        collector = _LinkCollector()
        try:
            collector.feed(code)
            collector.close()
        except Exception:  # malformed markup: report nothing rather than guess
            continue
        for line, link in collector.links:
            if not _is_local(link):
                continue
            target = _resolve(html_path, link)
            if target in written or target.rstrip("/") + "/index.html" in written:
                continue
            findings.append(StaticFinding(
                file=filepath,
                check="html_asset_links",
                severity=FindingSeverity.ERROR,
                line=line,
                message=(
                    f"'{link}' resolves to '{target}', which is not a written file. "
                    f"The frontend directory is served at '/': use a relative link "
                    f"to a file under {FRONTEND_ROOT} (e.g. 'style.css')."
                ),
            ))
    return findings
