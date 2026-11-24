"""
Integration Agent — static checks + bounded LLM repair loop.

Collects every file the coding agents wrote (generated_code buckets), runs the
checker seam (core.static_checks.run_all_checks) over them, and repairs:

  - error findings are grouped by owning agent through the code manifest,
    then by file (one repair call handles one file)
  - each failing file is sent to the LLM in a repair prompt holding the file
    content, all its findings with line numbers, and the relevant contract
    excerpt (API contract for HTTP issues, manifest field contract for
    schema issues)
  - the repaired file is written through the existing safe-write guard,
    replaces its bucket entry, and is re-checked on its own
  - maximum MAX_REPAIR_ROUNDS rounds per file — a failed attempt (LLM error,
    unusable reply, refused write) consumes a round but the file keeps its
    remaining allowance; warning findings never trigger repair; files that
    survive every round are logged to the state error list with
    file/line/message

Outcome (state.static_checks):

  - verdict  → pass (no errors) / repaired (every failing file healed) / failed
  - findings → the final checker findings (repaired files' errors dropped)
  - rounds   → total repair rounds performed
  - repaired_files → files that ended their rounds with zero error findings

Repair LLM calls go through the shared _call_with_system helper, so their
tokens/call counts land in the same state metrics as every other agent.

Follows the base-agent contract: accepts the full state (dict or ProjectState),
returns only the changed fields as a diff dict.
"""
import logging
import os
import threading
import time
from typing import Any

from agents.base import (
    BaseAgent,
    agent_refuse_corrupt,
    build_import_contract,
    filter_api_contracts_by_relevance,
)
from core.state import (
    FindingSeverity,
    ProjectState,
    StaticChecks,
    StaticFinding,
    StaticVerdict,
    backend_persistence_rules,
    render_persistence_contract,
)
from core.static_checks import run_all_checks
from core.static_checks.persistence_checker import PERSISTENCE_CHECKS

logger = logging.getLogger(__name__)

#: Maximum repair rounds per file. An unrepairable file stops after this many.
MAX_REPAIR_ROUNDS = 2

#: Checks whose repair prompt gets the API contract excerpt (HTTP surface issues).
_HTTP_CHECKS = {"js_urls", "backend_routes"}
_SCHEMA_CHECKS = {"sql_tables", "sql_columns"}

_REPAIR_SYSTEM_PROMPT = (
    "You are the Integration Agent's code repair step. You receive ONE generated "
    "file, the static-check findings against it, and the relevant contracts. "
    "Fix every finding while changing as little as possible. Reply with a single "
    "JSON object: {\"filepath\": \"<path>\", \"code\": \"<full corrected file>\"}."
)


class IntegrationAgent(BaseAgent):
    """Static-check agent: verifies generated code and repairs error findings via the LLM."""

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Integration Agent. You verify generated code statically "
            "and perform bounded, finding-driven repairs."
        )

    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)

        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        files: dict[str, str] = {}
        for layer in ("backend", "frontend", "database"):
            files.update(getattr(ps.generated_code, layer, {}) or {})

        dialect = str((ps.architecture.tech_stack or {}).get("database", "") or "")

        try:
            findings = run_all_checks(files, ps.code_manifest, ps.architecture, dialect)
        except Exception as exc:
            logger.exception("[IntegrationAgent] static checks crashed")
            findings = [StaticFinding(
                check="static_checks",
                severity=FindingSeverity.ERROR,
                message=f"Static checks failed: {exc}",
            )]

        errors = [f for f in findings if f.severity == FindingSeverity.ERROR]
        verdict = StaticVerdict.PASS
        rounds = 0
        repaired_files: list[str] = []
        final_findings = findings

        if errors:
            logger.info(
                "[IntegrationAgent] %d error finding(s) — starting repair loop",
                len(errors),
            )
            _, rounds, repaired_files, survivors_by_file = self._repair_loop(
                ps, files, errors, dialect
            )
            # Individual rounds re-check only the repaired file's findings while
            # retaining the full project as dependency/schema context. After all
            # sequential repairs, this authoritative project check refreshes
            # cross-file errors and warnings before deriving the final verdict.
            try:
                final_findings = run_all_checks(
                    files, ps.code_manifest, ps.architecture, dialect
                )
            except Exception as exc:
                logger.exception("[IntegrationAgent] final static checks crashed")
                final_findings = [
                    finding
                    for file_findings in survivors_by_file.values()
                    for finding in file_findings
                ]
                final_findings.append(StaticFinding(
                    check="static_checks",
                    severity=FindingSeverity.ERROR,
                    message=f"Final static checks failed: {exc}",
                ))

            final_error_files = {
                f.file
                for f in final_findings
                if f.severity == FindingSeverity.ERROR and f.file
            }
            repaired_files = [
                filepath
                for filepath in repaired_files
                if filepath not in final_error_files
            ]

            verdict = (
                StaticVerdict.FAILED
                if any(f.severity == FindingSeverity.ERROR for f in final_findings)
                else StaticVerdict.REPAIRED
            )

        survivors = [
            f for f in final_findings if f.severity == FindingSeverity.ERROR
        ]
        warns = [
            f for f in final_findings if f.severity == FindingSeverity.WARN
        ]

        ps.static_checks = StaticChecks(
            verdict=verdict,
            findings=final_findings,
            rounds=rounds,
            repaired_files=repaired_files,
        )

        for f in survivors:
            location = f"{f.file}:{f.line}" if f.line else f.file
            ps.log_error(self.name, f"{f.check}: {location} {f.message}".strip())

        ps.metrics.agent_durations[self.name] = round(time.perf_counter() - t0, 3)
        logger.info(
            "[IntegrationAgent] verdict=%s errors=%d warnings=%d rounds=%d repaired=%d files=%d",
            ps.static_checks.verdict.value,
            len(survivors),
            len(warns),
            rounds,
            len(repaired_files),
            len(files),
        )
        # Always emit the verdict — a clean pass equals the default StaticChecks,
        # so _compute_diff alone would drop it from the returned diff.
        diff = self._compute_diff(before, ps)
        diff["static_checks"] = ps.static_checks.model_dump(mode="json")
        return diff

    # ── Repair loop ────────────────────────────────────────────────────

    def _repair_loop(
        self,
        ps: ProjectState,
        files: dict[str, str],
        errors: list[StaticFinding],
        dialect: str,
    ) -> tuple[StaticVerdict, int, list[str], dict[str, list[StaticFinding]]]:
        """Repair error findings file by file, bounded rounds per file.

        Returns (verdict, total rounds, repaired files, survivors per file).
        Findings without a file (e.g. checker crashes) cannot be repaired and
        always count as survivors.
        """
        by_owner: dict[str, dict[str, list[StaticFinding]]] = {}
        manifest_files = getattr(ps.code_manifest, "files", {}) or {}
        for f in errors:
            if f.file:
                manifest_entry = manifest_files.get(f.file)
                owner = str(getattr(manifest_entry, "agent", "") or "unowned")
                by_owner.setdefault(owner, {}).setdefault(f.file, []).append(f)

        rounds = 0
        repaired: list[str] = []
        survivors: dict[str, list[StaticFinding]] = {}
        for finding in errors:
            if not finding.file:
                survivors.setdefault("", []).append(finding)
        refuse = agent_refuse_corrupt(ps, threading.Lock())

        for owner, owned_files in by_owner.items():
            logger.info(
                "[IntegrationAgent] repairing %d file(s) owned by %s",
                len(owned_files),
                owner,
            )
            for filepath, file_errors in owned_files.items():
                for _ in range(MAX_REPAIR_ROUNDS):
                    rounds += 1
                    file_errors = self._repair_round(
                        ps, files, filepath, file_errors, dialect, refuse
                    )
                    if not file_errors:
                        repaired.append(filepath)
                        break
                if file_errors:
                    survivors[filepath] = file_errors

        verdict = (
            StaticVerdict.FAILED
            if survivors
            else StaticVerdict.REPAIRED
        )
        return verdict, rounds, repaired, survivors

    def _repair_round(
        self,
        ps: ProjectState,
        files: dict[str, str],
        filepath: str,
        file_errors: list[StaticFinding],
        dialect: str,
        refuse: Any,
    ) -> list[StaticFinding]:
        """One repair attempt for *filepath*: prompt → LLM → safe-write → re-check.

        Returns the file's remaining error findings ([] = repaired). A failed
        attempt (LLM error, unusable reply, refused write) returns the file's
        findings unchanged: the attempt consumed a round, and the file keeps
        its remaining allowance.
        """
        prompt = self._build_repair_prompt(
            filepath, files.get(filepath, ""), file_errors, ps.code_manifest, ps.architecture
        )
        try:
            raw = self._call_with_system(
                _REPAIR_SYSTEM_PROMPT,
                prompt,
                agent_name="integration-repair",
                state=ps,
            )
        except Exception as exc:
            logger.error("[IntegrationAgent] repair LLM call failed for %s: %s", filepath, exc)
            return file_errors

        try:
            data = self.extract_json(raw)
        except Exception as exc:
            logger.error("[IntegrationAgent] repair reply not JSON for %s: %s", filepath, exc)
            return file_errors

        code = str(data.get("code", "") or "").strip()
        if not code:
            logger.warning("[IntegrationAgent] repair reply for %s has no code", filepath)
            return file_errors

        try:
            written = self.workspace.write(filepath, code, on_corrupt=refuse)
        except Exception as exc:
            logger.error("[IntegrationAgent] repair write failed for %s: %s", filepath, exc)
            return file_errors
        if written is None:
            logger.warning("[IntegrationAgent] repair write refused (corrupt) for %s", filepath)
            return file_errors

        self._replace_bucket_entry(ps, filepath, written)
        files[filepath] = written

        try:
            rechecked = run_all_checks(files, ps.code_manifest, ps.architecture, dialect)
        except Exception as exc:
            logger.error("[IntegrationAgent] re-check crashed for %s: %s", filepath, exc)
            return file_errors          # keep the pre-repair findings as survivors
        return [
            f for f in rechecked
            if f.file == filepath and f.severity == FindingSeverity.ERROR
        ]

    def _replace_bucket_entry(self, ps: ProjectState, filepath: str, content: str) -> None:
        """Write the repaired content into the generated_code bucket that owns the file."""
        layer = self._bucket_layer(ps, filepath)
        for candidate in ("backend", "frontend", "database"):
            bucket = dict(getattr(ps.generated_code, candidate) or {})
            bucket.pop(filepath, None)
            if candidate == layer:
                bucket[filepath] = content
            setattr(ps.generated_code, candidate, bucket)

    @staticmethod
    def _bucket_layer(ps: ProjectState, filepath: str) -> str:
        """Code bucket for *filepath*, preferring its manifest owner."""
        owner = ""
        manifest_entry = (getattr(ps.code_manifest, "files", {}) or {}).get(filepath)
        if manifest_entry is not None:
            owner = str(getattr(manifest_entry, "agent", "") or "").lower()
        owner_layers = {
            "backend": "backend",
            "frontend": "frontend",
            "database": "database",
            "data_base": "database",
            "db": "database",
        }
        if owner in owner_layers:
            return owner_layers[owner]

        code_map = ps.generated_code
        for layer in ("backend", "frontend", "database"):
            if filepath in (getattr(code_map, layer, None) or {}):
                return layer
        norm = f"/{filepath.replace(os.sep, '/')}"
        for layer in ("backend", "frontend", "database"):
            if f"/{layer}/" in norm:
                return layer
        raise ValueError(f"Cannot determine generated-code owner for '{filepath}'")

    # ── Repair prompt ──────────────────────────────────────────────────

    def _build_repair_prompt(
        self,
        filepath: str,
        content: str,
        file_errors: list[StaticFinding],
        manifest: Any,
        architecture: Any,
    ) -> str:
        """Repair prompt: task, findings with line numbers, file content, contract excerpt."""
        lines = [
            f"The file `{filepath}` failed the Integration Agent's static checks.",
            "Fix every finding below and return the FULL corrected file content.",
            'Reply with a single JSON object: {"filepath": "<path>", "code": "<full corrected file>"}.',
            "Change only what the findings require; keep everything else identical.",
            "",
            "## Findings (fix ALL of these)",
        ]
        for f in file_errors:
            location = f"line {f.line}" if f.line else "file"
            lines.append(f"- [{f.check}] {location}: {f.message}")

        lines += ["", "## Current file content"]
        ext = os.path.splitext(filepath)[1].lstrip(".")
        lines.append(f"```{ext}")
        lines.append(content if content.strip() else "(file content is missing — regenerate it from the contract below)")
        lines.append("```")

        if any(f.check in _HTTP_CHECKS for f in file_errors):
            lines += ["", "## API CONTRACT — endpoints the backend exposes (calls must match these)"]
            contracts = filter_api_contracts_by_relevance(
                getattr(architecture, "api_contracts", None) or [],
                filepath,
                purpose=" ".join(f.message for f in file_errors),
            )
            if not contracts:
                lines.append("(no API contracts available)")
            for contract in contracts:
                lines.append(self._contract_line(contract))

        if any(f.check in _SCHEMA_CHECKS for f in file_errors):
            lines += ["", "## SCHEMA FIELD CONTRACT — valid entities and fields"]
            field_contracts = self._schema_contract_lines(manifest, file_errors)
            lines.extend(field_contracts or ["(no manifest field contracts available)"])

        if any(f.check in PERSISTENCE_CHECKS for f in file_errors):
            contract = getattr(architecture, "persistence_contract", None)
            lines += ["", render_persistence_contract(contract)]
            rules = backend_persistence_rules(contract)
            if rules:
                lines += ["", rules]

        fm = getattr(manifest, "files", {}).get(filepath) if manifest is not None else None
        if fm is not None:
            lines += ["", "## FILE CONTRACT — planner contract for this file"]
            if getattr(fm, "purpose", ""):
                lines.append(f"- purpose: {fm.purpose}")
            exports = [e for e in (getattr(fm, "exports", None) or []) if str(e).strip()]
            lines.append(f"- exports: {', '.join(exports) if exports else '(none)'}")
            for export, fields in (getattr(fm, "contract", None) or {}).items():
                rendered = ", ".join(f"{n}: {t}" for n, t in fields.items())
                lines.append(f"- `{export}` fields: {rendered}")
            import_contract = build_import_contract(manifest, filepath)
            if import_contract:
                lines += ["", import_contract]

        return "\n".join(lines)

    @staticmethod
    def _schema_contract_lines(
        manifest: Any,
        file_errors: list[StaticFinding],
    ) -> list[str]:
        """Render the database field contracts most relevant to schema findings."""
        entries = getattr(manifest, "files", {}) or {}
        context = " ".join(f.message for f in file_errors).lower()
        ranked: list[tuple[int, str, Any]] = []
        for path, entry in entries.items():
            contract = getattr(entry, "contract", None) or {}
            if not contract:
                continue
            terms = {
                str(export).lower()
                for export in contract
            } | {
                str(field).lower()
                for fields in contract.values()
                for field in fields
            }
            score = sum(1 for term in terms if term and term in context)
            is_database = str(getattr(entry, "agent", "")).lower() in {
                "database", "data_base", "db"
            }
            ranked.append((score + int(is_database), path, entry))

        lines: list[str] = []
        for _, path, entry in sorted(ranked, key=lambda item: (-item[0], item[1]))[:10]:
            for export, fields in (getattr(entry, "contract", None) or {}).items():
                rendered = ", ".join(f"{name}: {kind}" for name, kind in fields.items())
                lines.append(f"- `{path}` → `{export}` fields: {rendered}")
        return lines

    @staticmethod
    def _contract_line(contract: Any) -> str:
        """One API contract as 'METHOD path — request: ...; response: ...'."""
        if isinstance(contract, dict):
            get = lambda key, default="": contract.get(key, default)  # noqa: E731
        else:
            get = lambda key, default="": getattr(contract, key, default)  # noqa: E731
        line = f"- {str(get('method', '')).upper()} {get('path', '')}".strip()
        for key, label in (("request_body", "request"), ("response_body", "response")):
            body = get(key) or {}
            if body:
                line += f" — {label}: " + ", ".join(f"{n}: {t}" for n, t in body.items())
        return line
