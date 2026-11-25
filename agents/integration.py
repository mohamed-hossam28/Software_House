"""
Integration Agent — static checks + bounded LLM repair loop.

First, every planned file that was never written (in the code manifest, no
Planned Content, absent from generated_code) is regenerated with its owning
agent's coding prompt through the coding loop's single-file path — at most
MAX_REGENERATE_ATTEMPTS attempts per file, each counted in `rounds`, the retry
carrying the last write-refusal reason. A file still missing afterwards is a
`missing_file` error naming the file and its owner (or saying no coding
profile exists for the owner). It is never sent to repair, and neither is any
import finding naming it, so repair never drops the dependency; its importers'
other errors are repaired as usual.

Then it collects every file the coding agents wrote (generated_code buckets), runs the
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
from typing import Any, Callable

from agents.base import BaseAgent
from agents.coding_loop import (
    CodingAgent,
    CodingProfile,
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
from core.workspace import dotted_import

logger = logging.getLogger(__name__)


def _owner_profiles() -> dict[str, CodingProfile]:
    """Each owning agent key's CodingProfile (imported lazily: the agents import the loop)."""
    from agents.backend import BACKEND_PROFILE
    from agents.data_base import DATABASE_PROFILE
    from agents.frontend import FRONTEND_PROFILE

    return {
        "backend": BACKEND_PROFILE,
        "frontend": FRONTEND_PROFILE,
        "database": DATABASE_PROFILE,
    }


def _imports_missing_module(finding: StaticFinding, missing_modules: set[str]) -> bool:
    """True for an import finding whose target is (inside) a still-missing module."""
    if finding.check != "python_imports" or not missing_modules:
        return False
    return any(
        f"'{module}'" in finding.message or f"'{module}." in finding.message
        for module in missing_modules
    )


def _log_refused_write(filepath: str, conflict: dict) -> None:
    """Log a refused safe-write so the corruption is visible in the run logs."""
    logger.warning("[safe-write] REFUSED corrupted content for %s: %s", filepath, conflict.get("reason"))

#: Maximum repair rounds per file. An unrepairable file stops after this many.
MAX_REPAIR_ROUNDS = 2
#: Maximum regeneration attempts per missing planned file, each counted as a round.
MAX_REGENERATE_ATTEMPTS = 2

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

    def _work(self, ps: ProjectState) -> None:
        # Planned files that were never written are regenerated first, with
        # their owner's coding prompt; the checks then see the recovered files.
        rounds, regenerated, still_missing = self._regenerate_missing(ps)
        files = self._collect_files(ps)
        missing_findings = [
            self._missing_finding(ps, filepath, owner)
            for filepath, owner in still_missing.items()
        ]
        # Never send a missing file's import finding to repair: the repair could
        # drop the dependency instead of the file being written. Its importers'
        # other errors are still repaired.
        missing_modules = {dotted_import(p) for p in still_missing if p.endswith(".py")}

        def unrepairable(f: StaticFinding) -> bool:
            return f.file in still_missing or _imports_missing_module(f, missing_modules)

        try:
            findings = run_all_checks(files, ps.code_manifest, ps.architecture)
        except Exception as exc:
            logger.exception("[IntegrationAgent] static checks crashed")
            findings = [StaticFinding(
                check="static_checks",
                severity=FindingSeverity.ERROR,
                message=f"Static checks failed: {exc}",
            )]

        errors = [
            f for f in findings
            if f.severity == FindingSeverity.ERROR and not unrepairable(f)
        ]
        verdict = StaticVerdict.REPAIRED if regenerated else StaticVerdict.PASS
        repaired_files: list[str] = list(regenerated)
        final_findings = findings

        if errors:
            logger.info(
                "[IntegrationAgent] %d error finding(s) — starting repair loop",
                len(errors),
            )
            _, repair_rounds, repaired, survivors_by_file = self._repair_loop(
                ps, files, errors, unrepairable
            )
            rounds += repair_rounds
            repaired_files += repaired
            # Individual rounds re-check only the repaired file's findings while
            # retaining the full project as dependency/schema context. After all
            # sequential repairs, this authoritative project check refreshes
            # cross-file errors and warnings before deriving the final verdict.
            try:
                final_findings = run_all_checks(
                    files, ps.code_manifest, ps.architecture
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

        final_findings = final_findings + missing_findings
        if any(f.severity == FindingSeverity.ERROR for f in final_findings):
            verdict = StaticVerdict.FAILED

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

        logger.info(
            "[IntegrationAgent] verdict=%s errors=%d warnings=%d rounds=%d repaired=%d files=%d",
            ps.static_checks.verdict.value,
            len(survivors),
            len(warns),
            rounds,
            len(repaired_files),
            len(files),
        )

    # ── Missing-file recovery ──────────────────────────────────────────

    @staticmethod
    def _collect_files(ps: ProjectState) -> dict[str, str]:
        files: dict[str, str] = {}
        for layer in ("backend", "frontend", "database"):
            files.update(getattr(ps.generated_code, layer, {}) or {})
        return files

    def _regenerate_missing(
        self, ps: ProjectState
    ) -> tuple[int, list[str], dict[str, str]]:
        """Regenerate every missing planned file with its owner's coding prompt.

        A file is missing when the code manifest plans it, it has no Planned
        Content, and no generated-code bucket holds it. Each gets up to
        MAX_REGENERATE_ATTEMPTS attempts, each counted as a round.

        Returns (rounds, regenerated files, still-missing file -> owner).
        """
        files = self._collect_files(ps)
        missing = [
            fm for fm in (getattr(ps.code_manifest, "files", {}) or {}).values()
            if fm.planned_content is None and fm.filepath not in files
        ]
        rounds = 0
        regenerated: list[str] = []
        still_missing: dict[str, str] = {}
        for fm in missing:
            owner = str(fm.agent or "unowned")
            coder = self._coder_for(owner)
            written = False
            for attempt in range(1, MAX_REGENERATE_ATTEMPTS + 1):
                if coder is None:
                    break
                rounds += 1
                logger.info(
                    "[IntegrationAgent] regenerating missing %s (owner %s), attempt %d/%d",
                    fm.filepath, owner, attempt, MAX_REGENERATE_ATTEMPTS,
                )
                try:
                    written = coder.regenerate(ps, fm.filepath)
                except Exception as exc:
                    logger.error("[IntegrationAgent] regenerating %s failed: %s", fm.filepath, exc)
                if written:
                    break
            if written:
                regenerated.append(fm.filepath)
            else:
                still_missing[fm.filepath] = owner
        return rounds, regenerated, still_missing

    def _coder_for(self, owner: str) -> CodingAgent | None:
        """The owning Coding agent, sharing this agent's provider and workspace."""
        profile = _owner_profiles().get(owner.lower())
        if profile is None:
            return None
        return CodingAgent(
            profile=profile,
            model=self.model,
            max_tokens=self.max_tokens,
            retries=self.max_retries,
            backoff=self.backoff,
            temperature=self.temperature,
            provider=self.provider,
            workspace=self.workspace,
        )

    @staticmethod
    def _missing_finding(ps: ProjectState, filepath: str, owner: str) -> StaticFinding:
        """The `missing_file` error for a planned file still unwritten after recovery."""
        if _owner_profiles().get(owner.lower()) is None:
            message = (
                f"Planned file '{filepath}' (owner: {owner}) was never written, and no "
                f"coding profile exists for owner '{owner}' to regenerate it"
            )
        else:
            message = (
                f"Planned file '{filepath}' (owner: {owner}) was never written after "
                f"{MAX_REGENERATE_ATTEMPTS} attempt(s)"
            )
        if filepath in ps.write_refusals:
            message += f" — last refusal: {ps.write_refusals[filepath]}"
        return StaticFinding(
            file=filepath, check="missing_file", severity=FindingSeverity.ERROR, message=message,
        )

    # ── Repair loop ────────────────────────────────────────────────────

    def _repair_loop(
        self,
        ps: ProjectState,
        files: dict[str, str],
        errors: list[StaticFinding],
        unrepairable: Callable[[StaticFinding], bool] = lambda f: False,
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
        def refuse(filepath: str, conflict: dict) -> None:
            _log_refused_write(filepath, conflict)
            ps.record_refusal(filepath, str(conflict.get("reason") or ""))

        for owner, owned_files in by_owner.items():
            logger.info(
                "[IntegrationAgent] repairing %d file(s) owned by %s",
                len(owned_files),
                owner,
            )
            for filepath, file_errors in owned_files.items():
                for _ in range(MAX_REPAIR_ROUNDS):
                    rounds += 1
                    file_errors = [
                        f for f in self._repair_round(ps, files, filepath, file_errors, refuse)
                        if not unrepairable(f)
                    ]
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
        ps.write_refusals.pop(filepath, None)
        files[filepath] = written

        try:
            rechecked = run_all_checks(files, ps.code_manifest, ps.architecture)
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
            contract = architecture.persistence_contract
            lines += ["", render_persistence_contract(contract), "", backend_persistence_rules(contract)]

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
