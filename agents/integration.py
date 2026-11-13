"""
Integration Agent — first full vertical cut of the static-check layer (no repair).

Collects every file the coding agents wrote (generated_code buckets), runs the
checker seam (core.static_checks.run_all_checks) over them, and records the
outcome:

  - state.static_checks.verdict → PASS / FAILED (no repair yet, so REPAIRED is unused)
  - state.static_checks.findings → every checker finding
  - error-level findings are mirrored into the shared state error list so they
    surface in the run summary

Follows the base-agent contract: accepts the full state (dict or ProjectState),
returns only the changed fields as a diff dict. No LLM call happens here —
the provider slot exists only to satisfy the shared constructor.
"""
import logging
import time
from typing import Any

from agents.base import BaseAgent
from core.state import (
    FindingSeverity,
    ProjectState,
    StaticChecks,
    StaticFinding,
    StaticVerdict,
)
from core.static_checks import run_all_checks

logger = logging.getLogger(__name__)


class IntegrationAgent(BaseAgent):
    """Static-check agent. Verifies generated code without calling the LLM."""

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Integration Agent. You verify generated code statically "
            "and report findings. This agent does not generate code."
        )

    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)

        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        files: dict[str, str] = {}
        for layer in ("backend", "frontend", "database", "devops"):
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
        warns = [f for f in findings if f.severity == FindingSeverity.WARN]

        ps.static_checks = StaticChecks(
            verdict=StaticVerdict.FAILED if errors else StaticVerdict.PASS,
            findings=findings,
            rounds=0,
            repaired_files=[],
        )

        for f in errors:
            location = f"{f.file}:{f.line}" if f.line else f.file
            ps.log_error(self.name, f"{f.check}: {location} {f.message}".strip())

        ps.metrics.agent_durations[self.name] = round(time.perf_counter() - t0, 3)
        logger.info(
            "[IntegrationAgent] verdict=%s errors=%d warnings=%d files=%d",
            ps.static_checks.verdict.value,
            len(errors),
            len(warns),
            len(files),
        )
        # Always emit the verdict — a clean pass equals the default StaticChecks,
        # so _compute_diff alone would drop it from the returned diff.
        diff = self._compute_diff(before, ps)
        diff["static_checks"] = ps.static_checks.model_dump(mode="json")
        return diff
