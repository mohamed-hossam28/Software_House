"""
agents/architecture.py
──────────────────────
Architecture Agent — parallel 3-round execution via ThreadPoolExecutor.

ROUND 1 (fully independent — run in parallel)
  ├── Call A: tech_stack
  ├── Call B: system_components
  └── Call C: db_schema_overview

ROUND 2 (depends on Round 1 — run in parallel)
  ├── Call D: api_contracts       (needs tech_stack + system_components)
  └── Call E: security_requirements (needs tech_stack)

ROUND 3 (depends on Round 2 — single call)
  └── Call F: data_flow           (needs api_contracts + system_components)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any

from .base import BaseAgent
from core.state import ProjectState, ArchData, SRSData, APIContract

logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 3

# ── Default tech stack ──────────────────────────────────────────────────────────
DEFAULT_TECH_STACK: dict[str, str] = {
    "backend":  "FastAPI (Python 3.12)",
    "frontend": "React 18 + TypeScript",
    "database": "PostgreSQL 16",
    "cache":    "Redis 7",
    "infra":    "Docker + nginx (reverse proxy)",
}

# ── Keyword → canonical technology name ─────────────────────────────────────────
# Used by TechStackResolver.from_srs() to detect user intent from the SRS text.
_LAYER_TECH_MAP: dict[str, dict[str, str]] = {
    "backend": {
        "fastapi":  "FastAPI (Python 3.12)",
        "django":   "Django (Python 3.12)",
        "flask":    "Flask (Python 3.12)",
        "express":  "Express.js (Node.js)",
        "nestjs":   "NestJS (Node.js)",
        "spring":   "Spring Boot (Java 21)",
        "rails":    "Ruby on Rails 7",
        "golang":   "Go 1.22 (net/http)",
        "laravel":  "Laravel 11 (PHP)",
    },
    "frontend": {
        "react":    "React 18 + TypeScript",
        "vue":      "Vue 3 + TypeScript",
        "angular":  "Angular 17",
        "next.js":  "Next.js 14 (React)",
        "nextjs":   "Next.js 14 (React)",
        "svelte":   "SvelteKit",
        "nuxt":     "Nuxt 3 (Vue)",
    },
    "database": {
        "postgresql": "PostgreSQL 16",
        "postgres":   "PostgreSQL 16",
        "mysql":      "MySQL 8",
        "mongodb":    "MongoDB 7",
        "sqlite":     "SQLite 3",
        "dynamodb":   "AWS DynamoDB",
        "cassandra":  "Apache Cassandra",
        "supabase":   "Supabase (PostgreSQL)",
    },
    "cache": {
        "redis":     "Redis 7",
        "memcached": "Memcached",
        "valkey":    "Valkey 7",
    },
    "infra": {
        "docker":      "Docker + nginx (reverse proxy)",
        "kubernetes":  "Kubernetes + Helm",
        "k8s":         "Kubernetes + Helm",
        "aws":         "AWS (ECS + RDS + ElastiCache)",
        "gcp":         "GCP (Cloud Run + Cloud SQL)",
        "azure":       "Azure (Container Apps)",
        "vercel":      "Vercel + Railway",
        "heroku":      "Heroku",
    },
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TechStackResolver                                                          ║
# ║  ─────────────────                                                          ║
# ║  Merges three sources (SRS, store, defaults) into one authoritative stack.  ║
# ║  Swap out from_store() when the persistent store is built.                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class TechStackResolver:
    """
    Resolves the tech stack with clear priority:

      SRS constraints/assumptions  (highest — explicit user intent)
            ↑ overrides
      Persistent user preference store   (stored from previous projects / settings)
            ↑ overrides
      DEFAULT_TECH_STACK                 (sensible fallback for every layer)
    """

    # ── Source 1: SRS text scanner ───────────────────────────────────────────────
    @classmethod
    def from_srs(cls, srs: SRSData) -> dict[str, str]:
        """
        Scan SRS constraints and assumptions for technology keywords.
        Returns only the layers where a clear preference was detected.
        """
        # Combine the text fields most likely to contain tech preferences
        combined_text = " ".join([
            *srs.constraints,
            *srs.assumptions,
            srs.raw_prompt,
        ]).lower()

        detected: dict[str, str] = {}
        for layer, keywords in _LAYER_TECH_MAP.items():
            for keyword, canonical in keywords.items():
                if keyword in combined_text:
                    detected[layer] = canonical
                    logger.debug(
                        "[TechStackResolver] SRS detected '%s' → layer '%s' = '%s'",
                        keyword, layer, canonical,
                    )
                    break   # first match wins per layer

        return detected

    # ── Source 2: Persistent user preference store (STUB) ───────────────────────
    @classmethod
    def from_store(cls, project_id: str) -> dict[str, str]:
        """
        Load the user's saved tech-stack preferences from the preference store.

        STUB — returns an empty dict until the store / checkpointer is wired up.
        When the store is built, replace the body with:

            from memory.store import PreferenceStore
            return PreferenceStore.get_tech_stack(project_id) or {}

        The store is expected to return a dict[str, str] shaped identically to
        DEFAULT_TECH_STACK — only include layers the user has a saved preference for.
        """
        # TODO: wire to PreferenceStore when memory layer is implemented
        _ = project_id  # suppress "unused" warning
        return {}

    # ── Resolver ─────────────────────────────────────────────────────────────────
    @classmethod
    def resolve(cls, state: ProjectState) -> dict[str, str]:
        """
        Merge all three sources.  Higher-priority sources overwrite lower ones
        only for layers they have an opinion on — layers with no preference fall
        through to the default.

        Priority (highest → lowest):
          1. SRS-detected preferences
          2. Store preferences
          3. DEFAULT_TECH_STACK
        """
        stack: dict[str, str] = {**DEFAULT_TECH_STACK}

        store_prefs = cls.from_store(state.project_id)
        if store_prefs:
            stack.update(store_prefs)
            logger.info(
                "[TechStackResolver] store overrode %d layer(s): %s",
                len(store_prefs), list(store_prefs.keys()),
            )

        srs_prefs = cls.from_srs(state.requirements)
        if srs_prefs:
            stack.update(srs_prefs)
            logger.info(
                "[TechStackResolver] SRS overrode %d layer(s): %s",
                len(srs_prefs), list(srs_prefs.keys()),
            )

        return stack


# ── Helpers ──────────────────────────────────────────────────────────────────────
def _bullet_list(items: list[str]) -> str:
    if not items:
        return "  (none)"
    return "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(items))


def _tech_stack_table(stack: dict[str, str]) -> str:
    """Format the resolved stack as a readable markdown table for the LLM."""
    rows = [f"| {layer:<10} | {tech} |" for layer, tech in stack.items()]
    header = "| Layer      | Technology |\n|------------|------------|"
    return header + "\n" + "\n".join(rows)


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _validated_contracts(raw: Any) -> list[APIContract]:
    """Parse raw LLM output into a list of typed APIContract objects."""
    if not isinstance(raw, list):
        logger.warning("[ArchitectureAgent] api_contracts is not a list — ignoring.")
        return []
    valid: list[APIContract] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            logger.warning("[ArchitectureAgent] contract[%d] is not a dict — skipped.", i)
            continue
        if not all(k in item for k in ("method", "path")):
            logger.warning(
                "[ArchitectureAgent] contract[%d] missing method/path — skipped.", i
            )
            continue
        # Normalise request/response field names — accept both old and new keys
        req_body = item.get("request_body") or item.get("request", {})
        res_body = item.get("response_body") or item.get("response", {})
        valid.append(APIContract(
            method        = str(item.get("method", "")).upper(),
            path          = str(item.get("path",   "")),
            auth_required = bool(item.get("auth_required", True)),
            description   = str(item.get("description", "")),
            request_body  = {str(k): str(v) for k, v in req_body.items()} if isinstance(req_body, dict) else {},
            response_body = {str(k): str(v) for k, v in res_body.items()} if isinstance(res_body, dict) else {},
        ))
    return valid


def _srs_block(srs: SRSData) -> str:
    """Format the full SRS into a reusable prompt block for Round 1 calls."""
    return (
        "## Software Requirements Specification\n\n"
        "### Functional Requirements\n"
        f"{_bullet_list(srs.functional_requirements)}\n\n"
        "### Non-Functional Requirements\n"
        f"{_bullet_list(srs.non_functional_requirements)}\n\n"
        "### Constraints\n"
        f"{_bullet_list(srs.constraints)}\n\n"
        "### Assumptions\n"
        f"{_bullet_list(srs.assumptions)}\n\n"
        "### Out of Scope\n"
        f"{_bullet_list(srs.out_of_scope)}\n"
    )


def _srs_summary_block(srs: SRSData, stack: dict[str, str]) -> str:
    """Format a condensed SRS summary (functional requirements + tech stack) for Round 2 & 3 calls to minimize token usage."""
    return (
        "## Requirements & Tech Stack Summary\n\n"
        "### Functional Requirements\n"
        f"{_bullet_list(srs.functional_requirements)}\n\n"
        "### Tech Stack\n"
        f"{_tech_stack_table(stack)}\n"
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Per-call system / user prompts                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_BASE_SYSTEM = (
    "You are a principal software architect with 20+ years of experience "
    "designing production-grade, cloud-native systems.\n"
    "Respond with a SINGLE valid JSON object — no prose outside the JSON."
)

# ── Round 1 prompts ──────────────────────────────────────────────────────────

TECH_STACK_SYSTEM = _BASE_SYSTEM + """
Your task: decide the backend, frontend, database, cache, and infrastructure
technologies for a project described in the SRS below.
Return JSON: {"tech_stack": {"<layer>": "<technology + version>", ...}}
Use ONLY the resolved tech stack provided when available; fill gaps with best choices.
"""

SYSTEM_COMPONENTS_SYSTEM = _BASE_SYSTEM + """
Your task: list every high-level service/component the system needs
(e.g. AuthService, BoardService, NotificationService, APIGateway, etc.).
Be exhaustive — derive every component the SRS implies.
Return JSON: {"system_components": ["<ServiceName: one-line description>", ...]}
"""

DB_SCHEMA_SYSTEM = _BASE_SYSTEM + """
Your task: produce a database schema overview — entity names and their primary
relationships. NO DDL, NO SQL — just entity names and how they relate.
Return JSON: {"database_schema_overview": "<multiline string: entity → related entities>"}
"""

# ── Round 2 prompts ──────────────────────────────────────────────────────────

API_CONTRACTS_SYSTEM = _BASE_SYSTEM + """
Your task: list every REST API endpoint the backend must expose.
Tag each endpoint with an owner_component from the system_components list.
Include realistic request_body and response_body shapes (field names + types as strings).
Return JSON:
{"api_contracts": [
  {"method":"<HTTP verb>","path":"<URL>","auth_required":true|false,
   "description":"<what it does>","owner_component":"<ServiceName>",
   "request_body":{"<field>":"<type>"},"response_body":{"<field>":"<type>"}}
]}
"""

SECURITY_SYSTEM = _BASE_SYSTEM + """
Your task: define security requirements — auth strategy, RBAC roles, data
encryption, rate limiting, CORS policy, secrets management.
Return JSON: {"security_requirements": ["<requirement statement>", ...]}
"""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ArchitectureAgent                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class ArchitectureAgent(BaseAgent):
    """
    Reads the completed SRS from ProjectState.requirements and writes a fully
    populated ArchData into ProjectState.architecture.

    Executes 5 focused LLM calls across 2 dependency-resolved rounds
    using ThreadPoolExecutor with a Lock on shared state.
    """

    @property
    def system_prompt(self) -> str:
        return _BASE_SYSTEM

    # ── unused (run is overridden) ───────────────────────────────────────────
    def _build_prompt(self, state: ProjectState) -> str:
        raise NotImplementedError("_build_prompt unused; run() is overridden.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        raise NotImplementedError("_parse_response unused; run() is overridden.")

    # ── LLM helper with custom system prompt ─────────────────────────────────
    def _call_with_system(self, system: str, user: str, agent_name: str = "") -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(
                    prompt=user,
                    system_prompt=system,
                    agent_name=agent_name or self.name,
                )
                return response.text
            except Exception as exc:
                logger.warning(
                    "[ArchitectureAgent] LLM attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)
        raise RuntimeError("Unreachable")

    # ── Individual sub-call workers ──────────────────────────────────────────

    def _call_tech_stack(self, srs_text: str, stack_table: str,
                         results: dict, lock: Lock, state: ProjectState) -> None:
        prompt = f"## Resolved Tech Stack (use as baseline)\n{stack_table}\n\n{srs_text}\n```json\n"
        try:
            from agents.schemas.architecture import TechStackSchema
            raw = self._call_with_system(TECH_STACK_SYSTEM, prompt, "arch-tech")
            data = self.extract_json_and_validate(raw, TechStackSchema)
            with lock:
                results["tech_stack"] = data.get("tech_stack", {})
                state.metrics.total_llm_calls += 1
            logger.info("[ArchitectureAgent] ✓ tech_stack complete")
        except Exception as exc:
            logger.error("[ArchitectureAgent] tech_stack failed: %s", exc)
            with lock:
                state.log_error(self.name, f"tech_stack sub-call failed: {exc}")

    def _call_system_components(self, srs_text: str,
                                results: dict, lock: Lock, state: ProjectState) -> None:
        try:
            from agents.schemas.architecture import SystemComponentsSchema
            raw = self._call_with_system(SYSTEM_COMPONENTS_SYSTEM, srs_text + "\n```json\n", "arch-components")
            data = self.extract_json_and_validate(raw, SystemComponentsSchema)
            with lock:
                results["system_components"] = data.get("system_components", [])
                state.metrics.total_llm_calls += 1
            logger.info("[ArchitectureAgent] ✓ system_components complete")
        except Exception as exc:
            logger.error("[ArchitectureAgent] system_components failed: %s", exc)
            with lock:
                state.log_error(self.name, f"system_components sub-call failed: {exc}")

    def _call_db_schema(self, srs_text: str,
                        results: dict, lock: Lock, state: ProjectState) -> None:
        try:
            from agents.schemas.architecture import DbSchemaOverviewSchema
            raw = self._call_with_system(DB_SCHEMA_SYSTEM, srs_text + "\n```json\n", "arch-db")
            data = self.extract_json_and_validate(raw, DbSchemaOverviewSchema)
            with lock:
                results["database_schema_overview"] = data.get("database_schema_overview", "")
                state.metrics.total_llm_calls += 1
            logger.info("[ArchitectureAgent] ✓ db_schema_overview complete")
        except Exception as exc:
            logger.error("[ArchitectureAgent] db_schema_overview failed: %s", exc)
            with lock:
                state.log_error(self.name, f"db_schema_overview sub-call failed: {exc}")

    def _call_api_contracts(self, srs_summary_text: str, tech_stack: dict, components: list,
                            results: dict, lock: Lock, state: ProjectState) -> None:
        context = (
            f"## System Components\n" +
            "\n".join(f"- {c}" for c in components) + "\n\n" +
            srs_summary_text + "\n```json\n"
        )
        try:
            from agents.schemas.architecture import APIContractsSchema
            raw = self._call_with_system(API_CONTRACTS_SYSTEM, context, "arch-api")
            data = self.extract_json_and_validate(raw, APIContractsSchema)
            with lock:
                results["api_contracts"] = data.get("api_contracts", [])
                state.metrics.total_llm_calls += 1
            logger.info("[ArchitectureAgent] ✓ api_contracts complete")
        except Exception as exc:
            logger.error("[ArchitectureAgent] api_contracts failed: %s", exc)
            with lock:
                state.log_error(self.name, f"api_contracts sub-call failed: {exc}")

    def _call_security(self, srs_summary_text: str, tech_stack: dict,
                       results: dict, lock: Lock, state: ProjectState) -> None:
        context = f"{srs_summary_text}\n```json\n"
        try:
            from agents.schemas.architecture import SecurityRequirementsSchema
            raw = self._call_with_system(SECURITY_SYSTEM, context, "arch-security")
            data = self.extract_json_and_validate(raw, SecurityRequirementsSchema)
            with lock:
                results["security_requirements"] = data.get("security_requirements", [])
                state.metrics.total_llm_calls += 1
            logger.info("[ArchitectureAgent] ✓ security_requirements complete")
        except Exception as exc:
            logger.error("[ArchitectureAgent] security_requirements failed: %s", exc)
            with lock:
                state.log_error(self.name, f"security_requirements sub-call failed: {exc}")

    # ── Main execution ───────────────────────────────────────────────────────
    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)

        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        # Resolve tech stack deterministically (no LLM needed)
        authoritative_stack = TechStackResolver.resolve(ps)
        stack_table = _tech_stack_table(authoritative_stack)
        srs_text = _srs_block(ps.requirements)

        logger.info("[ArchitectureAgent] resolved tech stack: %s", authoritative_stack)

        results: dict[str, Any] = {}
        state_lock = Lock()

        # ── ROUND 1: tech_stack, system_components, db_schema (parallel) ─────
        logger.info("[ArchitectureAgent] ── Round 1/2 (3 parallel calls) ──")
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            futures: list[Future] = [
                pool.submit(self._call_tech_stack, srs_text, stack_table,
                            results, state_lock, ps),
                pool.submit(self._call_system_components, srs_text,
                            results, state_lock, ps),
                pool.submit(self._call_db_schema, srs_text,
                            results, state_lock, ps),
            ]
            for f in futures:
                f.result()

        # Merge authoritative stack over LLM-returned stack
        merged_stack = {**results.get("tech_stack", {}), **authoritative_stack}
        results["tech_stack"] = merged_stack

        srs_summary_text = _srs_summary_block(ps.requirements, merged_stack)

        # ── ROUND 2: api_contracts, security_requirements (parallel) ─────────
        logger.info("[ArchitectureAgent] ── Round 2/2 (2 parallel calls) ──")
        time.sleep(1)  # small delay between rounds to respect rate limits
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            futures = [
                pool.submit(self._call_api_contracts, srs_summary_text,
                            merged_stack,
                            results.get("system_components", []),
                            results, state_lock, ps),
                pool.submit(self._call_security, srs_summary_text,
                            merged_stack,
                            results, state_lock, ps),
            ]
            for f in futures:
                f.result()

        # ── Assemble ArchData ────────────────────────────────────────────────
        security_block = "\n\n## Security Requirements\n" + "\n".join(
            f"  - {r}" for r in _as_list(results.get("security_requirements", []))
        )

        ps.architecture = ArchData(
            tech_stack               = merged_stack,
            system_components        = _as_list(results.get("system_components", [])),
            api_contracts            = _validated_contracts(results.get("api_contracts", [])),
            database_schema_overview = str(results.get("database_schema_overview", "")).strip(),
            deployment_strategy      = security_block,
            diagrams                 = [],
        )

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)
        logger.info(
            "[ArchitectureAgent] architecture complete in %.2fs — "
            "%d components, %d endpoints.",
            duration,
            len(ps.architecture.system_components),
            len(ps.architecture.api_contracts),
        )
        return self._compute_diff(before, ps)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Quick smoke-test  (run:  python -m agents.architecture)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    import json

    print("\n" + "=" * 60)
    print("ArchitectureAgent — local smoke-test (no LLM call)")
    print("=" * 60)

    from core.state import ProjectState, SRSData

    # ── TechStackResolver.from_srs ──────────────────────────────────────────────
    srs = SRSData(
        raw_prompt="Build a todo app using FastAPI and React with PostgreSQL.",
        constraints=["Must use FastAPI", "Must use PostgreSQL"],
        assumptions=[],
    )
    detected = TechStackResolver.from_srs(srs)
    assert "backend"  in detected and "fastapi"  in detected["backend"].lower(),  "FAIL: FastAPI not detected"
    assert "database" in detected and "postgres" in detected["database"].lower(), "FAIL: PostgreSQL not detected"
    assert "frontend" in detected and "react"    in detected["frontend"].lower(), "FAIL: React not detected"
    print("[PASS] TechStackResolver.from_srs — FastAPI, React, PostgreSQL detected")

    # ── TechStackResolver.resolve — SRS overrides default ──────────────────────
    state = ProjectState(project_id="arch-test", project_name="arch-test")
    state.requirements = srs
    resolved = TechStackResolver.resolve(state)
    assert resolved["backend"]  == detected["backend"],  "FAIL: SRS backend pref not applied"
    assert resolved["database"] == detected["database"], "FAIL: SRS database pref not applied"
    # Layers without SRS opinion fall through to DEFAULT_TECH_STACK
    assert "cache" in resolved, "FAIL: cache layer should fall through to default"
    print("[PASS] TechStackResolver.resolve — SRS overrides, defaults fill gaps")

    # ── _validated_contracts ────────────────────────────────────────────────────
    raw_contracts = [
        {"method": "post", "path": "/auth/login",    "auth_required": False,
         "description": "Login", "request_body": {"email": "str"}, "response_body": {"token": "str"}},
        {"method": "get",  "path": "/users/me",      "auth_required": True,
         "description": "Get me"},
        # Missing required field (no path) — must be dropped
        {"method": "delete", "description": "No path"},
        # Not a dict — must be dropped
        "bad_item",
    ]
    contracts = _validated_contracts(raw_contracts)
    assert len(contracts) == 2,                                 "FAIL: expected 2 valid contracts"
    assert contracts[0].method == "POST",                       "FAIL: method should be uppercased"
    assert contracts[0].auth_required is False,                 "FAIL: auth_required wrong"
    assert contracts[0].request_body == {"email": "str"},       "FAIL: request_body wrong"
    assert contracts[0].response_body == {"token": "str"},      "FAIL: response_body wrong"
    assert contracts[1].request_body == {},                     "FAIL: missing request_body should default to {}"
    print("[PASS] _validated_contracts — 2 valid APIContract objects, bad items dropped")

    print("\n" + "=" * 60)
    print("All ArchitectureAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
