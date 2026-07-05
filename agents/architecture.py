"""
agents/architecture.py
──────────────────────
Architecture Agent — runs directly after the Requirements Agent.

Input:  ProjectState.requirements  (SRSData — fully populated)
Output: ProjectState.architecture  (ArchData — fully populated)

Tech-stack resolution priority (highest → lowest):
  1. Explicit tech mentioned in SRS constraints / assumptions
  2. User preferences from the persistent store  (TechStackStore — stub until store is built)
  3. Sensible defaults per layer

What it DOES produce:
  • Tech stack decision + rationale
  • System component list (services)
  • API contract surface (method, path, auth, request / response shape)
  • Database schema overview  (entity names + key relationships — no DDL)
  • Folder structure          (monorepo layout — no actual file contents)
  • Communication pattern     (sync REST / async events / which services talk to which)
  • Security requirements     (auth strategy, RBAC, encryption, rate limiting)

What it does NOT produce:
  • Actual source code        → engineering agents
  • Detailed SQL / migrations → Database agent
  • Dockerfiles / CI config   → DevOps agent
"""

import logging
from typing import Any

from .base import BaseAgent
from core.state import ProjectState, ArchData, SRSData, APIContract

logger = logging.getLogger(__name__)


# ── Default tech stack (fallback when store + SRS provide no preference) ────────
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


# ── System prompt (generic — tech stack is injected per-run in the user prompt) ─
SYSTEM_PROMPT = """\
You are a principal software architect with 20+ years of experience designing \
production-grade, cloud-native systems.

Your job is to produce a complete system architecture document from a Software \
Requirements Specification (SRS).

## Rules
- Respond with a SINGLE valid JSON object — no prose outside the JSON.
- Follow the exact schema below. Every field is REQUIRED.
- Use ONLY the tech stack listed in the ## Resolved Tech Stack section of the \
  user message. Do not substitute or add technologies.
- Be exhaustive: derive every component, endpoint, and relationship that the \
  requirements imply, even if not explicitly stated.
- Do NOT write SQL DDL, Dockerfiles, or source code.
- API contracts: list every REST endpoint the backend must expose. \
  Include realistic request_body and response_body shapes (field names + types as strings).
- Database schema overview: entity names + primary relationships only — no DDL.
- Folder structure: monorepo directory tree as a plain multiline string.
- Communication pattern: describe which services call which and via sync HTTP \
  or async events.
- Security requirements: auth strategy, RBAC roles, data encryption, \
  rate limiting, CORS policy, secrets management.

## JSON schema
{
  "tech_stack": {
    "<layer>": "<technology + version>",
    ...
  },
  "system_components": ["<ServiceName: one-line description>", ...],
  "api_contracts": [
    {
      "method":        "<HTTP verb>",
      "path":          "<URL path>",
      "auth_required": true | false,
      "description":   "<what the endpoint does>",
      "request_body":  { "<field>": "<type as string>", ... },
      "response_body": { "<field>": "<type as string>", ... }
    },
    ...
  ],
  "database_schema_overview": "<multiline string: entity → related entities>",
  "data_flow":               "<multiline string: communication pattern between services>",
  "deployment_strategy":     "<multiline string: folder structure + infra topology>",
  "security_requirements":   ["<requirement statement>", ...]
}
"""

# ── User prompt template ─────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """\
## Resolved Tech Stack
(Use these technologies exactly — they reflect the user's stated preferences \
and/or organisation defaults.)

{tech_stack_table}

---

## Software Requirements Specification

### Functional Requirements
{functional_requirements}

### Non-Functional Requirements
{non_functional_requirements}

### Constraints
{constraints}

### Assumptions
{assumptions}

### Out of Scope
{out_of_scope}

---

Using the SRS and tech stack above, produce the full architecture JSON.
Remember: JSON only — no explanations outside the object.
```json
"""


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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ArchitectureAgent                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class ArchitectureAgent(BaseAgent):
    """
    Reads the completed SRS from ProjectState.requirements and writes a fully
    populated ArchData into ProjectState.architecture.

    Tech stack is resolved at call-time via TechStackResolver so every run
    reflects the latest user preferences from the store.
    """

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ── prompt builder ───────────────────────────────────────────────────────────
    def _build_prompt(self, state: ProjectState) -> str:
        # Resolve tech stack fresh on every run
        resolved_stack = TechStackResolver.resolve(state)
        logger.info(
            "[ArchitectureAgent] resolved tech stack: %s",
            resolved_stack,
        )

        srs = state.requirements
        return USER_PROMPT_TEMPLATE.format(
            tech_stack_table           = _tech_stack_table(resolved_stack),
            functional_requirements    = _bullet_list(srs.functional_requirements),
            non_functional_requirements= _bullet_list(srs.non_functional_requirements),
            constraints                = _bullet_list(srs.constraints),
            assumptions                = _bullet_list(srs.assumptions),
            out_of_scope               = _bullet_list(srs.out_of_scope),
        )

    # ── response parser ──────────────────────────────────────────────────────────
    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        try:
            data: dict[str, Any] = self.extract_json(raw)
        except ValueError as exc:
            logger.error("[ArchitectureAgent] JSON parse failed: %s", exc)
            state.log_error(self.name, f"JSON parse error: {exc}")
            return state

        # ── validate required keys ───────────────────────────────────────────────
        required_keys = {
            "tech_stack",
            "system_components",
            "api_contracts",
            "database_schema_overview",
            "data_flow",
            "deployment_strategy",
            "security_requirements",
        }
        missing = required_keys - data.keys()
        if missing:
            msg = f"LLM response is missing required architecture keys: {missing}"
            logger.error("[ArchitectureAgent] %s", msg)
            state.log_error(self.name, msg)
            return state

        # ── re-resolve stack to guard against LLM drift ─────────────────────────
        # The LLM might still hallucinate its own tech choices despite the prompt;
        # we overwrite the returned tech_stack with the authoritatively resolved one.
        authoritative_stack = TechStackResolver.resolve(state)
        merged_stack = {**data.get("tech_stack", {}), **authoritative_stack}

        # ── fold security_requirements into deployment_strategy ──────────────────
        # ArchData has no dedicated security field; append it as a labelled block
        # so the Security and DevOps agents can parse it from state later.
        security_block = "\n\n## Security Requirements\n" + "\n".join(
            f"  - {r}" for r in _as_list(data.get("security_requirements", []))
        )

        state.architecture = ArchData(
            tech_stack               = merged_stack,
            system_components        = _as_list(data["system_components"]),
            api_contracts            = _validated_contracts(data["api_contracts"]),
            database_schema_overview = str(data["database_schema_overview"]).strip(),
            data_flow                = str(data["data_flow"]).strip(),
            deployment_strategy      = str(data["deployment_strategy"]).strip()
                                       + security_block,
            diagrams                 = [],
        )

        logger.info(
            "[ArchitectureAgent] architecture complete — %d components, %d endpoints.",
            len(state.architecture.system_components),
            len(state.architecture.api_contracts),
        )
        return state


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

    # ── _parse_response: happy path ─────────────────────────────────────────────
    good_arch = json.dumps({
        "tech_stack":               {"backend": "FastAPI", "frontend": "React"},
        "system_components":        ["API Server", "Frontend SPA", "PostgreSQL DB"],
        "api_contracts":            [
            {"method": "GET", "path": "/health", "auth_required": False,
             "description": "Health check", "request_body": {}, "response_body": {"status": "str"}},
        ],
        "database_schema_overview": "User -> Post -> Comment",
        "data_flow":                "Client -> API -> DB",
        "deployment_strategy":      "Docker Compose",
        "security_requirements":    ["JWT auth", "HTTPS only"],
    })

    agent = ArchitectureAgent.__new__(ArchitectureAgent)
    state2 = ProjectState(project_id="t2", project_name="t2")
    state2.requirements = SRSData(
        raw_prompt="Build a blog.", functional_requirements=["Users can post"],
        non_functional_requirements=[], constraints=[], assumptions=[],
        out_of_scope=[], is_complete=True,
    )
    state2 = agent._parse_response(state2, f"```json\n{good_arch}\n```")

    assert len(state2.architecture.system_components) == 3
    assert len(state2.architecture.api_contracts) == 1
    assert "Security Requirements" in state2.architecture.deployment_strategy
    print("[PASS] _parse_response — happy path: arch complete, security block appended")

    # ── _parse_response: missing keys ───────────────────────────────────────────
    bad_arch = json.dumps({"tech_stack": {}, "system_components": []})
    state3 = ProjectState(project_id="t3", project_name="t3")
    state3 = agent._parse_response(state3, bad_arch)
    assert len(state3.errors) > 0, "FAIL: missing keys should log error"
    print("[PASS] _parse_response — missing keys: error logged, state unchanged")

    print("\n" + "=" * 60)
    print("All ArchitectureAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
