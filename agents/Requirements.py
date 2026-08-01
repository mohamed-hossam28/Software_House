import logging
from .base import BaseAgent, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_RETRIES, DEFAULT_BACKOFF
from providers.llm.base import BaseLLMProvider
from core.state import ProjectState, SRSData

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior software requirements analyst with 15+ years of experience.
Your sole job is to transform a user's project description into a complete,
unambiguous Software Requirements Specification (SRS).
 
Output rules:
- Respond ONLY with a single valid JSON object — no preamble, no markdown prose outside the JSON.
- Use the exact keys specified in the schema below.
- Be exhaustive: extract every requirement explicitly stated AND every requirement
  that is obviously implied by the domain.
- If the prompt is clear enough to proceed, leave "clarifying_questions" as an empty list.
- If critical information is genuinely missing (e.g. target platform, scale, auth model),
  list concise questions in "clarifying_questions" and still fill all other fields
  with reasonable assumptions.
 
JSON schema:
{
  "functional_requirements":     ["<verb-noun statement>", ...],
  "non_functional_requirements": ["<quality attribute statement>", ...],
  "constraints":                 ["<hard limit or given>", ...],
  "assumptions":                 ["<assumed fact not stated>", ...],
  "out_of_scope":                ["<explicitly excluded feature>", ...],
  "clarifying_questions":        ["<short question>", ...]
}
"""
 
USER_PROMPT_TEMPLATE = """\
## User Project Prompt
{raw_prompt}

## RAG Context (relevant prior projects / templates)
{rag_context}

## Instructions
Analyze the project prompt above and produce the complete SRS JSON object.
Ensure your response includes all required keys:
- functional_requirements (List of specific user features/actions)
- non_functional_requirements (List of performance, security, reliability attributes)
- constraints (List of technical or business limits)
- assumptions (List of working assumptions)
- out_of_scope (List of excluded features)
- clarifying_questions (List of questions ONLY if critical missing details exist; empty list if clear)

Respond strictly with valid JSON only.
"""
 

class RequirementsAgent(BaseAgent):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = DEFAULT_RETRIES,
        retries: int | None = None,
        backoff: float = DEFAULT_BACKOFF,
        temperature: float = 0.0,
        provider: BaseLLMProvider | str | None = None,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retries=retries,
            backoff=backoff,
            temperature=temperature,
            provider=provider,
        )

    @property
    def system_prompt(self)->str:
        return SYSTEM_PROMPT

    def _build_prompt(self,state:ProjectState)->str:
        raw_prompt=state.requirements.raw_prompt
        rag_context = self._get_rag_context(state)

        return USER_PROMPT_TEMPLATE.format(
            raw_prompt=raw_prompt,
            rag_context=rag_context or "No prior context available."
        )

    
    
    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        
        try:
            from agents.schemas.requirements import RequirementsSchema
            data = self.extract_json_and_validate(raw, RequirementsSchema)
            
        except Exception as exc:
            logger.error("[RequirementsAgent] JSON parse/validation failed: %s", exc)
            state.log_error(self.name, f"JSON parse/validation error: {exc}")
            return state
        
        clarifying_questions = _as_list(data.get("clarifying_questions", []))
 
        # ── Check for missing structural keys ──────────────────────────
        structural_keys = {
            "functional_requirements",
            "non_functional_requirements",
            "constraints",
            "assumptions",
            "out_of_scope",
        }
        missing = structural_keys - data.keys()
        if missing and clarifying_questions:
            # ── Case 1a: missing keys + has questions ──────────────────
            # The LLM is telling us WHY it can't fill the structural fields.
            # Don't treat this as an error — pause and ask the human instead.
            # The missing fields will be filled when the agent re-runs after
            # the human answers.
            logger.info(
                "[RequirementsAgent] structural keys missing %s but LLM raised "
                "%d clarifying question(s) — pausing for human input.",
                missing, len(clarifying_questions)
            )
            state.requirements.clarifying_questions = clarifying_questions
            state.pause(
                raised_by = self.name,
                reason    = (
                    f"Cannot determine {missing} without more information. "
                    "Human input needed."
                ),
                questions = clarifying_questions,
            )
            return state
        

        if missing and not clarifying_questions:
            # ── Case 3: missing keys + no questions ───────────────────
            # The LLM failed silently — it gave us an incomplete response
            # with no explanation. This is a real error — trigger a retry.
            msg = f"LLM response missing structural keys with no explanation: {missing}"
            logger.error("[RequirementsAgent] %s", msg)
            state.log_error(self.name, msg)
            return state


        # ── Case 2: all structural keys present ───────────────────────
        # Write the full SRS into state regardless of whether there are
        # clarifying questions — the LLM may have filled everything with
        # assumptions AND still want to confirm something with the human.
        state.requirements = SRSData(
            raw_prompt                   = state.requirements.raw_prompt,
            functional_requirements      = _as_list(data["functional_requirements"]),
            non_functional_requirements  = _as_list(data["non_functional_requirements"]),
            constraints                  = _as_list(data["constraints"]),
            assumptions                  = _as_list(data["assumptions"]),
            out_of_scope                 = _as_list(data["out_of_scope"]),
            clarifying_questions         = clarifying_questions,
            is_complete                  = len(clarifying_questions) == 0,
        )

        if clarifying_questions:
            # ── Case 2a: all keys present but has questions ────────────
            # LLM filled everything with assumptions but still wants to
            # confirm details. Pause — the re-run will refine the SRS.
            logger.info(
                "[RequirementsAgent] SRS filled with assumptions but %d question(s) "
                "raised — pausing for human confirmation.",
                len(clarifying_questions)
            )
            state.pause(
                raised_by = self.name,
                reason    = "Requirements filled with assumptions. Human confirmation needed.",
                questions = clarifying_questions,
            )
        else:
            # ── Case 2b: all keys present, no questions ────────────────
            # Clean run — SRS is complete, pipeline can continue.
            logger.info(
                "[RequirementsAgent] SRS complete — %d functional, %d non-functional, "
                "%d constraints.",
                len(state.requirements.functional_requirements),
                len(state.requirements.non_functional_requirements),
                len(state.requirements.constraints),
            )
 
        return state


    def _get_rag_context(self, state: ProjectState) -> str:
        """
        Placeholder for RAG retrieval.
        In production this calls rag/retriever.py and returns relevant snippets.
        """
        # TODO: wire up RAG retriever
        # from rag.retriever import retrieve
        # return retrieve(state.requirements.raw_prompt, top_k=3)
        return ""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _as_list(value: object) -> list[str]:
    """Coerce to list[str] — handles None and non-list values gracefully."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Quick smoke-test  (run:  python -m agents.Requirements)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    import json
    
    print("\n" + "=" * 60)
    print("RequirementsAgent — local smoke-test (no LLM call)")
    print("=" * 60)

    from core.state import ProjectState, SRSData

    # ── _as_list helper ─────────────────────────────────────────────────────────
    assert _as_list(None)          == [],           "FAIL: None should give []"
    assert _as_list([])            == [],           "FAIL: empty list should give []"
    assert _as_list(["a", "b"])    == ["a", "b"],   "FAIL: list passthrough"
    assert _as_list("single")      == ["single"],   "FAIL: scalar wrap"
    print("[PASS] _as_list — None, empty list, list, scalar")

    # ── _parse_response: happy path ─────────────────────────────────────────────
    good_json = json.dumps({
        "functional_requirements":     ["User can register", "User can login"],
        "non_functional_requirements": ["Response < 200ms"],
        "constraints":                 ["Must use PostgreSQL"],
        "assumptions":                 ["Single tenant"],
        "out_of_scope":                ["Mobile app"],
        "clarifying_questions":        [],
    })

    agent = RequirementsAgent.__new__(RequirementsAgent)  # bypass __init__ (no LLM)
    state = ProjectState(project_id="test", project_name="test")
    state.requirements.raw_prompt = "Build a todo app."
    state = agent._parse_response(state, f"```json\n{good_json}\n```")

    assert state.requirements.is_complete is True
    assert len(state.requirements.functional_requirements) == 2
    assert state.hitl is None
    print("[PASS] _parse_response — happy path: SRS complete, no HITL pause")

    # ── _parse_response: clarifying questions ───────────────────────────────────
    ambiguous_json = json.dumps({
        "functional_requirements":     ["TBD"],
        "non_functional_requirements": [],
        "constraints":                 [],
        "assumptions":                 [],
        "out_of_scope":                [],
        "clarifying_questions":        ["Is this mobile or web?", "Single or multi-tenant?"],
    })
    state2 = ProjectState(project_id="t2", project_name="t2")
    state2.requirements.raw_prompt = "Build something."
    state2 = agent._parse_response(state2, ambiguous_json)

    assert state2.hitl is not None,             "FAIL: expected HITL pause"
    assert len(state2.hitl.questions) == 2,     "FAIL: expected 2 questions"
    assert state2.requirements.is_complete is False
    print("[PASS] _parse_response — clarifying questions: HITL pause raised")

    # ── _parse_response: bad JSON ───────────────────────────────────────────────
    state3 = ProjectState(project_id="t3", project_name="t3")
    state3 = agent._parse_response(state3, "not json at all")
    assert len(state3.errors) > 0, "FAIL: bad JSON should log an error"
    print("[PASS] _parse_response — bad JSON: error logged, state unchanged")

    print("\n" + "=" * 60)
    print("All RequirementsAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
