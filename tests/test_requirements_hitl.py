import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import pytest

from providers.llm.base import BaseLLMProvider, LLMResponse
from core.state import HITLRequest, ProjectState
from agents.Requirements import RequirementsAgent, USER_PROMPT_TEMPLATE
from graph.edges import route_after_requirements


#: Instruction the final (bypass) requirements call must carry so every
#: unstated premise lands in the `assumptions` list instead of staying silent.
BYPASS_INSTRUCTION = "Clarification rounds exhausted"

GOOD_SRS = json.dumps({
    "functional_requirements":     ["User can register"],
    "non_functional_requirements": ["Response < 200ms"],
    "constraints":                 ["Must use PostgreSQL"],
    "assumptions":                 ["Single tenant"],
    "out_of_scope":                ["Mobile app"],
    "clarifying_questions":        [],
})


class CapturingProvider(BaseLLMProvider):
    """Records every user prompt it receives and replies with a valid SRS."""

    def __init__(self, text: str = GOOD_SRS) -> None:
        self.text = text
        self.prompts: list[str] = []

    @property
    def provider_name(self) -> str:
        return "capturing"

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.2,
        stream: bool = False,
        chunk_callback=None,
    ) -> LLMResponse:
        self.prompts.append(user_message)
        return LLMResponse(text=self.text, input_tokens=10, output_tokens=10)


def make_state(rounds: int, max_rounds: int = 3, resolved_hitl: bool = True) -> ProjectState:
    """State as it looks when the requirements agent is (re-)entered."""
    ps = ProjectState(
        project_id="t",
        project_name="t",
        max_clarfication_rounds=max_rounds,
        clarfication_rounds=rounds,
    )
    ps.requirements.raw_prompt = "Build a todo app."
    # The human node answered the previous round (is_resolved=True) before the
    # graph looped back into the requirements node; router tests flip it back
    # to unresolved to reproduce the pause the router must judge.
    ps.hitl = HITLRequest(
        raised_by="RequirementsAgent",
        reason="Confirmation needed.",
        questions=["Mobile or web?"],
        answers=["Web."] if resolved_hitl else [],
        is_resolved=resolved_hitl,
    )
    return ps


class TestBypassAssumptionsInstruction:
    def test_final_call_when_rounds_exhausted_carries_instruction(self):
        provider = CapturingProvider()
        agent = RequirementsAgent(provider=provider)
        state = make_state(rounds=3, max_rounds=3)

        agent.run(state)

        assert len(provider.prompts) == 1
        prompt = provider.prompts[0]
        assert BYPASS_INSTRUCTION in prompt
        assert "assumptions" in prompt

    def test_final_call_instruction_mentions_assumptions_list(self):
        provider = CapturingProvider()
        agent = RequirementsAgent(provider=provider)
        agent.run(make_state(rounds=3, max_rounds=3))

        prompt = provider.prompts[0]
        # The whole point: guesses must be recorded, not kept silent.
        assert "assumptions" in prompt
        assert "MUST" in prompt


class TestNonBypassUnchanged:
    @pytest.mark.parametrize("rounds", [0, 1, 2])
    def test_earlier_rounds_do_not_carry_instruction(self, rounds):
        provider = CapturingProvider()
        agent = RequirementsAgent(provider=provider)

        agent.run(make_state(rounds=rounds, max_rounds=3))

        prompt = provider.prompts[0]
        assert BYPASS_INSTRUCTION not in prompt

    def test_first_call_prompt_is_byte_for_byte_the_plain_template(self):
        provider = CapturingProvider()
        agent = RequirementsAgent(provider=provider)
        state = make_state(rounds=0)

        agent.run(state)

        expected = USER_PROMPT_TEMPLATE.format(
            raw_prompt=state.requirements.raw_prompt,
            rag_context="No prior context available.",
        )
        assert provider.prompts[0] == expected


class TestRouterBypass:
    def test_rounds_exhausted_routes_to_architecture(self):
        state = make_state(rounds=3, max_rounds=3, resolved_hitl=False)

        assert route_after_requirements(state) == ["architecture"]

    def test_rounds_remaining_routes_to_human(self):
        state = make_state(rounds=2, max_rounds=3, resolved_hitl=False)

        assert route_after_requirements(state) == ["human"]
