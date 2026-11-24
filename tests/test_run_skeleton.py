"""Ticket #34: one run skeleton for all seven agents.

Every agent is driven through its public ``run(state)``, the call shape the
graph nodes use, and only the returned changes are checked: the agent's
duration lands in the metrics, and early-return paths still send the right
changes.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.architecture import ArchitectureAgent
from agents.backend import BackendAgent
from agents.base import BaseAgent
from agents.data_base import DatabaseAgent
from agents.frontend import FrontendAgent
from agents.integration import IntegrationAgent
from agents.planner import PlannerAgent
from agents.Requirements import RequirementsAgent
from core.state import ProjectState, StaticVerdict
from providers.llm.base import BaseLLMProvider, LLMResponse

ALL_AGENTS = (
    RequirementsAgent,
    ArchitectureAgent,
    PlannerAgent,
    DatabaseAgent,
    BackendAgent,
    FrontendAgent,
    IntegrationAgent,
)

#: Agents whose work step calls the LLM even on an empty project.
LLM_AGENTS = (RequirementsAgent, ArchitectureAgent, PlannerAgent)

LLM_LATENCY = 0.02


class SlowFailingProvider(BaseLLMProvider):
    """Every call takes LLM_LATENCY seconds and then fails."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "slow-failing"

    def complete(self, system_prompt, user_message, model="", max_tokens=8192,
                 temperature=0.2, stream=False, chunk_callback=None) -> LLMResponse:
        self.calls += 1
        time.sleep(LLM_LATENCY)
        raise RuntimeError("LLM unavailable")


class FixedProvider(BaseLLMProvider):
    """Every call answers with the same text."""

    def __init__(self, text: str) -> None:
        self.text = text

    @property
    def provider_name(self) -> str:
        return "fixed"

    def complete(self, system_prompt, user_message, model="", max_tokens=8192,
                 temperature=0.2, stream=False, chunk_callback=None) -> LLMResponse:
        return LLMResponse(text=self.text, input_tokens=10, output_tokens=10)


def empty_project() -> ProjectState:
    state = ProjectState(project_id="skeleton", project_name="Skeleton")
    state.requirements.raw_prompt = "Build a todo app."
    return state


def make_agent(agent_cls: type[BaseAgent], provider: BaseLLMProvider) -> BaseAgent:
    return agent_cls(provider=provider, retries=1, backoff=0.0)


# ── Duration ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("agent_cls", ALL_AGENTS, ids=lambda c: c.__name__)
def test_every_agent_records_its_duration_and_marks_its_start(agent_cls):
    agent = make_agent(agent_cls, SlowFailingProvider())

    changes = agent.run(empty_project())

    assert changes["current_agent"] == agent.name
    duration = changes["metrics"]["agent_durations"][agent.name]
    assert isinstance(duration, float) and duration >= 0.0


@pytest.mark.parametrize("agent_cls", LLM_AGENTS, ids=lambda c: c.__name__)
def test_recorded_duration_covers_the_work_step(agent_cls):
    provider = SlowFailingProvider()
    agent = make_agent(agent_cls, provider)

    changes = agent.run(empty_project())

    assert provider.calls >= 1
    # Calls may run in parallel, so only one call's latency is guaranteed.
    assert changes["metrics"]["agent_durations"][agent.name] >= LLM_LATENCY


def test_run_accepts_the_graph_state_dict_and_leaves_it_untouched():
    state = empty_project().model_dump(mode="json")
    snapshot = ProjectState(**state).model_dump(mode="json")

    changes = make_agent(PlannerAgent, SlowFailingProvider()).run(state)

    assert "PlannerAgent" in changes["metrics"]["agent_durations"]
    assert state == snapshot


# ── Early returns ────────────────────────────────────────────────────────────


def test_planner_skeleton_failure_returns_only_its_error_and_duration():
    changes = make_agent(PlannerAgent, SlowFailingProvider()).run(empty_project())

    assert set(changes) == {"current_agent", "metrics", "errors"}
    assert len(changes["errors"]) == 1
    assert "skeleton call failed" in changes["errors"][0]["message"]
    assert "PlannerAgent" in changes["metrics"]["agent_durations"]


def test_planner_empty_skeleton_returns_no_plan():
    changes = make_agent(PlannerAgent, FixedProvider('{"skeleton": []}')).run(empty_project())

    assert "tasks" not in changes
    assert "code_manifest" not in changes
    assert [e["message"] for e in changes["errors"]] == ["No skeleton produced — check LLM response."]
    assert changes["metrics"]["total_llm_calls"] == 1
    assert "PlannerAgent" in changes["metrics"]["agent_durations"]


@pytest.mark.parametrize("agent_cls", (DatabaseAgent, BackendAgent, FrontendAgent),
                         ids=lambda c: c.__name__)
def test_coding_agent_without_tasks_returns_only_its_start_and_duration(agent_cls):
    agent = make_agent(agent_cls, SlowFailingProvider())

    changes = agent.run(empty_project())

    assert set(changes) == {"current_agent", "metrics"}
    assert set(changes["metrics"]["agent_durations"]) == {agent.name}
    assert changes["metrics"]["total_llm_calls"] == 0


def test_integration_on_an_empty_project_passes_without_the_llm():
    provider = SlowFailingProvider()

    changes = make_agent(IntegrationAgent, provider).run(empty_project())

    assert provider.calls == 0
    assert changes["static_checks"]["verdict"] == StaticVerdict.PASS.value
    assert "errors" not in changes
    assert "IntegrationAgent" in changes["metrics"]["agent_durations"]


# ── One skeleton ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("agent_cls", ALL_AGENTS, ids=lambda c: c.__name__)
def test_no_agent_reimplements_the_run_skeleton(agent_cls):
    """Ticket #34 acceptance: each agent supplies only its work step."""
    assert agent_cls.run is BaseAgent.run
