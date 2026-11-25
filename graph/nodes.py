import logging
import os
import sys
import json
from threading import Lock
from typing import Callable

# Force workspace root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the agents
from agents.base import BaseAgent
from agents.Requirements import RequirementsAgent
from agents.architecture import ArchitectureAgent
from agents.planner import PlannerAgent
from agents.data_base import DatabaseAgent
from agents.backend import BackendAgent
from agents.frontend import FrontendAgent
from agents.integration import IntegrationAgent
from agents.docs import DocsAgent

# Import core state and statuses
from core.state import ProjectState, ProjectStatus, diff_state
from core.config import settings
from core.workspace import Workspace
from providers.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _state_from_dict(state: dict | ProjectState) -> ProjectState:
    """Reconstruct a ProjectState from a graph state dict or return as-is if already ProjectState."""
    if isinstance(state, ProjectState):
        return state
    return ProjectState(**state)


# ── Node Definitions ─────────────────────────────────────────────────────────

# ── Node Definitions ─────────────────────────────────────────────────────────

def make_requirements_node(provider: BaseLLMProvider | str) -> Callable[[dict], dict]:
    """
    Requirements node: runs the RequirementsAgent to generate the SRS.
    Receives full state dict, returns only changed fields as dict.
    """
    model, max_tokens = settings.model_name, settings.req_max_tokens
    retries, backoff = settings.default_retries, settings.default_backoff

    def requirements_node(state: dict) -> dict:
        logger.info("[GraphNode] Running RequirementsAgent...")
        agent = RequirementsAgent(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
            backoff=backoff,
            temperature=0.0,
        )
        return agent.run(state)

    return requirements_node


def human_clarifying_questions_node(state: dict) -> dict:
    """
    Human-in-the-loop node: handles clarifying questions raised by the RequirementsAgent.
    If running in an interactive terminal, prompts the user via stdin.
    Otherwise, pauses execution and waits for external input/resolution.
    Returns only the fields that changed.
    """
    ps = _state_from_dict(state)

    if not ps.hitl or ps.hitl.is_resolved:
        return {}

    before = ps.model_copy(deep=True)

    ps.clarfication_rounds += 1

    print("\n" + "=" * 60)
    print(f"HUMAN INPUT REQUIRED (Agent: {ps.hitl.raised_by})")
    print(f"Reason: {ps.hitl.reason}")
    print("=" * 60)

    # Check if we are in an interactive console
    if sys.stdin.isatty():
        answers = []
        for idx, q in enumerate(ps.hitl.questions):
            print(f"\n[Question {idx + 1}/{len(ps.hitl.questions)}]: {q}")
            ans = input("Answer: ").strip()
            while not ans:
                ans = input("Answer cannot be empty. Please enter a valid response: ").strip()
            answers.append(ans)

        ps.resume(answers)

        # Append Q&A context to the requirements raw prompt for the agent's re-run
        q_a_block = "\n\n## Clarifying Q&A:\n"
        for q, a in zip(ps.hitl.questions, answers):
            q_a_block += f"- **Question**: {q}\n  **Answer**: {a}\n"
        ps.requirements.raw_prompt += q_a_block

        print("\n" + "=" * 60)
        print("Answers successfully recorded. Resuming graph execution...")
        print("=" * 60 + "\n")
    else:
        logger.warning("[GraphNode] Non-interactive environment detected. Graph execution paused.")
        mock_answers_env = os.getenv("MOCK_HITL_ANSWERS")
        if mock_answers_env:
            try:
                answers = json.loads(mock_answers_env)
                ps.resume(answers)
                logger.info("[GraphNode] HITL resolved using MOCK_HITL_ANSWERS env var.")
            except Exception as e:
                answers = ["Yes, proceed with defaults."] * len(ps.hitl.questions)
                ps.resume(answers)
                logger.info(f"[GraphNode] HITL resolved using fallback defaults due to error: {e}")
        else:
            try:
                from langgraph.errors import NodeInterrupt
                raise NodeInterrupt(f"Awaiting human response for questions: {ps.hitl.questions}")
            except ImportError:
                logger.info("[GraphNode] Langsmith/Langgraph NodeInterrupt not available. Waiting...")

    return diff_state(before, ps)


def make_architecture_node(provider: BaseLLMProvider | str) -> Callable[[dict], dict]:
    """
    Architecture node: runs the ArchitectureAgent to generate design specs and API contracts.
    Receives full state dict, returns only changed fields as dict.
    """
    model, max_tokens = settings.model_name, settings.arch_max_tokens
    retries, backoff = settings.default_retries, settings.default_backoff

    def architecture_node(state: dict) -> dict:
        logger.info("[GraphNode] Running ArchitectureAgent...")
        agent = ArchitectureAgent(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
            backoff=backoff,
            temperature=0.0,
        )
        return agent.run(state)

    return architecture_node


def make_planner_node(provider: BaseLLMProvider | str) -> Callable[[dict], dict]:
    """
    Planner node: runs the PlannerAgent to decompose the design into prioritized files and tasks.
    Receives full state dict, returns only changed fields as dict.
    """
    model, max_tokens = settings.model_name, settings.planner_max_tokens
    retries, backoff = settings.default_retries, settings.default_backoff

    def planner_node(state: dict) -> dict:
        logger.info("[GraphNode] Running PlannerAgent...")
        agent = PlannerAgent(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
            backoff=backoff,
            temperature=0.0,
        )
        return agent.run(state)

    return planner_node


def _make_coding_node(
    agent_cls: type[BaseAgent], provider: BaseLLMProvider | str, max_tokens: int, workspace: Workspace,
) -> Callable[[dict], dict]:
    """A coding agent node: runs *agent_cls*, writing its files through *workspace*."""
    model = settings.coding_model_name
    retries, backoff = settings.default_retries, settings.default_backoff

    def coding_node(state: dict) -> dict:
        logger.info("[GraphNode] Running %s...", agent_cls.__name__)
        agent = agent_cls(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            retries=retries,
            backoff=backoff,
            temperature=0.2,
            workspace=workspace,
        )
        return agent.run(state)

    return coding_node


def make_database_node(provider: BaseLLMProvider | str, workspace: Workspace) -> Callable[[dict], dict]:
    """Database node: runs the DatabaseAgent to generate SQL and schema structures."""
    return _make_coding_node(DatabaseAgent, provider, settings.db_max_tokens, workspace)


def make_backend_node(provider: BaseLLMProvider | str, workspace: Workspace) -> Callable[[dict], dict]:
    """Backend node: runs the BackendAgent to generate backend endpoint logic."""
    return _make_coding_node(BackendAgent, provider, settings.be_max_tokens, workspace)


def make_frontend_node(provider: BaseLLMProvider | str, workspace: Workspace) -> Callable[[dict], dict]:
    """Frontend node: runs the FrontendAgent to generate user interface components and styling."""
    return _make_coding_node(FrontendAgent, provider, settings.fe_max_tokens, workspace)


def make_integration_node(provider: BaseLLMProvider | str, workspace: Workspace) -> Callable[[dict], dict]:
    """
    Integration node: runs the IntegrationAgent's static checks over everything
    the coding agents generated, repairs error findings through *workspace*, and
    records the verdict in state.static_checks. Error-level findings flow into
    the state error list. Returns only changed fields as dict.
    """
    model = settings.model_name
    retries, backoff = settings.default_retries, settings.default_backoff

    def integration_node(state: dict) -> dict:
        logger.info("[GraphNode] Running IntegrationAgent (static checks)...")
        agent = IntegrationAgent(
            provider=provider,
            model=model,
            retries=retries,
            backoff=backoff,
            workspace=workspace,
        )
        return agent.run(state)

    return integration_node


def make_docs_node(provider: BaseLLMProvider | str, workspace: Workspace) -> Callable[[dict], dict]:
    """
    Docs node: runs the DocsAgent after Integration, writing the README from
    the final, repaired files through *workspace* with the backend coding prompt.
    """
    return _make_coding_node(DocsAgent, provider, settings.be_max_tokens, workspace)


def make_finalize_node(workspace: Workspace) -> Callable[[dict], dict]:
    """
    Finalize node: persists the run summary and complete static-check findings
    report next to the generated project in *workspace*. Never fails the run.
    """
    from agents.utils import write_run_summary, write_static_checks_report

    def finalize_node(state: dict) -> dict:
        logger.info("[GraphNode] Finalizing — writing run summary...")
        ps = _state_from_dict(state)

        report_path = write_static_checks_report(ps, workspace)
        path = write_run_summary(ps, static_report_path=report_path, workspace=workspace)
        logger.info("[GraphNode] Static findings report written: %s", report_path or "(skipped)")
        logger.info("[GraphNode] Run summary written: %s", path or "(skipped)")
        return {}

    return finalize_node
