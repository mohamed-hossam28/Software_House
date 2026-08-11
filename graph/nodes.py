import logging
import os
import sys
import json
from threading import Lock

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

# Import core state and statuses
from core.state import ProjectState, ProjectStatus
from core.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _state_from_dict(state: dict | ProjectState) -> ProjectState:
    """Reconstruct a ProjectState from a graph state dict or return as-is if already ProjectState."""
    if isinstance(state, ProjectState):
        return state
    return ProjectState(**state)


# ── Node Definitions ─────────────────────────────────────────────────────────

# ── Node Definitions ─────────────────────────────────────────────────────────

def requirements_node(state: dict) -> dict:
    """
    Requirements node: runs the RequirementsAgent to generate the SRS.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running RequirementsAgent...")
    agent = RequirementsAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.req_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.0,
    )
    return agent.run(state)


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

    return BaseAgent._compute_diff(before, ps)


def architecture_node(state: dict) -> dict:
    """
    Architecture node: runs the ArchitectureAgent to generate design specs and API contracts.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running ArchitectureAgent...")
    agent = ArchitectureAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.arch_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.0,
    )
    return agent.run(state)


def planner_node(state: dict) -> dict:
    """
    Planner node: runs the PlannerAgent to decompose the design into prioritized files and tasks.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running PlannerAgent...")
    agent = PlannerAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.planner_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.0,
    )
    return agent.run(state)


def database_node(state: dict) -> dict:
    """
    Database node: runs the DatabaseAgent to generate SQL and schema structures.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running DatabaseAgent...")
    agent = DatabaseAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.db_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.2,
    )
    return agent.run(state)


def backend_node(state: dict) -> dict:
    """
    Backend node: runs the BackendAgent to generate backend endpoint logic.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running BackendAgent...")
    agent = BackendAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.be_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.2,
    )
    return agent.run(state)


def frontend_node(state: dict) -> dict:
    """
    Frontend node: runs the FrontendAgent to generate user interface components and styling.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running FrontendAgent...")
    agent = FrontendAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.fe_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.2,
    )
    return agent.run(state)


def integration_node(state: dict) -> dict:
    """
    Integration node: runs the IntegrationAgent to cross-verify files, API contracts, and resolve conflicts.
    Receives full state dict, returns only changed fields as dict.
    """
    logger.info("[GraphNode] Running IntegrationAgent...")
    agent = IntegrationAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.integration_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
        temperature=0.0,
    )
    return agent.run(state)


def finalize_node(state: dict) -> dict:
    """
    Finalize node: persists the run summary (manifest + conflicts + fingerprints)
    next to the generated project as `output/run_summary.json`. Never fails the run.
    """
    from agents.utils import write_run_summary

    logger.info("[GraphNode] Finalizing — writing run summary...")
    ps = _state_from_dict(state)
    path = write_run_summary(ps)
    logger.info("[GraphNode] Run summary written: %s", path or "(skipped)")
    return {}

