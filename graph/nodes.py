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

# ── Node Definitions ─────────────────────────────────────────────────────────

def requirements_node(state: ProjectState) -> ProjectState:
    """
    Requirements node: runs the RequirementsAgent to generate the SRS.
    """
    logger.info("[GraphNode] Running RequirementsAgent...")
    agent = RequirementsAgent(
        provider=settings.default_provider,
        model=settings.model_name,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


def human_clarifying_questions_node(state: ProjectState) -> ProjectState:
    """
    Human-in-the-loop node: handles clarifying questions raised by the RequirementsAgent.
    If running in an interactive terminal, prompts the user via stdin.
    Otherwise, pauses execution and waits for external input/resolution.
    """
    if not state.hitl or state.hitl.is_resolved:
        return state

    state.clarfication_rounds += 1

    print("\n" + "=" * 60)
    print(f"HUMAN INPUT REQUIRED (Agent: {state.hitl.raised_by})")
    print(f"Reason: {state.hitl.reason}")
    print("=" * 60)

    # Check if we are in an interactive console
    if sys.stdin.isatty():
        answers = []
        for idx, q in enumerate(state.hitl.questions):
            print(f"\n[Question {idx + 1}/{len(state.hitl.questions)}]: {q}")
            ans = input("Answer: ").strip()
            while not ans:
                ans = input("Answer cannot be empty. Please enter a valid response: ").strip()
            answers.append(ans)

        state.resume(answers)

        # Append Q&A context to the requirements raw prompt for the agent's re-run
        q_a_block = "\n\n## Clarifying Q&A:\n"
        for q, a in zip(state.hitl.questions, answers):
            q_a_block += f"- **Question**: {q}\n  **Answer**: {a}\n"
        state.requirements.raw_prompt += q_a_block

        print("\n" + "=" * 60)
        print("Answers successfully recorded. Resuming graph execution...")
        print("=" * 60 + "\n")
    else:
        logger.warning("[GraphNode] Non-interactive environment detected. Graph execution paused.")
        # If mock answers environment variable is present, resolve automatically (useful for testing)
        mock_answers_env = os.getenv("MOCK_HITL_ANSWERS")
        if mock_answers_env:
            try:
                answers = json.loads(mock_answers_env)
                state.resume(answers)
                logger.info("[GraphNode] HITL resolved using MOCK_HITL_ANSWERS env var.")
            except Exception as e:
                # Fallback to default answers
                answers = ["Yes, proceed with defaults."] * len(state.hitl.questions)
                state.resume(answers)
                logger.info(f"[GraphNode] HITL resolved using fallback defaults due to error: {e}")
        else:
            # Under LangGraph, to pause/interrupt execution, we can raise a NodeInterrupt
            try:
                from langgraph.errors import NodeInterrupt
                raise NodeInterrupt(f"Awaiting human response for questions: {state.hitl.questions}")
            except ImportError:
                # If running in a context where NodeInterrupt is not imported, we just log and return
                logger.info("[GraphNode] Langsmith/Langgraph NodeInterrupt not available. Waiting...")

    return state


def architecture_node(state: ProjectState) -> ProjectState:
    """
    Architecture node: runs the ArchitectureAgent to generate design specs and API contracts.
    """
    logger.info("[GraphNode] Running ArchitectureAgent...")
    agent = ArchitectureAgent(
        provider=settings.default_provider,
        model=settings.default_model,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


def planner_node(state: ProjectState) -> ProjectState:
    """
    Planner node: runs the PlannerAgent to decompose the design into prioritized files and tasks.
    """
    logger.info("[GraphNode] Running PlannerAgent...")
    agent = PlannerAgent(
        provider=settings.default_provider,
        model=settings.default_model,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


def database_node(state: ProjectState) -> ProjectState:
    """
    Database node: runs the DatabaseAgent to generate SQL and schema structures.
    """
    logger.info("[GraphNode] Running DatabaseAgent...")
    agent = DatabaseAgent(
        provider=settings.default_provider,
        model=settings.default_model,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


def backend_node(state: ProjectState) -> ProjectState:
    """
    Backend node: runs the BackendAgent to generate backend endpoint logic.
    """
    logger.info("[GraphNode] Running BackendAgent...")
    agent = BackendAgent(
        provider=settings.default_provider,
        model=settings.default_model,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


def frontend_node(state: ProjectState) -> ProjectState:
    """
    Frontend node: runs the FrontendAgent to generate user interface components and styling.
    """
    logger.info("[GraphNode] Running FrontendAgent...")
    agent = FrontendAgent(
        provider=settings.default_provider,
        model=settings.default_model,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


def integration_node(state: ProjectState) -> ProjectState:
    """
    Integration node: runs the IntegrationAgent to cross-verify files, API contracts, and resolve conflicts.
    """
    logger.info("[GraphNode] Running IntegrationAgent...")
    agent = IntegrationAgent(
        provider=settings.default_provider,
        model=settings.default_model,
        max_tokens=settings.default_max_tokens,
        retries=settings.default_retries,
        backoff=settings.default_backoff,
    )
    return agent.run(state)


