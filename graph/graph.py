import os
import sys
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# Force workspace root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import settings
from core.state import ProjectState
from core.workspace import Workspace
from graph.nodes import (
    make_requirements_node,
    human_clarifying_questions_node,
    make_architecture_node,
    make_planner_node,
    make_database_node,
    make_backend_node,
    make_frontend_node,
    make_integration_node,
    make_finalize_node,
)
from graph.edges import route_after_requirements
from providers.llm.base import BaseLLMProvider


@dataclass(frozen=True)
class AgentProviders:
    """
    One LLM provider per agent: a provider instance, or a provider name the
    agent resolves through the LLM factory when it is constructed.
    """
    requirements: BaseLLMProvider | str
    architecture: BaseLLMProvider | str
    planner: BaseLLMProvider | str
    database: BaseLLMProvider | str
    backend: BaseLLMProvider | str
    frontend: BaseLLMProvider | str
    integration: BaseLLMProvider | str

    @classmethod
    def from_settings(cls) -> "AgentProviders":
        """The providers chosen by settings: the default provider for planning
        agents and Integration, the coding provider for the coding agents."""
        planning, coding = settings.default_provider, settings.coding_provider_name
        return cls(
            requirements=planning,
            architecture=planning,
            planner=planning,
            database=coding,
            backend=coding,
            frontend=coding,
            integration=planning,
        )


def build_graph(
    providers: AgentProviders | None = None,
    workspace: Workspace | None = None,
    checkpointer=None,
):
    """
    Build and compile the pipeline graph.

    Every node closes over its agent's provider and the *workspace* generated
    files are written under. Defaults: the providers chosen by settings, the
    real project directory, and an in-memory checkpointer.
    """
    providers = providers or AgentProviders.from_settings()
    workspace = workspace or Workspace()

    # Define the StateGraph with ProjectState
    workflow = StateGraph(ProjectState)

    # Add Nodes
    workflow.add_node("requirements", make_requirements_node(providers.requirements))
    workflow.add_node("human", human_clarifying_questions_node)
    workflow.add_node("architecture", make_architecture_node(providers.architecture))
    workflow.add_node("planner", make_planner_node(providers.planner))
    workflow.add_node("database", make_database_node(providers.database, workspace))
    workflow.add_node("backend", make_backend_node(providers.backend, workspace))
    workflow.add_node("frontend", make_frontend_node(providers.frontend, workspace))
    workflow.add_node("integration", make_integration_node(providers.integration, workspace))
    workflow.add_node("finalize", make_finalize_node(workspace))

    # Add Edges
    workflow.add_edge(START, "requirements")

    # Route conditionally after requirements node (to human if needed, else to architecture)
    workflow.add_conditional_edges(
        "requirements",
        route_after_requirements,
        {
            "human": "human",
            "architecture": "architecture",
        }
    )

    # Loop back from human clarifying node to requirements for re-evaluation
    workflow.add_edge("human", "requirements")

    # Architecture routes to planner
    workflow.add_edge("architecture", "planner")

    # Planner routes to backend, frontend, and database in parallel
    workflow.add_edge("planner", "backend")
    workflow.add_edge("planner", "frontend")
    workflow.add_edge("planner", "database")

    # Engineering agents converge into the integration node (static checks),
    # which then hands off to finalize (persists the run summary next to output/).
    workflow.add_edge("backend", "integration")
    workflow.add_edge("frontend", "integration")
    workflow.add_edge("database", "integration")

    workflow.add_edge("integration", "finalize")

    workflow.add_edge("finalize", END)

    # Compile the workflow with a checkpointer to remember context
    return workflow.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


# The default graph `main.py` runs: providers chosen by settings, the real output directory.
graph = build_graph()
