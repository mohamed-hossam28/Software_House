import os
import sys
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# Force workspace root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.state import ProjectState
from graph.nodes import (
    requirements_node,
    human_clarifying_questions_node,
    architecture_node,
    planner_node,
    database_node,
    backend_node,
    frontend_node,
    integration_node,
    finalize_node,
)
from graph.edges import route_after_requirements, route_after_integration

# Define the StateGraph with ProjectState
workflow = StateGraph(ProjectState)

# Add Nodes
workflow.add_node("requirements", requirements_node)
workflow.add_node("human", human_clarifying_questions_node)
workflow.add_node("architecture", architecture_node)
workflow.add_node("planner", planner_node)
workflow.add_node("database", database_node)
workflow.add_node("backend", backend_node)
workflow.add_node("frontend", frontend_node)
workflow.add_node("integration", integration_node)
workflow.add_node("finalize", finalize_node)

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

# Engineering agents merge into integration
workflow.add_edge("backend", "integration")
workflow.add_edge("frontend", "integration")
workflow.add_edge("database", "integration")

# Mod 1 — integration REWORK loop back to the coding agents when conflicts remain,
# else END via the finalize node (persists the run summary next to output/).
workflow.add_conditional_edges(
    "integration",
    route_after_integration,
    {
        "backend": "backend",
        "frontend": "frontend",
        "database": "database",
        "__end__": "finalize",
    },
)
workflow.add_edge("finalize", END)


memory = MemorySaver()

# Compile the workflow with memory checkpointer to remember context
graph = workflow.compile(checkpointer=memory)
