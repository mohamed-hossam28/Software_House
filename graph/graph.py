import os
import sys
from langgraph.graph import StateGraph, START, END

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
)
from graph.edges import route_after_requirements

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

# Add Edges
workflow.add_edge(START, "requirements")

# Route conditionally after requirements node
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

# Standard linear stages after architecture
workflow.add_edge("architecture", "planner")
workflow.add_edge("planner", "database")

# Sequential execution of engineering and integration
workflow.add_edge("database", "backend")
workflow.add_edge("backend", "frontend")
workflow.add_edge("frontend", "integration")
workflow.add_edge("integration", END)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Compile the workflow with memory checkpointer to remember context
graph = workflow.compile(checkpointer=memory)
