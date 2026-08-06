import logging
from typing import Any
from core.state import ProjectState, ProjectStatus

logger = logging.getLogger(__name__)

def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to extract a field safely from either a dict or a Pydantic state instance."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

#: Mapping from owning-agent name (as recorded by IntegrationStatus.conflicts_by_agent)
#: to the graph node that owns that code, so the REWORK loop targets the right agent.
AGENT_TO_NODE = {
    "backend": "backend",
    "frontend": "frontend",
    "database": "database",
}


def route_after_integration(state: Any) -> list[str]:
    """
    Mod 1 — decide where to go after the Integration Agent.

    - If the code merged cleanly → END.
    - If some conflicts remain, route back to the coding agents that own them for a
      REWORK round (integration → {backend, frontend, database}), capped by
      `integration_rework_rounds` vs `max_integration_rounds`.
    - When the cap is reached, proceed to END with the remaining conflicts logged.
    """
    integration = _get_field(state, "integration")
    merged = bool(_get_field(integration, "merged", False))
    conflicts_by_agent = _get_field(integration, "conflicts_by_agent", {}) or {}
    rounds = _get_field(state, "integration_rework_rounds", 0) or 0
    max_rounds = _get_field(state, "max_integration_rounds", 2) or 2

    if merged:
        logger.info("[Router] Integration merged cleanly. END.")
        return ["__end__"]

    next_nodes = [AGENT_TO_NODE[a] for a in conflicts_by_agent if a in AGENT_TO_NODE]

    if not next_nodes:
        logger.warning("[Router] Integration unresolved but no matching coding agent for conflicts. END.")
        return ["__end__"]

    if rounds >= max_rounds:
        logger.warning(
            "[Router] Integration REWORK cap reached (%d/%d). Remaining conflicts (%d) are left unresolved. END.",
            rounds, max_rounds, len(conflicts_by_agent),
        )
        return ["__end__"]

    logger.info(
        "[Router] Integration REWORK round %d/%d. Routing back to: %s",
        rounds + 1, max_rounds, next_nodes,
    )
    return next_nodes

def route_after_requirements(state: Any) -> list[str]:
    """
    Decides whether to route to the human interaction node or proceed to architecture.
    Handles both dict and ProjectState instances defensively.
    """
    status = _get_field(state, "status", ProjectStatus.PENDING)
    hitl = _get_field(state, "hitl")
    clarfication_rounds = _get_field(state, "clarfication_rounds", 0)
    max_clarfication_rounds = _get_field(state, "max_clarfication_rounds", 3)

    is_awaiting = status in (ProjectStatus.AWAITING_HUMAN, ProjectStatus.AWAITING_HUMAN.value, "awaiting_human")
    
    is_resolved = True
    if hitl is not None:
        is_resolved = bool(_get_field(hitl, "is_resolved", False))

    has_unresolved_hitl = hitl is not None and not is_resolved

    if is_awaiting or has_unresolved_hitl:
        if clarfication_rounds < max_clarfication_rounds:
            logger.info(f"[Router] Clarifying questions detected (Round {clarfication_rounds + 1}/{max_clarfication_rounds}). Routing to human.")
            return ["human"]
        else:
            logger.warning(f"[Router] Max clarification rounds ({max_clarfication_rounds}) reached. Bypassing human interaction and routing to architecture.")
            return ["architecture"]
    logger.info("[Router] SRS complete. Routing to architecture.")
    return ["architecture"]
