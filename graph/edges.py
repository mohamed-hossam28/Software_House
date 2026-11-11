import logging
from typing import Any
from core.state import ProjectState, ProjectStatus

logger = logging.getLogger(__name__)

def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to extract a field safely from either a dict or a Pydantic state instance."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

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
