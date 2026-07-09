import logging
from core.state import ProjectState, ProjectStatus

logger = logging.getLogger(__name__)

def route_after_requirements(state: ProjectState) -> str:
    """
    Decides whether to route to the human interaction node or proceed to architecture.
    """
    if state.status == ProjectStatus.AWAITING_HUMAN or (state.hitl and not state.hitl.is_resolved):
        if state.clarfication_rounds < state.max_clarfication_rounds:
            logger.info(f"[Router] Clarifying questions detected (Round {state.clarfication_rounds + 1}/{state.max_clarfication_rounds}). Routing to human_node.")
            return "human"
        else:
            logger.warning(f"[Router] Max clarification rounds ({state.max_clarfication_rounds}) reached. Bypassing human interaction and routing to architecture.")
            # Revert status to RUNNING to avoid blocking downstream execution
            state.status = ProjectStatus.RUNNING
            return "architecture"
    logger.info("[Router] SRS complete. Routing to architecture_node.")
    return "architecture"
