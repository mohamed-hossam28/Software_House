from pydantic import BaseModel, Field
from typing import List, Optional

class FixSchema(BaseModel):
    action: Optional[str] = None
    filepath: Optional[str] = None
    target_content: Optional[str] = None
    replacement_content: Optional[str] = None

class RepairSchema(BaseModel):
    """
    A single mechanical repair for one integration conflict.
    No HUMAN classification exists — every conflict the model addresses must
    carry a concrete fix, or `action: "none"` when a fix is genuinely infeasible
    (literally = unresolved and routed back to the coding agent via the graph).
    """
    conflict_index: int
    action: str               # "fix" when a concrete fix is provided, else "none"
    rationale: str
    fix: Optional[FixSchema] = None

class IntegrationRepairSchema(BaseModel):
    repairs: List[RepairSchema] = Field(default_factory=list)