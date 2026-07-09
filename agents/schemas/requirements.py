from pydantic import BaseModel, Field
from typing import List

class RequirementsSchema(BaseModel):
    functional_requirements: List[str] = Field(default_factory=list)
    non_functional_requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    out_of_scope: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)
