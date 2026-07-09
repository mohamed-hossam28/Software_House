from pydantic import BaseModel, Field
from typing import List, Optional

class FixSchema(BaseModel):
    action: Optional[str] = None
    filepath: Optional[str] = None
    target_content: Optional[str] = None
    replacement_content: Optional[str] = None

class ClassificationSchema(BaseModel):
    conflict_index: int
    classification: str
    rationale: str
    fix: Optional[FixSchema] = None

class IntegrationSchema(BaseModel):
    classifications: List[ClassificationSchema] = Field(default_factory=list)
