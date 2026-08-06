from pydantic import BaseModel, Field
from typing import List, Dict

class TaskSchema(BaseModel):
    id: str
    agent: str = ""
    title: str = ""
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)
    priority: int = 1
    status: str = "pending"
    file: str = ""
    exports: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)


class SkeletonNodeSchema(BaseModel):
    """A single file in the full-project skeleton (Pass 1 of two-pass planning)."""
    id: str
    agent: str = ""
    title: str = ""
    description: str = ""
    file: str = ""
    exports: List[str] = Field(default_factory=list)
    imports_from: Dict[str, List[str]] = Field(default_factory=dict)  # source file -> imported names


class SkeletonSchema(BaseModel):
    """Pass-1 output: the complete, globally-consistent project file tree."""
    skeleton: List[SkeletonNodeSchema] = Field(default_factory=list)
