from pydantic import BaseModel, Field
from typing import List

class TaskSchema(BaseModel):
    id: str
    agent: str
    title: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    priority: int = 1
    status: str = "pending"
    file: str = ""
    exports: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
