from pydantic import BaseModel, Field
from typing import List, Optional

class ArgumentSchema(BaseModel):
    name: str
    type: str
    default: Optional[str] = ""

class ClassSchema(BaseModel):
    name: str
    arguments: List[ArgumentSchema] = Field(default_factory=list)
    docstring: str = ""

class FunctionSchema(BaseModel):
    name: str
    arguments: List[ArgumentSchema] = Field(default_factory=list)
    return_type: str = "None"
    docstring: str = ""

class FrontendSchema(BaseModel):
    code: str
    classes: List[ClassSchema] = Field(default_factory=list)
    functions: List[FunctionSchema] = Field(default_factory=list)
