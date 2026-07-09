from pydantic import BaseModel, Field
from typing import List, Optional, Any

class ColumnSchema(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None

class DBManifestSchema(BaseModel):
    table_name: str
    columns: List[ColumnSchema] = Field(default_factory=list)
    indexes: List[str] = Field(default_factory=list)
    exports: List[str] = Field(default_factory=list)
    suggested_additions: List[str] = Field(default_factory=list)

class ArgumentSchema(BaseModel):
    name: str
    type: str
    default: Optional[str] = ""

class DBFunctionSchema(BaseModel):
    name: str
    arguments: List[ArgumentSchema] = Field(default_factory=list)
    return_type: str = "None"
    docstring: str = ""

class DatabaseSchema(BaseModel):
    migration_code: str
    manifest: Optional[DBManifestSchema] = None
    functions: List[DBFunctionSchema] = Field(default_factory=list)
