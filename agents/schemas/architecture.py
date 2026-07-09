from pydantic import BaseModel, Field
from typing import List, Dict, Any

class APIContractSchema(BaseModel):
    method: str
    path: str
    auth_required: bool
    description: str
    request_body: Dict[str, Any] = Field(default_factory=dict)
    response_body: Dict[str, Any] = Field(default_factory=dict)

class ArchitectureSchema(BaseModel):
    tech_stack: Dict[str, str] = Field(default_factory=dict)
    system_components: List[str] = Field(default_factory=list)
    api_contracts: List[APIContractSchema] = Field(default_factory=list)
    database_schema_overview: str = ""
    data_flow: str = ""
    deployment_strategy: str = ""
    security_requirements: List[str] = Field(default_factory=list)
