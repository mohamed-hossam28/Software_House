from pydantic import BaseModel, Field
from typing import List, Dict, Any


# ── Round 1 schemas ───────────────────────────────────────────────────────────
class TechStackSchema(BaseModel):
    tech_stack: Dict[str, str] = Field(default_factory=dict)

class SystemComponentsSchema(BaseModel):
    system_components: List[str] = Field(default_factory=list)

class DbSchemaOverviewSchema(BaseModel):
    database_schema_overview: str = ""

# ── Round 2 schemas ───────────────────────────────────────────────────────────
class APIContractSchema(BaseModel):
    method: str
    path: str
    auth_required: bool
    description: str
    request_body: Dict[str, Any] = Field(default_factory=dict)
    response_body: Dict[str, Any] = Field(default_factory=dict)

class APIContractsSchema(BaseModel):
    api_contracts: List[APIContractSchema] = Field(default_factory=list)

class SecurityRequirementsSchema(BaseModel):
    security_requirements: List[str] = Field(default_factory=list)

# ── Full schema (kept for backward compat / smoke tests) ─────────────────────
class ArchitectureSchema(BaseModel):
    tech_stack: Dict[str, str] = Field(default_factory=dict)
    system_components: List[str] = Field(default_factory=list)
    api_contracts: List[APIContractSchema] = Field(default_factory=list)
    database_schema_overview: str = ""
    deployment_strategy: str = ""
    security_requirements: List[str] = Field(default_factory=list)
