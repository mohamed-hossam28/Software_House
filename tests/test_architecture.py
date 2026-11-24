"""Architecture agent: the resolved tech stack can never be overridden by the LLM."""

import json

import pytest
from pydantic import ValidationError

from agents.architecture import ArchitectureAgent, TechStackResolver, merge_tech_stack
from core.state import (
    ArchData,
    PersistenceContract,
    ProjectState,
    SRSData,
    build_persistence_contract,
    persistence_mandate,
)
from providers.llm.base import BaseLLMProvider, LLMResponse


def test_resolver_value_beats_llm_proposal_for_the_same_layer():
    resolved = {"database": "SQLite 3"}
    llm_proposed = {"database": "PostgreSQL 16"}

    assert merge_tech_stack(resolved, llm_proposed) == {"database": "SQLite 3"}


def test_llm_fills_only_layers_the_resolver_left_unspecified():
    resolved = {"backend": "FastAPI (Python 3.12)", "database": "SQLite 3"}
    llm_proposed = {
        "backend": "Django (Python 3.12)",
        "database": "PostgreSQL 16",
        "message_queue": "RabbitMQ 3",
    }

    assert merge_tech_stack(resolved, llm_proposed) == {
        "backend": "FastAPI (Python 3.12)",
        "database": "SQLite 3",
        "message_queue": "RabbitMQ 3",
    }


def test_resolved_stack_from_srs_and_defaults_beats_llm_proposal():
    state = ProjectState(project_id="merge-test", project_name="Merge Test")
    state.requirements = SRSData(raw_prompt="A todo app that stores tasks in SQLite.")
    llm_proposed = {
        "database": "PostgreSQL 16",
        "frontend": "Vue 3 + TypeScript",
        "message_queue": "RabbitMQ 3",
    }

    merged = merge_tech_stack(TechStackResolver.resolve(state).stack, llm_proposed)

    assert merged["database"] == "SQLite 3"                  # SRS-detected beats LLM
    assert merged["frontend"] == "React 18 + TypeScript"     # default beats LLM
    assert merged["message_queue"] == "RabbitMQ 3"           # LLM fills an unspecified layer


def test_llm_layer_differing_only_in_case_or_whitespace_cannot_shadow_a_resolved_layer():
    resolved = {"database": "SQLite 3"}
    llm_proposed = {"Database": "PostgreSQL 16", " DATABASE ": "MySQL 8"}

    assert merge_tech_stack(resolved, llm_proposed) == {"database": "SQLite 3"}


def test_llm_layers_with_blank_values_are_dropped():
    resolved = {"database": "SQLite 3"}
    llm_proposed = {"message_queue": "", "search": "   ", "storage": "MinIO"}

    assert merge_tech_stack(resolved, llm_proposed) == {
        "database": "SQLite 3",
        "storage": "MinIO",
    }


def _resolve(raw_prompt: str):
    state = ProjectState(project_id="resolve-test", project_name="Resolve Test")
    state.requirements = SRSData(raw_prompt=raw_prompt)
    return TechStackResolver.resolve(state)


def test_unsupported_database_is_replaced_by_postgresql_and_the_substitution_recorded():
    resolved = _resolve("A todo app that stores tasks in DynamoDB.")

    assert resolved.stack["database"] == "PostgreSQL 16"
    assert resolved.assumptions == ["AWS DynamoDB is not supported; using PostgreSQL 16"]


def test_supported_database_is_kept_with_no_assumption():
    resolved = _resolve("A todo app that stores tasks in SQLite.")

    assert resolved.stack["database"] == "SQLite 3"
    assert resolved.assumptions == []


class RecordingArchitectureLLM(BaseLLMProvider):
    """Answers every architecture sub-call with one fixed JSON body; records the prompts.

    The body proposes an unsupported database, which the resolver must still win over.
    """

    BODY = {
        "tech_stack": {"database": "AWS DynamoDB"},
        "system_components": ["TodoService"],
        "security_requirements": [],
        "database_schema_overview": "todo_items",
        "api_contracts": [],
    }

    def __init__(self):
        self.user_messages: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-architecture"

    def complete(self, system_prompt, user_message, model, max_tokens=8192, temperature=0.2, **kwargs):
        self.user_messages.append(user_message)
        return LLMResponse(text=json.dumps(self.BODY), model="mock", provider=self.provider_name)


def _run_architecture(raw_prompt: str) -> tuple[ArchData, list[str]]:
    state = ProjectState(project_id="arch-run", project_name="Arch Run")
    state.requirements = SRSData(raw_prompt=raw_prompt, functional_requirements=["Manage todos"])
    llm = RecordingArchitectureLLM()
    diff = ArchitectureAgent(provider=llm).run(state)
    return ArchData(**diff["architecture"]), llm.user_messages


def test_architecture_step_with_unsupported_database_produces_a_postgresql_contract():
    arch, _ = _run_architecture("A todo app that stores tasks in DynamoDB.")

    assert arch.tech_stack["database"] == "PostgreSQL 16"
    assert arch.persistence_contract == build_persistence_contract("PostgreSQL 16")
    assert "AWS DynamoDB is not supported; using PostgreSQL 16" in arch.assumptions


def test_architecture_prompt_always_carries_the_persistence_mandate():
    _, prompts = _run_architecture("A very simple todo app.")

    core_prompt = next(p for p in prompts if "## Complexity: MINIMAL" in p)
    assert persistence_mandate("MINIMAL") in core_prompt


def test_architecture_data_cannot_lack_a_persistence_contract():
    assert isinstance(ArchData().persistence_contract, PersistenceContract)
    with pytest.raises(ValidationError):
        ArchData(persistence_contract=None)
