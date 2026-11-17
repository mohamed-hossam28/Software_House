"""Architecture agent: the resolved tech stack can never be overridden by the LLM."""

from agents.architecture import TechStackResolver, merge_tech_stack
from core.state import ProjectState, SRSData


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

    merged = merge_tech_stack(TechStackResolver.resolve(state), llm_proposed)

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
