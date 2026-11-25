"""Ticket #25: the fan-in after the coding agents keeps every File Manifest Written.

The database, backend and frontend coding agents run in parallel from the same
Planner-era state. Their changes are combined by the project state's reducers,
in whatever order they finish. These tests assert only on the merged state.
"""
import itertools
import os
import sys
import typing

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.state import (
    CodeManifest,
    FileManifest,
    ProjectState,
    Task,
    TaskList,
    diff_state,
)

CODING_AGENTS = ("database", "backend", "frontend")

#: File Manifests the Planner assigns, grouped by owning coding agent.
PLANNED_FILES = {
    "database": ["output/app/models/todo.py", "output/app/db/session.py"],
    "backend": ["output/app/api/todo.py", "output/app/main.py"],
    "frontend": ["output/frontend/index.html", "output/frontend/app.js"],
}


def planner_era_state() -> ProjectState:
    """A state as the Planner leaves it: every File Manifest planned, none Written."""
    state = ProjectState(project_id="fan-in", project_name="FanIn")
    tasks: list[Task] = []
    for agent, paths in PLANNED_FILES.items():
        for i, path in enumerate(paths):
            task_id = f"{agent[:2]}-{i:03d}"
            tasks.append(Task(id=task_id, agent=agent, title=path, description=path))
            state.code_manifest.files[path] = FileManifest(
                agent=agent, task_id=task_id, filepath=path, purpose=f"{agent} file"
            )
    state.tasks = TaskList(tasks=tasks)
    return state


def code_for(path: str) -> str:
    return f"# generated {path}\n"


def coding_agent_writes(agent: str, state: ProjectState) -> ProjectState:
    """Simulate one coding agent: write its own File Manifests, log an error, add metrics."""
    state = state.model_copy(deep=True)
    state.mark_agent_start(agent)
    for fm in state.code_manifest.by_agent(agent):
        getattr(state.generated_code, agent)[fm.filepath] = code_for(fm.filepath)
        state.code_manifest.mark_written(fm.filepath)
    state.log_error(agent, f"{agent} warning")
    state.metrics.total_tokens_used += 100
    state.metrics.total_llm_calls += 2
    state.metrics.agent_durations[agent] = 1.5
    return state


def reducer_for(key: str):
    """The reducer LangGraph uses for *key*: the one annotated on ProjectState, if any."""
    hints = typing.get_type_hints(ProjectState, include_extras=True)
    for meta in getattr(hints[key], "__metadata__", ()):
        if callable(meta):
            return meta
    return None


def fold(state: dict, changes: dict) -> dict:
    """Apply one node's returned changes the way LangGraph does."""
    merged = dict(state)
    for key, value in changes.items():
        reducer = reducer_for(key)
        merged[key] = reducer(merged[key], value) if reducer else value
    return merged


def coding_agent_changes() -> tuple[ProjectState, dict[str, dict]]:
    start = planner_era_state()
    return start, {
        agent: diff_state(start, coding_agent_writes(agent, start))
        for agent in CODING_AGENTS
    }


def assert_every_file_written_and_present(final: ProjectState) -> None:
    for agent, paths in PLANNED_FILES.items():
        for path in paths:
            assert final.code_manifest.files[path].written, f"{path} lost Written"
            assert getattr(final.generated_code, agent)[path] == code_for(path)


def test_write_refusals_from_parallel_agents_merge_and_clear_once_written():
    start = planner_era_state()
    db_path, fe_path = PLANNED_FILES["database"][0], PLANNED_FILES["frontend"][0]
    db = start.model_copy(deep=True)
    db.record_refusal(db_path, "truncated")
    fe = start.model_copy(deep=True)
    fe.record_refusal(fe_path, "unbalanced")

    state = start.model_dump(mode="python")
    for after in (db, fe):
        state = fold(state, diff_state(start, after))
    assert state["write_refusals"] == {db_path: "truncated", fe_path: "unbalanced"}

    refused = ProjectState(**state)
    healed = refused.model_copy(deep=True)
    healed.record_written("database", db_path, code_for(db_path))
    state = fold(state, diff_state(refused, healed))
    assert state["write_refusals"] == {fe_path: "unbalanced"}


@pytest.mark.parametrize("order", list(itertools.permutations(CODING_AGENTS)))
def test_folding_coding_agent_changes_in_any_order_keeps_every_file_written(order):
    start, changes = coding_agent_changes()

    merged = start.model_dump(mode="json")
    for agent in order:
        merged = fold(merged, changes[agent])
    final = ProjectState.model_validate(merged)

    assert_every_file_written_and_present(final)
    assert sorted(e["message"] for e in final.errors) == sorted(
        f"{agent} warning" for agent in CODING_AGENTS
    )
    assert final.metrics.total_tokens_used == 300
    assert final.metrics.total_llm_calls == 6
    assert final.metrics.agent_durations == {agent: 1.5 for agent in CODING_AGENTS}


def test_every_fold_order_gives_the_same_state():
    start, changes = coding_agent_changes()

    def final_state(order) -> dict:
        merged = start.model_dump(mode="json")
        for agent in order:
            merged = fold(merged, changes[agent])
        final = ProjectState.model_validate(merged).model_dump(mode="json")
        # Errors keep arrival order and current_agent is latest-wins by design.
        final["errors"] = sorted(final["errors"], key=lambda e: e["message"])
        final.pop("current_agent")
        return final

    states = [final_state(order) for order in itertools.permutations(CODING_AGENTS)]
    assert all(s == states[0] for s in states[1:])


def test_planner_assigns_every_file_manifest_to_exactly_one_coding_agent():
    """One owner per file: each File Manifest belongs to one coding agent, the
    same one that owns its task, so a coding agent working through its own
    tasks only ever changes its own File Manifests. The fan-in relies on this."""
    from agents.planner import PlannerAgent
    from core.state import ArchData, build_persistence_contract
    from tests.fixtures.scripted_llm import scripted_planner

    state = ProjectState(project_id="owners", project_name="Owners")
    state.requirements.raw_prompt = "Build a todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "frontend": "HTML + Vanilla JS",
                    "database": "SQLite 3"},
        persistence_contract=build_persistence_contract("SQLite 3"),
    )
    plan = ProjectState.model_validate(
        fold(state.model_dump(mode="json"), PlannerAgent(provider=scripted_planner()).run(state.model_copy(deep=True)))
    )

    task_owner = {t.id: t.agent for t in plan.tasks.tasks}
    assert plan.code_manifest.files
    for path, fm in plan.code_manifest.files.items():
        assert fm.agent in CODING_AGENTS, f"{path} has no coding-agent owner: {fm.agent!r}"
        assert task_owner[fm.task_id] == fm.agent, f"{path} owner differs from its task's owner"
    for agent in CODING_AGENTS:
        owned = {fm.filepath for fm in plan.code_manifest.by_agent(agent)}
        others = {fm.filepath for other in CODING_AGENTS if other != agent
                  for fm in plan.code_manifest.by_agent(other)}
        assert not owned & others


def test_langgraph_fan_in_keeps_every_file_manifest_written():
    """A real LangGraph fan-out/fan-in on ProjectState uses the annotated reducers."""
    from langgraph.graph import END, START, StateGraph

    def as_state(state) -> ProjectState:
        return state if isinstance(state, ProjectState) else ProjectState.model_validate(state)

    def planner(state):
        before = as_state(state)
        return diff_state(before, planner_era_state())

    def coding_node(agent: str):
        def node(state):
            before = as_state(state)
            return diff_state(before, coding_agent_writes(agent, before))
        return node

    def final(state):
        return {}

    graph = StateGraph(ProjectState)
    graph.add_node("planner", planner)
    for agent in CODING_AGENTS:
        graph.add_node(agent, coding_node(agent))
        graph.add_edge("planner", agent)
        graph.add_edge(agent, "final")
    graph.add_node("final", final)
    graph.add_edge(START, "planner")
    graph.add_edge("final", END)

    result = graph.compile().invoke(ProjectState())
    final_state = as_state(result)

    assert_every_file_written_and_present(final_state)
    assert final_state.metrics.total_tokens_used == 300
    assert len(final_state.errors) == 3
