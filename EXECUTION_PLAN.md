# AI Software Engineering House — Execution Plan

## Pipeline with New Edges

```
START → Requirements → (HITL) → Architecture → Planner
  → {Database, Backend, Frontend}                 (parallel)
  → Integration ──────────────► {Database, Backend, Frontend}   [mod 1: conflict rework loop]
  → Testing ────────────────► {Database, Backend, Frontend}     [mod 2: failed tests loop]
  → Security
  → Reviewer ─► HITL-Approval ──► END
      │                │
      └ (REWORK) ─→ coding agents                                 [mod 2 + mod 3]
```

Three graph modifications:
1. **Integration → coding agents** edge: unresolved conflicts are sent back to the relevant engineering agent (backend / frontend / database), capped by rework rounds.
2. **Testing → coding agents** edge: failed tests are sent back to the relevant engineering agent, capped by rework rounds.
3. **Reviewer → human approval** edge: when the code passes the tests, the reviewer pauses and **asks the human for approval** — the human can approve (→ END) or request modifications (→ coding agents rework).

---

## Phase 1 — Integration Agent: automatic conflict resolution

- [x] `agents/integration.py`: add `_resolve_by_word_match()` — deterministic token / casing / prefix-suffix similarity matching against `code_manifest` (paths, exports, function/class names, API paths) to auto-fix mechanical conflicts before invoking the LLM.
- [x] Replace the two-tier `AUTO_FIX` / `HUMAN` classification (`SYSTEM_PROMPT`, `_get_llm_classifications`) with **batched LLM repair**: group unresolved conflicts into batches, and have the LLM return concrete fixes (`replace_text` / `create_file` / `update_manifest`).
- [x] Remove all `HUMAN` deferral logic in `_apply_fixes`; route remaining unresolved conflicts to the batch LLM repair.
- [x] Track integration rework rounds (cap ~2) and re-run Phase 1 deterministic checks after each repair pass.
- [x] Update `IntegrationStatus.conflicts` to reflect only truly-unresolved items (+ add `conflicts_by_agent` / `rework_targets` for the graph REWORK edge).
- [x] Update `agents/schemas/integration.py` schema (drop `HUMAN`, require a concrete `fix` for every conflict).
- [x] Update the standalone smoke-test block (`integration.py`) and `tests/integration.py` to cover word-match, batch-repair, and exhausted-rounds paths.

## Phase 2 — State models (`core/state.py`)

- [x] Add `integration_rework_rounds` / `max_integration_rounds` to `ProjectState` (used by mod-1 edge).
- [ ] Add a `DevOpsPlan` model (Dockerfile, compose, CI, deploy files).
- [ ] Ensure `CodeMap.devops` exists (referenced at `integration.py:738`).
- [ ] Add `testing_rework_rounds`, `review_rework_rounds`, `max_rework_rounds` to `ProjectState`.
- [ ] Add `ApprovalRequest` / `ProtectedAction` models for HITL approval (Phase 5 / mod 3).
- [ ] Add Pydantic schemas under `agents/schemas/` for all new agents (devops, testing, security, reviewer).

## Phase 3 — Missing pipeline agents

- [ ] `agents/devops.py` — `DevOpsAgent`: generate Dockerfile, docker-compose, CI workflow, deploy scripts from `ArchData.deployment_strategy`; write to output and fill `CodeMap.devops` + manifest.
- [ ] `agents/testing.py` — `TestingAgent`: generate test files from the manifest/exports, execute the test command, and populate `TestResults`.
- [ ] `agents/security.py` — `SecurityAgent`: static scan (imports, secrets, path traversal) over `CodeMap`, populate `SecurityScan`.
- [ ] `agents/reviewer.py` — `ReviewerAgent`: compare the final code against the SRS plus test / security / integration status → verdict `PASS` / `REWORK`. On `PASS` with passing tests, **raise an approval request to the human** (approve or request modifications).

## Phase 4 — Graph wiring (`graph/nodes.py`, `graph/graph.py`, `graph/edges.py`)

- [ ] Add `devops_node`, `testing_node`, `security_node`, `reviewer_node`, and `reviewer_approval_node` to `nodes.py`.
- [x] **Mod 1:** after `integration` add a conditional edge → `{backend, frontend, database}` on unresolved conflicts (REWORK), else `END`. Cap via `integration_rework_rounds` / `max_integration_rounds` (`graph/edges.py: route_after_integration`).
- [ ] Add edge `integration → testing` (on clean).
- [ ] **Mod 2:** add conditional edge `testing → {coding agents}` on test failure (capped via `testing_rework_rounds`), else `→ security`.
- [ ] Add edge `testing → security` (on pass).
- [ ] Add edge `security → reviewer`.
- [ ] **Mod 3:** `reviewer → reviewer_approval_node`; on human approval `Yes → END`, on `No` / requested changes → `{backend, frontend, database}` rework, capped via `review_rework_rounds`.
- [ ] Add `reviewer_approval_node` handling human approval pause/resume (reuse the `NodeInterrupt` pattern).
- [ ] `main.py`: stream/log all new nodes and print a final summary (test report, security scan, approval verdict, deliverable location).

## Phase 5 — Human-in-the-Loop for protected actions

- [ ] Generalize the HITL into a reusable approval gate for protected actions: git push, docker build, cloud deploy, file deletion, shell execution, DB migrations.
- [ ] Add a `--auto-approve` flag for non-interactive runs.
- [ ] Refactor `human_clarifying_questions_node` + new approval node to share one mechanism.

## Phase 6 — Memory & RAG layer

- [ ] Long-term memory: persistent user preferences, previous projects, templates; archive completed projects.
- [ ] RAG: embeddings + Qdrant vector store + retrieval layer over LTM & cache.
- [ ] Redis caching layer feeding `BaseAgent` prompt assembly.

## Phase 7 — API / Queue / UI

- [ ] FastAPI `app/` exposing pipeline endpoints; stream LangGraph via checkpointer.
- [ ] Celery + Redis async job queue.
- [ ] Human approval UI.

## Phase 8 — Observability & verification

- [ ] Wire LangSmith tracing / logging across all agents and nodes.
- [ ] Add tests for new agents + graph routing (incl. the three new edge mods).
- [ ] Run the full `pytest` suite.

---

## Suggested order

Phase 1 → 2 → 3 → 4 (graph wiring incl. the three mods) → 5 → 6 → 7 → 8.