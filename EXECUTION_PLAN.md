# AI Software Engineering House — Execution Plan

## Pipeline with New Edges

```
START → Requirements → (HITL) → Architecture → Planner
  → {Database, Backend, Frontend}                 (parallel)
  → Testing ────────────────► {Database, Backend, Frontend}     [mod 2: failed tests loop]
  → Security
  → Reviewer ─► HITL-Approval ──► END
      │                │
      └ (REWORK) ─→ coding agents                                 [mod 2 + mod 3]
```

Two graph modifications:
1. **Testing → coding agents** edge: failed tests are sent back to the relevant engineering agent, capped by rework rounds.
2. **Reviewer → human approval** edge: when the code passes the tests, the reviewer pauses and **asks the human for approval** — the human can approve (→ END) or request modifications (→ coding agents rework).

> Note: the former Mod 1 (Integration Agent conflict-rework loop) was **removed** along with the IntegrationAgent itself — coding agents now converge directly into the `finalize` node.

---

## Phase 1 — State models (`core/state.py`)

- [ ] Add a `DevOpsPlan` model (Dockerfile, compose, CI, deploy files).
- [ ] Add `testing_rework_rounds`, `review_rework_rounds`, `max_rework_rounds` to `ProjectState`.
- [ ] Add `ApprovalRequest` / `ProtectedAction` models for HITL approval (Phase 5 / mod 3).
- [ ] Add Pydantic schemas under `agents/schemas/` for all new agents (devops, testing, security, reviewer).

## Phase 3 — Missing pipeline agents

- [ ] `agents/devops.py` — `DevOpsAgent`: generate Dockerfile, docker-compose, CI workflow, deploy scripts from `ArchData.deployment_strategy`; write to output and fill `CodeMap.devops` + manifest.
- [ ] `agents/testing.py` — `TestingAgent`: generate test files from the manifest/exports, execute the test command, and populate `TestResults`.
- [ ] `agents/security.py` — `SecurityAgent`: static scan (imports, secrets, path traversal) over `CodeMap`, populate `SecurityScan`.
- [ ] `agents/reviewer.py` — `ReviewerAgent`: compare the final code against the SRS plus test / security status → verdict `PASS` / `REWORK`. On `PASS` with passing tests, **raise an approval request to the human** (approve or request modifications).

## Phase 2 — Graph wiring (`graph/nodes.py`, `graph/graph.py`, `graph/edges.py`)

- [ ] Add `devops_node`, `testing_node`, `security_node`, `reviewer_node`, and `reviewer_approval_node` to `nodes.py`.
- [ ] Add edge `engineering agents → testing` (after finalize).
- [ ] **Mod 1 (formerly Mod 2):** add conditional edge `testing → {coding agents}` on test failure (capped via `testing_rework_rounds`), else `→ security`.
- [ ] Add edge `testing → security` (on pass).
- [ ] Add edge `security → reviewer`.
- [ ] **Mod 3:** `reviewer → reviewer_approval_node`; on human approval `Yes → END`, on `No` / requested changes → `{backend, frontend, database}` rework, capped via `review_rework_rounds`.
- [ ] Add `reviewer_approval_node` handling human approval pause/resume (reuse the `NodeInterrupt` pattern).
- [ ] `main.py`: stream/log all new nodes and print a final summary (test report, security scan, approval verdict, deliverable location).

## Phase 3 — Human-in-the-Loop for protected actions

- [ ] Generalize the HITL into a reusable approval gate for protected actions: git push, docker build, cloud deploy, file deletion, shell execution, DB migrations.
- [ ] Add a `--auto-approve` flag for non-interactive runs.
- [ ] Refactor `human_clarifying_questions_node` + new approval node to share one mechanism.

## Phase 4 — Memory & RAG layer

- [ ] Long-term memory: persistent user preferences, previous projects, templates; archive completed projects.
- [ ] RAG: embeddings + Qdrant vector store + retrieval layer over LTM & cache.
- [ ] Redis caching layer feeding `BaseAgent` prompt assembly.

## Phase 5 — API / Queue / UI

- [ ] FastAPI `app/` exposing pipeline endpoints; stream LangGraph via checkpointer.
- [ ] Celery + Redis async job queue.
- [ ] Human approval UI.

## Phase 6 — Observability & verification

- [ ] Wire LangSmith tracing / logging across all agents and nodes.
- [ ] Add tests for new agents + graph routing (incl. the two new edge mods).
- [ ] Run the full `pytest` suite.

---

## Suggested order

Phase 1 → 2 → 3 (graph wiring incl. the two mods) → 4 → 5 → 6.