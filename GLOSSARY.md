# Software House

A multi-agent pipeline that turns a product request into a generated codebase: requirements, architecture, a plan, code from three coding agents working in parallel, then integration checks and repair.

## Language

### Pipeline

**Coding agent**:
One of the three agents (database, backend, frontend) that run in parallel after the Planner, each writing only the files it owns through the coding loop. The Docs agent is not a Coding agent: it borrows the backend Coding agent's single-file path for one file.
_Avoid_: worker, code generator

**Docs agent**:
The agent that runs after Integration and writes the project README from the final, repaired files, so the README documents the code that ships. It owns no planned file: the Planner never plans a README.
_Avoid_: README agent, documentation generator

**Write refusal**:
The safe-write guard's refusal to write a file whose content looks corrupted, recorded with its reason until the file is written. A retry of that file carries the reason so it avoids the same mistake.
_Avoid_: rejected write, write failure, corrupt write

### Workspace

**Layer root**:
The fixed directory under `output/` that one layer runs from: `output/backend/` or `output/frontend/`. Every import inside a layer is relative to its Layer root, and database code lives under the backend's.
_Avoid_: backend root, app root, project root

### Architecture

**Persistence Contract**:
The architecture's fixed answer to how the generated app stores data: which database, ORM and SQL dialect, where models live, how the schema is created, and the one connection module every file reaches it through. Every run has exactly one.
_Avoid_: DB config, database settings, ORM map

**Entity Contract**:
The architecture's field-level data model: each persisted entity's name, table and fields, every field typed from a closed vocabulary that maps to its ORM column, schema and route-parameter types. The database and backend Coding agents both code against it, so a field's type is decided once. Unlike the Persistence Contract, which fixes how data is stored, the Entity Contract fixes what is stored.
_Avoid_: schema, data model, entity list

### Plan

**Code Manifest**:
The Planner's set of File Manifests: every file the run will produce.
_Avoid_: manifest (on its own), file plan

**File Manifest**:
The Planner's record of one file to produce: its path, owning coding agent, purpose, exports, and whether it is Written. Each file has exactly one owning coding agent.
_Avoid_: manifest entry, file entry, file contract

**Planned Content**:
A File Manifest's content fixed by the Planner, written verbatim without the LLM.
_Avoid_: deterministic content, template, stub

**Written**:
A File Manifest is Written once its owning coding agent has produced its content. It is the only signal that a planned file is done.
_Avoid_: completed, done, in progress
