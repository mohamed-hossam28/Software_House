# Software House

A multi-agent pipeline that turns a product request into a generated codebase: requirements, architecture, a plan, code from three coding agents working in parallel, then integration checks and repair.

## Language

### Pipeline

**Coding agent**:
One of the three agents (database, backend, frontend) that run in parallel after the Planner, each writing only the files it owns.
_Avoid_: worker, code generator

### Workspace

**Layer root**:
The fixed directory under `output/` that one layer runs from: `output/backend/` or `output/frontend/`. Every import inside a layer is relative to its Layer root, and database code lives under the backend's.
_Avoid_: backend root, app root, project root

### Architecture

**Persistence Contract**:
The architecture's fixed answer to how the generated app stores data: which database, ORM and SQL dialect, where models live, how the schema is created, and the one connection module every file reaches it through. Every run has exactly one.
_Avoid_: DB config, database settings, ORM map

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
