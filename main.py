import os
import sys
import logging
from threading import Lock
from typing import TextIO
from dotenv import load_dotenv
load_dotenv()

# Force workspace root on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logger = logging.getLogger("MainOrchestrator")
_logging_configured = False
_configured_log_path = ""
_tee_lock = Lock()


class _TeeStream:
    """Mirror terminal output into the pipeline log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        with _tee_lock:
            self.terminal.write(text)
            self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        with _tee_lock:
            self.terminal.flush()
            if not self.log_file.closed:
                self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def configure_run_logging(log_path: str | None = None) -> str:
    """Tee pipeline output to the terminal and `output/logs.txt` once per process."""
    global _logging_configured, _configured_log_path
    if _logging_configured:
        return _configured_log_path

    path = log_path or os.path.join(ROOT, "output", "logs.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_file = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Keep generation noise down while retaining repair-round evidence.
    logging.getLogger("agents").setLevel(logging.WARNING)
    logging.getLogger("agents.integration").setLevel(logging.INFO)
    logging.getLogger("graph").setLevel(logging.WARNING)
    _logging_configured = True
    _configured_log_path = path
    return path


def run_pipeline(prompt: str, project_name: str = "MySoftwareProject", max_rounds: int = 3):
    configure_run_logging()
    # Import project modules only after the tee is active so dependency/provider
    # warnings emitted during import are preserved in output/logs.txt.
    from agents.utils import format_static_checks_summary
    from core.config import Settings
    from core.state import ProjectState
    from graph.graph import graph

    settings = Settings()
    print("\n" + "=" * 60)
    print("STARTING MULTI-AGENT SOFTWARE BUILD PIPELINE (STREAMING)")
    print("=" * 60)
    print(f"Project Name      : {project_name}")
    print(f"Initial Prompt    : {prompt}")
    print(f"Max HITL Rounds   : {max_rounds}")
    print(f"Model Name        : {settings.model_name}")
    print("=" * 60 + "\n")

    # 1. Initialize project state
    state = ProjectState(
        project_id=f"proj_{int(os.urandom(4).hex(), 16)}",
        project_name=project_name,
        max_clarfication_rounds=max_rounds,
        clarfication_rounds=0
    )
    state.requirements.raw_prompt = prompt

    # 2. Config with a unique thread ID to enable MemorySaver checkpointing
    config = {
        "configurable": {
            "thread_id": state.project_id
        }
    }

    current_state = state
    try:
        # Stream the graph step-by-step
        for event in graph.stream(current_state, config=config):
            for node_name, _ in event.items():
                print(f"\n>>> [Node Executed] '{node_name}'")
                
                # Fetch full state snapshot from thread checkpointer memory
                snapshot = graph.get_state(config)
                current_state = snapshot.values
                if isinstance(current_state, dict):
                    current_state = ProjectState(**current_state)

                # Stream specific outputs from the node results
                if node_name == "requirements":
                    reqs = current_state.requirements
                    print(f"  |- Functional Reqs: {len(reqs.functional_requirements)}")
                    print(f"  |- Non-Functional:  {len(reqs.non_functional_requirements)}")
                    if reqs.clarifying_questions:
                        resolved_str = "RESOLVED" if (current_state.hitl and current_state.hitl.is_resolved) else "PENDING"
                        print(f"  +- Clarification:   {len(reqs.clarifying_questions)} questions ({resolved_str})")
                
                elif node_name == "human":
                    print(f"  +- Clarification Round Completed. Total Rounds: {current_state.clarfication_rounds}")

                elif node_name == "architecture":
                    arch = current_state.architecture
                    print(f"  |- System Components: {len(arch.system_components)}")
                    print(f"  |- API Contracts:     {len(arch.api_contracts)} endpoints")
                    print(f"  +- Database Schema:   {arch.database_schema_overview[:80]}...")

                elif node_name == "planner":
                    tasks = current_state.tasks.tasks
                    manifest = current_state.code_manifest.files
                    print(f"  |- Tasks Created:     {len(tasks)}")
                    print(f"  +- Files Planned:     {len(manifest)}")

                elif node_name in ["database", "backend", "frontend"]:
                    written_files = [
                        path for path, manifest in current_state.code_manifest.files.items()
                        if manifest.written and manifest.agent == node_name
                    ]
                    file_names = [os.path.basename(f) for f in written_files]
                    print(f"  +- Written files ({node_name}): {len(written_files)} files ({', '.join(file_names)})")

    except Exception as e:
        logger.error(f"Pipeline execution encountered an error: {e}", exc_info=True)
        # Attempt to retrieve current state from checkpointer
        try:
            snapshot = graph.get_state(config)
            current_state = snapshot.values
            if isinstance(current_state, dict):
                current_state = ProjectState(**current_state)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Final Status        : {current_state.status}")
    print(f"Clarification Rounds: {current_state.clarfication_rounds}/{current_state.max_clarfication_rounds}")
    print(f"Model Name          : {settings.model_name}")
    print(f"Functional Reqs Cnt : {len(current_state.requirements.functional_requirements)}")
    print(f"System Components   : {len(current_state.architecture.system_components)}")
    print(f"Tasks Generated     : {len(current_state.tasks.tasks)}")
    print(f"Total LLM Calls     : {current_state.metrics.total_llm_calls}")
    print(f"Total Tokens Used   : {current_state.metrics.total_tokens_used}")
    print(f"Errors Logged       : {len(current_state.errors)}")
    print(format_static_checks_summary(current_state))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Software House Multi-Agent Pipeline")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Build a very simple todo app with a FastAPI backend and html and css frontend that does not use any frameworks for the frontend",
        help="Natural language prompt for the software project"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="TodoApp",
        help="Name of the project"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max clarification rounds before bypassing human interaction"
    )
    args = parser.parse_args()

    run_pipeline(args.prompt, args.name, args.max_rounds)
