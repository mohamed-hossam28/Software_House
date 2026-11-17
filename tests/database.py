import os
import sys
import json
import time
from dotenv import load_dotenv

workspace_dir = r"c:\Users\bebawy1\Desktop\Software_House"
sys.path.insert(0, workspace_dir)

# Load env file explicitly using absolute path before importing config
load_dotenv(os.path.join(workspace_dir, ".env"))

# Import settings to load env variables
from core.config import settings

from core.state import ProjectState, Task, TaskList, FileManifest, CodeManifest, ArchData
from agents.data_base import DatabaseAgent

def test_database_agent():
    print("==================================================")
    print("STARTING TEST FOR DBAgent")
    print("==================================================")
    t0 = time.perf_counter()
    
    print("\n--- STEP 1: Initializing ProjectState with Database Task ---")
    state = ProjectState(project_id="test_db", project_name="Test Database")
    
    state.architecture = ArchData(
        tech_stack={
            "database": "PostgreSQL 16",
            "backend": "FastAPI (Python 3.12)"
        },
        database_schema_overview="Users table: id (UUID, PK), email (VARCHAR, unique), password_hash (VARCHAR), created_at (TIMESTAMP)."
    )
    
    task = Task(
        id="db-001",
        agent="database",
        title="Create users table",
        description="Write migration script to create the users table with UUID primary key and index on email.",
        depends_on=[],
        priority=1,
        status="pending"
    )
    state.tasks = TaskList(tasks=[task])
    
    fm = FileManifest(
        agent="database",
        task_id="db-001",
        filepath="backend/migrations/001_create_users_table.sql",
        purpose="Create users table",
        exports=["users"]
    )
    state.code_manifest = CodeManifest(files={fm.filepath: fm})
    
    print("\n--- STEP 2: Initializing DBAgent ---")
    agent = DatabaseAgent(
        provider=settings.default_provider,
        model=settings.model_name
    )
    print(f"Agent name: {agent.name}")
    print(f"Agent model: {agent.model}")
    print(f"Agent provider: {agent.provider.provider_name if hasattr(agent, 'provider') and agent.provider else 'None'}")
    
    print("\n--- STEP 3: Running DBAgent ---")
    result_state = agent.run(state)
    
    print("\n--- STEP 4: Execution finished. Printing final state ---")
    print(json.dumps(result_state.to_dict(), indent=2))
    
    duration = time.perf_counter() - t0
    print(f"duration = {duration}")
    
    print("\n==================================================")
    print("DBAgent Test Complete")
    print("==================================================")

if __name__ == "__main__":
    test_database_agent()
