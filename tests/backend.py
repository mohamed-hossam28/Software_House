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

from core.state import ProjectState, Task, TaskList, FileManifest, CodeManifest, ArchData, APIContract
from agents.backend import BackendAgent

def test_backend_agent():
    print("==================================================")
    print("STARTING TEST FOR BackendAgent")
    print("==================================================")
    t0 = time.perf_counter()
    
    print("\n--- STEP 1: Initializing ProjectState with Backend Task ---")
    state = ProjectState(project_id="test_be", project_name="Test Backend")
    
    state.architecture = ArchData(
        tech_stack={
            "database": "PostgreSQL 16",
            "backend": "FastAPI (Python 3.12)"
        },
        api_contracts=[
            APIContract(
                method="POST",
                path="/api/v1/auth/login",
                auth_required=False,
                description="Authenticate user and return access token.",
                request_body={"email": "str", "password": "str"},
                response_body={"access_token": "str", "token_type": "str"}
            )
        ]
    )
    
    task = Task(
        id="be-001",
        agent="backend",
        title="Implement Auth router login",
        description="Implement post endpoint for login at /api/v1/auth/login validating credentials.",
        depends_on=[],
        priority=2,
        status="pending"
    )
    state.tasks = TaskList(tasks=[task])
    
    fm = FileManifest(
        agent="backend",
        task_id="be-001",
        filepath="backend/app/routes/auth.py",
        purpose="Auth routes handling login and authentication",
        exports=["login_router"]
    )
    state.code_manifest = CodeManifest(files={fm.filepath: fm})
    
    print("\n--- STEP 2: Initializing BackendAgent ---")
    agent = BackendAgent(
        provider=settings.default_provider,
        model=settings.model_name
    )
    print(f"Agent name: {agent.name}")
    print(f"Agent model: {agent.model}")
    print(f"Agent provider: {agent.provider.provider_name if hasattr(agent, 'provider') and agent.provider else 'None'}")
    
    print("\n--- STEP 3: Running BackendAgent ---")
    result_state = agent.run(state)
    
    print("\n--- STEP 4: Execution finished. Printing final state ---")
    print(json.dumps(result_state.to_dict(), indent=2))
    
    # Clean up written file mock from disk if created
    root_dir = "output"
    target_path = os.path.join(root_dir, "backend/app/routes/auth.py")
    
    duration = time.perf_counter() - t0
    print(f"duration = {duration}")
    
    print("\n==================================================")
    print("BackendAgent Test Complete")
    print("==================================================")

if __name__ == "__main__":
    test_backend_agent()
