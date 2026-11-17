import os
import sys
import json
import time
from dotenv import load_dotenv

workspace_dir = r"c:\Users\bebawy1\Desktop\Software_House"
sys.path.insert(0, workspace_dir)

# Load env file explicitly using absolute path before importing config
load_dotenv(os.path.join(workspace_dir, ".env"))

from core.config import settings
from core.state import ProjectState, Task, TaskList, FileManifest, CodeManifest, ArchData, APIContract
from agents.frontend import FrontendAgent


def test_frontend_agent_prompts():
    print("==================================================")
    print("STARTING FRONTEND AGENT PROMPT INSPECTION TEST")
    print("==================================================")
    t0 = time.perf_counter()
    
    state = ProjectState(project_id="test_fe_prompt", project_name="Frontend Prompt Inspection")
    
    state.architecture = ArchData(
        tech_stack={
            "frontend": "React 18 + TypeScript",
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
            ),
            APIContract(
                method="GET",
                path="/api/v1/users/me",
                auth_required=True,
                description="Get current authenticated user profile.",
                request_body={},
                response_body={"id": "str", "email": "str", "name": "str"}
            ),
            APIContract(
                method="POST",
                path="/api/v1/todos",
                auth_required=True,
                description="Create a new todo item.",
                request_body={"title": "str", "priority": "str"},
                response_body={"id": "str", "title": "str", "status": "str"}
            )
        ]
    )
    
    task1 = Task(
        id="fe-001",
        agent="frontend",
        title="Create API Client",
        description="Create TypeScript API client module for login authentication and fetching user profile.",
        depends_on=[],
        priority=1,
        status="pending"
    )
    task2 = Task(
        id="fe-002",
        agent="frontend",
        title="Create Login Page component",
        description="Create a React login page component that imports loginUser from API client.",
        depends_on=["fe-001"],
        priority=2,
        status="pending"
    )
    state.tasks = TaskList(tasks=[task1, task2])
    
    fm1 = FileManifest(
        agent="frontend",
        task_id="fe-001",
        filepath="frontend/src/api/client.ts",
        purpose="API client wrapper for auth and user endpoints",
        exports=["loginUser", "getCurrentUser"]
    )
    fm2 = FileManifest(
        agent="frontend",
        task_id="fe-002",
        filepath="frontend/src/components/LoginPage.tsx",
        purpose="Login page user interface component",
        exports=["LoginPage"]
    )
    state.code_manifest = CodeManifest(files={fm1.filepath: fm1, fm2.filepath: fm2})
    
    agent = FrontendAgent(
        provider=settings.default_provider,
        model=settings.model_name
    )
    
    # Wrap _call_with_system to intercept and print the exact prompts sent to the LLM
    original_call_with_system = agent._call_with_system
    prompt_counter = 0

    def prompt_logging_wrapper(sys_prompt: str, user_prompt: str, agent_name: str = "") -> str:
        nonlocal prompt_counter
        prompt_counter += 1
        print("\n" + "═" * 80)
        print(f"  PROMPT CALL #{prompt_counter} | Agent Task: {agent_name}")
        print("═" * 80)
        print("\n┌── [ SYSTEM PROMPT ] ──────────────────────────────────────────────────────────")
        print(sys_prompt)
        print("└───────────────────────────────────────────────────────────────────────────────")
        print("\n┌── [ USER PROMPT SENT TO LLM ] ────────────────────────────────────────────────")
        print(user_prompt)
        print("└───────────────────────────────────────────────────────────────────────────────\n")
        
        return original_call_with_system(sys_prompt, user_prompt, agent_name=agent_name)

    agent._call_with_system = prompt_logging_wrapper

    print(f"\nFrontendAgent initialized with provider: {agent.provider.provider_name if hasattr(agent, 'provider') and agent.provider else 'None'}")
    print("Running FrontendAgent ...\n")
    
    result_dict = agent.run(state)
    
    # Clean up test files if generated on disk
    for fp in [fm1.filepath, fm2.filepath]:
        target_path = os.path.join(workspace_dir, "output", fp)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                print(f"Cleaned up test file: {target_path}")
            except Exception as exc:
                print(f"Could not remove test file {target_path}: {exc}")

    duration = time.perf_counter() - t0
    print("\n==================================================")
    print(f"FRONTEND AGENT PROMPT TEST COMPLETE ({prompt_counter} prompts displayed in {duration:.2f}s)")
    print("==================================================")

if __name__ == "__main__":
    test_frontend_agent_prompts()
