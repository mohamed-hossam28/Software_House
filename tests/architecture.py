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

from core.state import ProjectState, SRSData
from agents.architecture import ArchitectureAgent

def test_architecture_agent():
    print("==================================================")
    print("STARTING TEST FOR ArchitectureAgent")
    print("==================================================")
    t0 = time.perf_counter()
    
    print("\n--- STEP 1: Initializing ProjectState with Requirements ---")
    state = ProjectState(project_id="test_arch", project_name="Test Architecture")
    state.requirements = SRSData(
        raw_prompt="Build a simple web-based collaborative task manager with real-time updates.",
        functional_requirements=[
            "User must be able to log in and sign up securely.",
            "User must be able to create, read, update, and delete tasks.",
            "Real-time updates when tasks are updated."
        ],
        non_functional_requirements=[
            "Application response times under 200ms.",
            "Passwords must be encrypted."
        ],
        constraints=[
            "Must use PostgreSQL as the primary database.",
            "Must use FastAPI for the backend."
        ],
        assumptions=[
            "The client is a web browser."
        ],
        out_of_scope=[
            "Mobile applications."
        ],
        is_complete=True
    )
    
    print("\n--- STEP 2: Initializing ArchitectureAgent ---")
    agent = ArchitectureAgent(
        provider=settings.default_provider,
        model=settings.model_name
    )
    print(f"Agent name: {agent.name}")
    print(f"Agent model: {agent.model}")
    print(f"Agent provider: {agent.provider.provider_name if hasattr(agent, 'provider') and agent.provider else 'None'}")
    
    print("\n--- STEP 3: Running ArchitectureAgent ---")
    result_state = agent.run(state)
    
    print("\n--- STEP 4: Execution finished. Printing final state ---")
    print(json.dumps(result_state.to_dict(), indent=2))
    
    duration = time.perf_counter() - t0
    print(f"duration = {duration}")
    
    print("\n==================================================")
    print("ArchitectureAgent Test Complete")
    print("==================================================")

if __name__ == "__main__":
    test_architecture_agent()
