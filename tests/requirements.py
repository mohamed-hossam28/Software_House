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

from core.state import ProjectState
from agents.Requirements import RequirementsAgent

def test_requirements_agent():
    print("==================================================")
    print("STARTING TEST FOR RequirementsAgent")
    print("==================================================")
    t0=time.perf_counter()
    print("\n--- STEP 1: Initializing ProjectState ---")
    state = ProjectState(project_id="test_req", project_name="Test Requirements")
    state.requirements.raw_prompt = "Build a todo app"
    print("\n--- STEP 2: Initializing RequirementsAgent ---")
    agent = RequirementsAgent(
        provider=settings.default_provider,
        model=settings.model_name
    )
    print(f"Agent name: {agent.name}")
    print(f"Agent model: {agent.model}")
    print(f"Agent provider: {agent.provider.provider_name if hasattr(agent, 'provider') and agent.provider else 'None'}")
    
    print("\n--- STEP 3: Running RequirementsAgent ---")
    result_state = agent.run(state)
    
    print("\n--- STEP 4: Execution finished. Printing final state ---")
    print(json.dumps(result_state.to_dict(), indent=2))

    duration=time.perf_counter()-t0
    print(f"duration = {duration}" )
    
    
    print("\n==================================================")
    print("RequirementsAgent Test Complete")
    print("==================================================")

if __name__ == "__main__":
    test_requirements_agent()
