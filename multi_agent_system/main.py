from threading import Thread
import subprocess
import time
import sys
# List of all agents to launch
AGENT_SCRIPTS = [
    #"concept_agent.py",
    "similarity_agent.py",
    "relation_agent.py",
    "goal_agent.py",
    "contradiction_agent.py",
    "critic_agent.py",
    "debater_agent.py",
    "verifier_agent.py",
    "response_agent.py",
    "coordinator_agent.py"
]

def launch_agent(script):
    """Launch each agent in a separate subprocess."""
    print(f"[Launcher] Starting {script}...")
    subprocess.Popen([sys.executable, script])

if __name__ == "__main__":
    print(" Launching Multi-Agent System...\n")
    for script in AGENT_SCRIPTS:
        Thread(target=launch_agent, args=(script,), daemon=True).start()

    print(" All agents launched. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n Shutdown requested. Exiting launcher.")
