from task_executor import *
import sys
import json
import asyncio

def log_to_file(msg):
    with open("/tmp/agent_runner.log", "a") as f:
        f.write(msg + "\n")

if __name__ == "__main__":
    descriptions = sys.argv[1:]
    try:
        log_to_file(f"Running tasks: {descriptions}")
        results = asyncio.run(run_tasks_concurrently(descriptions))
        print(json.dumps(results))  # Already present or add this
        with open('history_only.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(json.dumps([{"error": str(e)}]))
        sys.exit(1)
