import subprocess
import json
import os
from collections import deque

NUM_CONTAINERS = 5
container_names = [f"browser_use_user_{i+1}" for i in range(NUM_CONTAINERS)]
USER_CONTAINER_MAP = "user_container_map.json"
WAIT_QUEUE = deque()

def load_user_container_map():
    if not os.path.exists(USER_CONTAINER_MAP):
        return {}
    with open(USER_CONTAINER_MAP, 'r') as f:
        return json.load(f)

def save_user_container_map(mapping):
    with open(USER_CONTAINER_MAP, 'w') as f:
        json.dump(mapping, f, indent=4)

def start_container(container_name):
    """Create and start a Podman container with shared volume and environment settings."""
    image_name = "podman_deploy"  
    shared_host_path = os.path.abspath("shared")
    container_mount_path = "/app/shared"
    os.makedirs(shared_host_path, exist_ok=True)

    result = subprocess.run(
        ["podman", "ps", "-a", "--format", "{{.Names}}"],
        stdout=subprocess.PIPE,
        text=True
    )
    existing_containers = result.stdout.splitlines()
    if container_name not in existing_containers:
        print(f"Creating and starting new container: {container_name}")
        subprocess.run([
            "podman", "run", "-d",
            "--name", container_name,
            "--env", "DISPLAY=:99",
            "--env", f"CONTAINER_NAME={container_name}",
            "--privileged",
            "-v", f"{shared_host_path}:{container_mount_path}:Z",
            image_name,
            "sleep", "infinity"
        ], check=True)
    else:
        print(f"Starting existing container: {container_name}")
        subprocess.run(["podman", "start", container_name], check=True)


def allocate_container_to_user(username):
    """Allocates a container to the user or queues them."""
    mapping = load_user_container_map()
    if username in mapping:
        return mapping[username]

    assigned = set(mapping.values())
    available = [c for c in container_names if c not in assigned]

    if available:
        container = available[0]
        start_container(container)
        mapping[username] = container
        save_user_container_map(mapping)
        return container
    else:
        if username not in WAIT_QUEUE:
            WAIT_QUEUE.append(username)
        return None 

def release_container_for_user(username):
    """Releases a container and assigns it to next user in queue (if any)."""
    mapping = load_user_container_map()
    if username not in mapping:
        return

    container = mapping[username]
    subprocess.run(["podman", "rm", "-f", container])
    del mapping[username]
    save_user_container_map(mapping)
    if WAIT_QUEUE:
        next_user = WAIT_QUEUE.popleft()
        allocate_container_to_user(next_user)
