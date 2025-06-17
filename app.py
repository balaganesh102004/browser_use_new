import asyncio
import subprocess
import queue
import logging
import re
from browser_use import Agent
from flask import Flask, render_template, request, redirect, send_file, send_from_directory, url_for, jsonify, Response, session
import os
import uuid
from datetime import datetime
import json
import hashlib
from functools import wraps
from task_manager import load_tasks, save_tasks, get_next_id, count_files_in_folder
from container_manager import allocate_container_to_user,release_container_for_user,start_container
from task_executor import get_llm
from report_generator import render_report

app = Flask(__name__)
app.secret_key = os.urandom(24)

DATA_FILE = 'tasks.json'
SCREENSHOTS_FOLDER = os.path.join(os.getcwd(), 'screenshots')
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("execution_logs.txt", mode="w")
    ]
)
logger = logging.getLogger('browser_use')
logger.setLevel(logging.INFO)

# Thread-safe queue for logs
log_queue = queue.Queue()

class LogTextHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.timestamp_pattern = re.compile(
            r'(\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?(?:\s*[+-]\d{4})?\]|'
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?(?:\s*[+-]\d{4})?|'
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:,\d+)?(?:Z|[+-]\d{2}:\d{2})?)'
        )

    def emit(self, record):
        msg = record.getMessage()
        has_timestamp = bool(self.timestamp_pattern.search(msg))
        if has_timestamp:
            formatted_msg = msg
        else:
            formatted_msg = self.format(record)
        log_queue.put(formatted_msg)

# Set up logging handler
logger = logging.getLogger('browser_use')
logger.setLevel(logging.INFO)
log_handler = LogTextHandler()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

# --- Authentication helpers ---
USERS_FILE = 'users.json'

def save_user(username, password_hash):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r+', encoding='utf-8') as f:
            users = json.load(f)
            if username in users:
                return False  # User already exists
            users[username] = password_hash
            f.seek(0)
            json.dump(users, f, ensure_ascii=False, indent=4)
    else:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump({username: password_hash}, f, ensure_ascii=False, indent=4)
    return True

def get_user_id(username):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
            if username in users:
                return list(users.keys()).index(username) + 1
    return None

def check_user(username, password_hash):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
            if username in users and users[username] == password_hash:
                return True
    return False

# --- Authentication routes ---
@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/auth', methods=['POST'])
def auth():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Load users
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            try:
                users = json.load(f)
            except Exception:
                users = {}

    # Signup
    if username not in users:
        users[username] = password_hash
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
    elif users[username] != password_hash:
        return jsonify({"status": "fail", "message": "Invalid credentials"})

    # Both signup and login: assign user_id and container
    user_id = list(users.keys()).index(username) + 1
    session['user_id'] = user_id
    session['username'] = username
    return jsonify({"status": "success", "user_id": user_id, "message": "Login/Signup successful"})

# --- Task CRUD routes ---
@app.route('/')
@login_required
def index():
    username = session.get('username')  # Get the username from the session

    tasks = load_tasks()
    tasks_count = len(tasks)
    history_folder = os.path.join(os.path.dirname(__file__), "history", username)
    os.makedirs(history_folder, exist_ok=True)  # Ensure the folder exists
    history_count = count_files_in_folder(history_folder)
    return render_template('index.html', tasks=tasks, tasks_count=tasks_count, history_count=history_count)

@app.route('/add', methods=['POST'])
def add_task():
    tasks = load_tasks()
    new_task = {
        "ID": get_next_id(tasks),
        "Task name": request.form['task_name'],
        "Task description": request.form['task_description'],
        "Tags": request.form['tags'].split(',')
    }
    tasks.append(new_task)
    save_tasks(tasks)
    return redirect(url_for('index'))

@app.route('/update/<int:task_id>', methods=['POST'])
def update_task(task_id):
    tasks = load_tasks()
    for task in tasks:
        if task["ID"] == task_id:
            task["Task name"] = request.form['task_name']
            task["Task description"] = request.form['task_description']
            task["Tags"] = request.form['tags'].split(',')
            break
    save_tasks(tasks)
    return redirect(url_for('index'))

@app.route('/delete/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    tasks = load_tasks()
    tasks = [task for task in tasks if task["ID"] != task_id]
    save_tasks(tasks)
    return redirect(url_for('index'))

# --- Task execution and reporting ---
@app.route('/run', methods=['POST'])
@login_required
def run():
    username = session.get('username')
    test_run_id = str(uuid.uuid4())[:8]
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    all_tasks = load_tasks()

    selected_task_names = request.form.getlist('tasks[]')
    selected_tasks = [task for task in all_tasks if task["Task name"] in selected_task_names]
    selected_descriptions = [task["Task description"] for task in selected_tasks]

    container_name = allocate_container_to_user(username)
    try:
        if container_name is None:
            return "All containers are busy. You're in the queue. Please wait..."
        else:
            task_args = " ".join(f'"{desc}"' for desc in selected_descriptions)
            cmd = [
                "podman", "exec", container_name,
                "bash", "-c",
                f"DISPLAY=:99 Xvfb :99 -screen 0 1920x1080x24 & python agent_runner.py {task_args}"
            ]

            print(f"Running tasks in container {container_name} with descriptions: {selected_descriptions}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8"
            )
        output_lines = []
        log_lines = []

        for line in process.stdout:
            print(line, end='')  # Terminal
            log_queue.put(line.rstrip())  # For live streaming
            log_lines.append(line.rstrip())  # For log file
            output_lines.append(line.rstrip())  # For result extraction
        process.wait()

        # Remove empty lines
        output_lines = [l for l in output_lines if l.strip()]
        if output_lines:
            agent_json_line = output_lines[-1]
            log_lines_only = output_lines[:-1]
        else:
            agent_json_line = ''
            log_lines_only = output_lines

        history_folder = os.path.join(os.path.dirname(__file__), "history", username)
        os.makedirs(history_folder, exist_ok=True)

        log_filename = f"test_run_{timestamp}_{test_run_id}_log.txt"
        with open(os.path.join(history_folder, log_filename), "w", encoding="utf-8") as f:
            f.write('\n'.join(log_lines_only))

        try:
            raw_results = json.loads(agent_json_line)
        except Exception as e:
            print("Error parsing agent_runner output:", e)
            raw_results = [agent_json_line for _ in selected_descriptions]
        
        with open(os.path.join(history_folder, "history_only.json"), "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=4)

        history_data = {
            name: output
            for name, output in zip(selected_task_names, raw_results)
        }

        # Save the HTML report as before
        html_filename = f"test_run_{timestamp}_{test_run_id}.html"
        html_report = render_report(history_data, test_run_id=test_run_id, all_tasks=all_tasks)

        with open(os.path.join(history_folder, html_filename), "w", encoding="utf-8") as f:
            f.write(html_report)
    finally:
        release_container_for_user(username)

    return Response(html_report, mimetype='text/html')

@app.route('/history')
@login_required
def history():
    username = session.get('username')
    history_folder = os.path.join(os.path.dirname(__file__), "history", username)
    runs = []
    if os.path.exists(history_folder):
        for fname in sorted(os.listdir(history_folder), reverse=True):
            if fname.endswith(".html"):
                with open(os.path.join(history_folder, fname), encoding="utf-8") as f:
                    content = f.read()
                    def extract(field):
                        import re
                        m = re.search(rf'<strong>{field}:</strong>\s*([^<]+)', content)
                        return m.group(1).strip() if m else "N/A"
                    runs.append({
                        "id": extract("Test Run ID"),
                        "file": fname,
                        "status": extract("Status"),
                        "duration": extract("Total Duration"),
                        "tokens": extract("Total Tokens"),  
                        "start": extract("Start Time"),
                        "end": extract("End Time"),
                    })
    return render_template("history.html", runs=runs)

@app.route('/history/<filename>')
@login_required
def history_report(filename):
    username = session.get('username')
    history_folder = os.path.join(os.path.dirname(__file__), "history", username)
    return send_file(os.path.join(history_folder, filename))

@app.route('/history/delete/<filename>', methods=['POST'])
def delete_history(filename):
    history_folder = os.path.join(os.path.dirname(__file__), "History")
    file_path = os.path.join(history_folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "File not found"}), 404

SETTINGS_FILE = 'settings.json'
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        # Default settings
        return {
            "agent_llm": "gemini",
            "agent_llm_args": {"gemini_api_key": ""},
            "planner_llm": "gemini",
            "planner_llm_args": {"gemini_api_key": ""},
            "headless_mode": False,
            "highlight_elements": False
        }
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        agent_llm = request.form.get('agent_llm', 'gemini')
        planner_llm = request.form.get('planner_llm', 'gemini')
        agent_llm_args = {}
        planner_llm_args = {}

        # Collect dynamic LLM args
        for key in request.form:
            if key.startswith('agent-llm_'):
                arg_name = key.replace('agent-llm_', '')
                agent_llm_args[arg_name] = request.form[key]
            if key.startswith('planner-llm_'):
                arg_name = key.replace('planner-llm_', '')
                planner_llm_args[arg_name] = request.form[key]

        headless_mode = 'headless_mode' in request.form
        highlight_elements = 'highlight_elements' in request.form

        settings_obj = {
            "agent_llm": agent_llm,
            "agent_llm_args": agent_llm_args,
            "planner_llm": planner_llm,
            "planner_llm_args": planner_llm_args,
            "headless_mode": headless_mode,
            "highlight_elements": highlight_elements
        }
        save_settings(settings_obj)
        return redirect(url_for('settings'))

    settings_obj = load_settings()
    # Pass LLM args as JS variables for dynamic rendering
    return render_template(
        'settings.html',
        settings=settings_obj,
        agentLlmSettings=settings_obj.get("agent_llm_args", {}),
        plannerLlmSettings=settings_obj.get("planner_llm_args", {})
    )

@app.route('/generate')
def generate():
    return render_template('generate.html')

# Ensure the screenshots folder exists
SCREENSHOTS_FOLDER = os.path.join(os.getcwd(), 'screenshots')
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

@app.route('/api/fetch_screenshot', methods=['POST'])
def api_fetch_screenshot():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        agent = Agent(task=f"go to {url}")
        history = asyncio.run(agent.run())
        screenshots = history.screenshots() if hasattr(history, 'screenshots') else []
        if screenshots:
            # Save the first screenshot to the folder
            screenshot_path = os.path.join(SCREENSHOTS_FOLDER, 'screenshot1.png')
            with open(screenshot_path, 'wb') as f:
                f.write(screenshots[0])
            return jsonify({'screenshot': f'/screenshots/screenshot1.png'})
        return jsonify({'error': 'No screenshot found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fetch_all_screenshots', methods=['POST'])
def api_fetch_all_screenshots():
    url = request.json.get('url')
    info = request.json.get('info', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        prompt = f"{info}\ngo to {url}" if info else f"go to {url}"
        agent = Agent(task=prompt)
        history = asyncio.run(agent.run())
        screenshots = history.screenshots() if hasattr(history, 'screenshots') else []
        screenshot_paths = []
        for i, screenshot in enumerate(screenshots):
            screenshot_path = os.path.join(SCREENSHOTS_FOLDER, f'screenshot{i + 1}.png')
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot)
            screenshot_paths.append(f'/screenshots/screenshot{i + 1}.png')
        return jsonify({'screenshots': screenshot_paths})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOTS_FOLDER, filename)

@app.route('/api/generate_scenarios', methods=['POST'])
def api_generate_scenarios():
    data = request.json
    url = data.get('url', '')
    info = data.get('info', '')
    screenshots = data.get('screenshots', [])
    if not screenshots:
        return jsonify({'error': 'No screenshots provided'}), 400

    # Compose the prompt as per your reference code
    task_prompt = f"Additional Notes: {info}" if info else ""
    url_prompt = f"Refer URL if provided: {url}" if url else ""
    prompt = f"""
    You are a senior QA automation engineer. 
    Generate an elaborate end-to-end test scenario that covers all relevant test cases for the given feature in a single test flow, group by features. 

    Each scenario should:
    Include multiple sequential steps that mimic a real user or system behavior.
    Always start the test scenario with a opening a web page or application.
    Clearly describe the actions to be performed, including any necessary inputs or interactions.
    Clearly specify input test data and expected outcomes.
    Include assertions after each significant step to validate system behavior.
    Mention any preconditions, setup data, or environment assumptions.
    Cover happy path, edge cases, and any negative checks that can be reasonably tested within the flow.
    Be written in a clear, concise, and structured format (you may use Gherkin-style, pseudocode, or narrative style depending on clarity).
    Finally, ensure that the scenarios are grouped logically by feature or functionality to avoid redundancy and improve readability.
    Each scenario should be self-contained and not rely on external context or previous scenarios.
    
    The goal is to have a comprehensive grouped testcases that effectively verifies most aspects of the feature under test through a realistic and comprehensive scenario.

    Analyze this image of a web page or application UI.
    {url_prompt}

    {task_prompt}

    For the scenario, provide:
    1. A descriptive name
    2. A detailed comprehensive conjunctive instruction to perform the test scenario
    3. Relevant tags (comma-separated)
    
    Format your response as JSON like this:
    {
        "scenarios": [
        {
            "name": "test scenario name",
            "description": "Go to the https://url and do all the potential login related steps...",
            "tags": "sample, tags, here"
        },
        ...
        ]
    }
    """

    # Use the first screenshot for LLM (extend to all if needed)
    import base64, re
    from langchain_core.messages import HumanMessage, SystemMessage
    try:
        llm = get_llm()  # You should implement get_llm() as in your reference
        messages = [
            SystemMessage(content="You are a QA expert specializing in identifying test scenarios from UI images."),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": screenshots[0]}}
            ])
        ]
        response = llm.invoke(messages)
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            scenarios_data = json.loads(json_match.group(0))
            return jsonify({'scenarios': scenarios_data.get("scenarios", [])})
        else:
            return jsonify({'error': 'Could not parse JSON from LLM response'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            message = log_queue.get()  
            yield f"data: {message}\n"
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(port=5000)