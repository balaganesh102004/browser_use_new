import json
import os

SETTINGS_FILE = 'settings.json'

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
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