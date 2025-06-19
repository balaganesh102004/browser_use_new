import sys
import json
import asyncio
import threading
from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from settings_manager import load_settings

browser_session = BrowserSession(
    headless=True,
    user_data_dir=None,
    chromium_sandbox=False,
)

def run_async_in_thread(coro):
    result = {}
    exception = {}

    def runner():
        try:
            result['value'] = asyncio.run(coro)
        except Exception as e:
            exception['error'] = e

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if 'error' in exception:
        raise exception['error']

    return result['value']

def get_browser_profile():
    settings = load_settings()
    return BrowserProfile(
        highlight_elements=settings.get("highlight_elements", False),
        user_data_dir=None,
        headless=settings.get("headless_mode", False)
    )

def get_llm():
    settings = load_settings()
    agent_llm = settings.get("agent_llm", "gemini")
    agent_llm_args = settings.get("agent_llm_args", {})
    agent_model = agent_llm_args.get("model-name", "gemini-2.0-flash-exp")
    if agent_llm == "gemini":
        return ChatGoogleGenerativeAI(
            model=agent_model,
            api_key=SecretStr(agent_llm_args.get("gemini_api_key")),
            temperature=0.2,
            seed=42
        )
    elif agent_llm == "azure_openai":
        return AzureChatOpenAI(
            model=agent_model,
            api_version=agent_llm_args.get("azure_openai_api_version"),
            azure_endpoint=agent_llm_args.get("azure_openai_api_endpoint"),
            api_key=SecretStr(agent_llm_args.get("azure_openai_api_key"))
        )
    else:
        raise ValueError(f"Unsupported agent_llm: {agent_llm}")

async def run_task_async(task):
    llm = get_llm()
    agent = Agent(
        task=task,
        llm=llm,
        browser_profile=get_browser_profile(),browser_session=browser_session,
    )
    try:
        history = await agent.run(max_steps=10)
        return history.model_dump()
    finally:
        if hasattr(agent, "close"):
            await agent.close()

async def run_tasks_concurrently(task_descriptions):
    coros = [run_task_async(description) for description in task_descriptions]
    return await asyncio.gather(*coros)



