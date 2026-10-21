"""ADK definition of the router agent for interactive exploration: `adk web agents` from the repo root
(with the repo root on PYTHONPATH). The pipeline uses agents/router/core.py."""
from google.adk.agents import Agent

from agents.adk_tools import profile_and_precheck
from app.config import get_settings
from app.llm import load_prompt

root_agent = Agent(name="router", model=get_settings().gemini_model, instruction=load_prompt("router"), tools=[profile_and_precheck])
