"""ADK definition of the data_auditor agent for interactive exploration: `adk web agents` from the repo root
(with the repo root on PYTHONPATH). The pipeline uses agents/data_auditor/core.py."""
from google.adk.agents import Agent

from agents.adk_tools import audit_csv
from app.config import get_settings
from app.llm import load_prompt

root_agent = Agent(name="data_auditor", model=get_settings().gemini_model, instruction=load_prompt("data_auditor"), tools=[audit_csv])
