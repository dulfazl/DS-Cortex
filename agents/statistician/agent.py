"""ADK definition of the statistician agent for interactive exploration: `adk web agents` from the repo root
(with the repo root on PYTHONPATH). The pipeline uses agents/statistician/core.py."""
from google.adk.agents import Agent

from agents.adk_tools import statistics_csv
from app.config import get_settings
from app.llm import load_prompt

root_agent = Agent(name="statistician", model=get_settings().gemini_model, instruction=load_prompt("statistician"), tools=[statistics_csv])
