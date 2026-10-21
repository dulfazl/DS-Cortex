"""Phase 0 hello-world agent. Run `adk web agents` from the repo root, pick "hello"
and ask "how many rows does tests/fixtures/classification.csv have?"."""
from google.adk.agents import Agent

from app.config import get_settings
from tools.pandas_tools import describe_csv

root_agent = Agent(
    name="hello",
    model=get_settings().gemini_model,
    instruction="Answer in one sentence. When asked about a CSV file, call describe_csv and report its numbers exactly.",
    tools=[describe_csv],
)
