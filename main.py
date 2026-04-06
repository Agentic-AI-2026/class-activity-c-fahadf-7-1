# ============================================================
# main.py
# Entry point — runs the LangGraph ReAct agent.
# ============================================================

import asyncio
from graph import run_agent

# ─── TEST QUERY from the quiz ─────────────────────────────────────────────────
QUERY = (
    "What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
    "Now get the age of PM and tell us will this weather suits PM health."
)

if __name__ == "__main__":
    asyncio.run(run_agent(QUERY))