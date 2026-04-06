# ============================================================
# Tools/test.py
# Test runner — verifies the LangGraph ReAct agent works
# Run from the project root: python Tools/test.py
# ============================================================

import asyncio
import sys
import os

# Add project root to path so we can import graph.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import run_agent

# ─── TEST CASES ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "Quiz Test Case (Multi-step)",
        "query": (
            "What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
            "Now get the age of PM and tell us will this weather suits PM health."
        )
    },
    {
        "name": "Single Tool — Weather",
        "query": "What is the current weather in Islamabad?"
    },
    {
        "name": "Single Tool — Search",
        "query": "Who won the 2024 US Presidential election?"
    },
    {
        "name": "Multi-step Math",
        "query": "What is 15 multiplied by 37, then subtract 200 from the result?"
    },
]


async def run_tests():
    print("=" * 60)
    print("  LangGraph ReAct Agent — Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}] {test['name']}")
        print(f"Query: {test['query']}\n")

        try:
            answer = await run_agent(test["query"])
            if answer and len(answer) > 10:
                print(f"✅ PASSED — Got answer ({len(answer)} chars)")
                passed += 1
            else:
                print(f"⚠️  SUSPECT — Answer too short: '{answer}'")
                failed += 1
        except Exception as e:
            print(f"❌ FAILED — Exception: {e}")
            failed += 1

        print("-" * 60)

    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")


if __name__ == "__main__":
    asyncio.run(run_tests())