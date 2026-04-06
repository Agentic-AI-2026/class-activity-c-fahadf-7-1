# ============================================================
# graph.py
# LangGraph ReAct Agent — converts the LangChain ReAct loop
# into a proper graph with nodes, edges, and conditional routing.
# ============================================================

import os
import asyncio
from typing import TypedDict, Annotated, List
import operator

#from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, END

# ─── 1. STATE ─────────────────────────────────────────────────────────────────
# This TypedDict holds everything the graph passes between nodes.

class AgentState(TypedDict):
    input: str                          # Original user query
    agent_scratchpad: List              # Full message history (LangChain messages)
    final_answer: str                   # Populated when LLM gives Final Answer
    steps: Annotated[List, operator.add]  # Tracks each Action + Observation


# ─── 2. LLM SETUP ─────────────────────────────────────────────────────────────
# Using Claude. You can swap to Gemini or Ollama by changing this line.

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key="",
    temperature=0
)

# System prompt that enforces the ReAct Thought → Action → Observation loop
REACT_SYSTEM = """You are a ReAct agent. Strictly follow this loop:
Thought → Action (tool call) → Observation → Thought → ...

RULES:
1. ALWAYS use a tool for factual information — never answer from memory.
2. For multi-part questions, make one tool call per fact.
3. ALWAYS use calculator for any arithmetic — never compute in your head.
4. Only give Final Answer AFTER all required tool calls are complete."""


# ─── 3. MCP CLIENT SETUP ──────────────────────────────────────────────────────
# Connects to the math, search, and weather MCP servers.

mcp_client = MultiServerMCPClient({
    "math": {
        "command": "python",
        "args": ["Tools/math_server.py"],
        "transport": "stdio",
    },
    "search": {
        "command": "python",
        "args": ["Tools/search_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
})


async def load_tools():
    """Load all tools from MCP servers and return tools list + tools map."""
    tools = []
    for server in ["math", "search", "weather"]:
        try:
            server_tools = await mcp_client.get_tools(server_name=server)
            tools.extend(server_tools)
            print(f"  ✓ Loaded tools from '{server}': {[t.name for t in server_tools]}")
        except Exception as e:
            print(f"  ✗ Could not load '{server}': {e}")
    tools_map = {t.name: t for t in tools}
    return tools, tools_map


# ─── 4. NODE: react_node ──────────────────────────────────────────────────────
# This node calls the LLM with the current scratchpad.
# The LLM either:
#   (a) calls a tool → we get tool_calls in the response
#   (b) gives a final answer → plain text response, no tool_calls

def make_react_node(llm_with_tools):
    """
    Factory that returns the react_node function.
    We use a factory so we can inject the LLM (with tools bound).
    """
    def react_node(state: AgentState) -> AgentState:
        # Build message list: system + history
        messages = [SystemMessage(content=REACT_SYSTEM)] + state["agent_scratchpad"]

        print(f"\n[react_node] Calling LLM... (scratchpad length: {len(messages)})")

        response = llm_with_tools.invoke(messages)

        print(f"[react_node] Response: tool_calls={bool(response.tool_calls)}, content_preview={str(response.content)[:100]}")

        # Add the LLM response to scratchpad
        new_scratchpad = state["agent_scratchpad"] + [response]

        # If no tool calls → this is the Final Answer
        if not response.tool_calls:
            return {
                **state,
                "agent_scratchpad": new_scratchpad,
                "final_answer": response.content,
            }

        # Otherwise → tool calls pending, final_answer stays empty
        return {
            **state,
            "agent_scratchpad": new_scratchpad,
            "final_answer": "",
        }

    return react_node


# ─── 5. NODE: tool_node ───────────────────────────────────────────────────────
# This node executes whatever tool(s) the LLM requested.
# It appends ToolMessage(s) back to the scratchpad so the LLM sees the results.

def make_tool_node(tools_map: dict):
    """
    Factory that returns the tool_node function.
    We pass in tools_map so the node can look up tools by name.
    """
    def tool_node(state: AgentState) -> AgentState:
        # The last message in scratchpad is the AIMessage with tool_calls
        last_message = state["agent_scratchpad"][-1]

        new_messages = []
        new_steps = []

        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            print(f"[tool_node] Executing '{tool_name}' with args: {tool_args}")

            tool = tools_map.get(tool_name)
            if tool is None:
                result = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    # MCP tools are async — run them in the event loop
                    result = asyncio.get_event_loop().run_until_complete(
                        tool.ainvoke(tool_args)
                    )
                except Exception as e:
                    result = f"Tool error: {e}"

            print(f"[tool_node] Observation: {str(result)[:200]}")

            # Append ToolMessage so LLM sees the result
            new_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )

            # Track this step for logging
            new_steps.append({
                "action": tool_name,
                "args": tool_args,
                "observation": str(result)
            })

        return {
            **state,
            "agent_scratchpad": state["agent_scratchpad"] + new_messages,
            "steps": new_steps,  # operator.add will extend the list
        }

    return tool_node


# ─── 6. ROUTER: conditional edge ──────────────────────────────────────────────
# After react_node runs, we check: did it produce tool calls or a final answer?

def router(state: AgentState) -> str:
    """
    Returns 'tool_node' if LLM wants to call a tool.
    Returns 'end' if LLM produced a Final Answer.
    """
    # final_answer is set only when there are no tool calls
    if state["final_answer"]:
        print("[router] → Final Answer detected → END")
        return "end"
    else:
        print("[router] → Tool call detected → tool_node")
        return "tool_node"


# ─── 7. BUILD GRAPH ───────────────────────────────────────────────────────────

async def build_graph():
    """
    Loads MCP tools, binds them to the LLM, and builds the LangGraph.
    Returns a compiled graph ready to invoke.
    """
    print("\n📦 Loading MCP tools...")
    tools, tools_map = await load_tools()
    print(f"   Total tools loaded: {list(tools_map.keys())}")

    # Bind tools to LLM so it knows what tools are available
    llm_with_tools = llm.bind_tools(tools)

    # Create node functions
    react_fn = make_react_node(llm_with_tools)
    tool_fn  = make_tool_node(tools_map)

    # Build the StateGraph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("react_node", react_fn)
    workflow.add_node("tool_node",  tool_fn)

    # Set entry point
    workflow.set_entry_point("react_node")

    # Conditional edge: after react_node, route based on output
    workflow.add_conditional_edges(
        "react_node",           # from this node
        router,                 # call this function to decide
        {
            "tool_node": "tool_node",   # if router returns "tool_node"
            "end": END                  # if router returns "end"
        }
    )

    # After tool_node always goes back to react_node (the loop)
    workflow.add_edge("tool_node", "react_node")

    # Compile the graph
    graph = workflow.compile()
    print("✅ Graph compiled successfully!\n")
    return graph


# ─── 8. RUN FUNCTION ──────────────────────────────────────────────────────────

async def run_agent(query: str) -> str:
    """
    High-level function to run the LangGraph ReAct agent on a query.
    Returns the final answer string.
    """
    graph = await build_graph()

    # Initial state
    initial_state: AgentState = {
        "input": query,
        "agent_scratchpad": [HumanMessage(content=query)],
        "final_answer": "",
        "steps": [],
    }

    print(f"🚀 Running agent on query:\n   \"{query}\"\n")
    print("=" * 60)

    result = graph.invoke(initial_state)

    print("=" * 60)
    print(f"\n✅ Final Answer:\n{result['final_answer']}")
    print(f"\n📋 Steps taken: {len(result['steps'])}")
    for i, step in enumerate(result["steps"], 1):
        print(f"   Step {i}: [{step['action']}] → {str(step['observation'])[:80]}...")

    return result["final_answer"]