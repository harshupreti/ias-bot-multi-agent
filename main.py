from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from tools import ALL_TOOLS

# Unpack tools from ALL_TOOLS
(
    semantic_search_tool,
    filter_officers_tool,
    web_search_tool,
    web_trait_search_tool,
    reasoning_tool,
    check_role_intent_tool
) = ALL_TOOLS

# Shared state definition
class AgentState(TypedDict):
    input: str
    filters: dict
    queries: list[str]
    current_title: str
    search_results: list
    reasoning_output: str
    steps: list[str]

# Tool wrappers

def check_role_intent(state: AgentState):
    result = check_role_intent_tool.invoke(state["input"])
    state["queries"] = result["queries"]
    current_title = result.get("current_title")
    if current_title:
        state["filters"]["current_title"] = current_title
    state["steps"].append("Checked role intent.")
    return state

def semantic_search(state: AgentState):
    out = semantic_search_tool.invoke({
        "query": state["queries"],
        "filters": state["filters"],
        "top_k": 2
    })
    state["search_results"] = out if isinstance(out, list) else out.get("results", [])
    state["steps"].append("Performed semantic search.")
    return state

def reasoning(state: AgentState):
    result = reasoning_tool.invoke({
        "query": state["input"],
        "officers": state.get("search_results", []),
        "filters": state["filters"]
    })

    # 👇 Defensive check
    if isinstance(result, dict) and "output" in result:
        state["reasoning_output"] = result["output"]
    elif isinstance(result, str):
        state["reasoning_output"] = result
    else:
        state["reasoning_output"] = "No output generated."

    state["steps"].append("Ran reasoning tool.")
    return state


def filter_only(state: AgentState):
    # Remove None values from filters to avoid validation errors
    clean_filters = {k: v for k, v in state["filters"].items() if v is not None}
    result = filter_officers_tool.invoke({"filters": clean_filters})
    state["search_results"] = result["output"] if isinstance(result, dict) and "output" in result else result
    state["steps"].append("Filtered officers.")
    return state

def web_search(state: AgentState):
    result = web_search_tool.invoke({"input": state["input"]})
    state["search_results"] = result["results"]
    state["steps"].append("Performed web search.")
    return state

def finalize(state: AgentState):
    state["steps"].append("Finalized.")
    return state

# Routers

def start_router(state: AgentState) -> str:
    if state["input"].strip().lower() == "officers":
        return "filter_only"
    return "check_role_intent"

def router_after_reasoning(state: AgentState) -> str:
    reasoning_output = state.get("reasoning_output", "").lower()
    if (
        "i don't know" in reasoning_output
        or "no relevant results" in reasoning_output
        or "not enough information" in reasoning_output
    ):
        return "web_search"
    return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("check_role_intent", check_role_intent)
workflow.add_node("semantic_search", semantic_search)
workflow.add_node("reasoning", reasoning)
workflow.add_node("filter_only", filter_only)
workflow.add_node("web_search", web_search)

workflow.set_conditional_entry_point(start_router, {
    "check_role_intent": "check_role_intent",
    "filter_only": "filter_only"
})

workflow.add_node("finalize", finalize)
workflow.add_edge("check_role_intent", "semantic_search")
workflow.add_edge("semantic_search", "reasoning")
workflow.add_conditional_edges("reasoning", router_after_reasoning,{
    "web_search": "web_search",  # Pass the actual function, not the string
    END: "finalize"
})
workflow.add_edge("finalize", END)
workflow.add_edge("web_search", "reasoning")
workflow.add_edge("filter_only", "reasoning")

app = workflow.compile()