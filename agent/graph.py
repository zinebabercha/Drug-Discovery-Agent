from langgraph.graph import StateGraph, END
from state import AgentState
from strategy import strategist_node, tool_executor_node, synthesizer_node

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("strategist", strategist_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("synthesizer", synthesizer_node)

# Set entry point
workflow.set_entry_point("strategist")

# Define edges
# From Strategist to Tool Executor
workflow.add_edge("strategist", "tool_executor")

# From Tool Executor to Synthesizer (after all planned tools are executed)
workflow.add_edge("tool_executor", "synthesizer")

# From Synthesizer to END (after the report is generated)
workflow.add_edge("synthesizer", END)

# Compile the graph
app = workflow.compile()
print("LangGraph workflow compiled!")