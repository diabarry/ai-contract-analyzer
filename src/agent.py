from typing import List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

from src.nodes import retrieve_node, grade_documents_node, rewrite_node, generate_node

class AgentState(TypedDict):
    question: str
    # Use operator.add to accumulate documents across iterations, 
    # or leave default to overwrite/replace them in each loop.
    documents: List[str] 
    generation: str
    loop_count: int
    technical_query: str # Stores the rewritten/optimized technical query version


# Decision Logic (Router)
def decide_to_generate(state: AgentState):
    """
    Determines whether to generate a response or rewrite the query for a new search.
    """
    print("--- 🤖 ROUTER DECISION ---")
    relevant_docs = state["documents"]
    
    if not relevant_docs or len(relevant_docs) == 0:
        # If no documents are deemed relevant after grading
        if state.get("loop_count", 0) < 3:
            print("❌ No relevant documents found. Routing to REWRITE.")
            return "rewrite"
        else:
            print("⚠️ Loop limit reached. Forcing final GENERATE step.")
            return "generate"
    
    print("✨ High-quality documents found. Routing to GENERATE.")
    return "generate"

# 3. Agentic Graph Construction
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)

# Define workflow topology
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# CONDITIONAL EDGE: The core routing logic of the agent
workflow.add_conditional_edges(
    "grade_documents", 
    decide_to_generate,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

# Feedback loop: after rewriting, execute a new retrieval phase
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile the workflow
app = workflow.compile()