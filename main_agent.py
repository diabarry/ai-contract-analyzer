from src.agent import app  # Imports the compiled LangGraph workflow from src/agent.py
import pprint
import src.bootstrap

def run_contract_assistant(query: str, stream_mode: bool = True):
    """
    Executes the assistant with streaming support to monitor graph state transitions step-by-step.
    """
    # Initial state initialization
    initial_state = {
        "question": query,
        "loop_count": 0,
        "documents": []
    }

    # Graph configuration (recursion limit added as a safeguard against infinite routing loops)
    config = {"recursion_limit": 15}

    print(f" Launching AI Contract Analysis...")
    print(f" Query: {query}\n")
    print("-" * 30)

    if stream_mode:
        final_state = initial_state # Tracks the cumulative updated state across steps
        
        for output in app.stream(initial_state, config=config):
            for node_name, state_update in output.items():
                print(f"📍 Active Node Reached: {node_name.upper()}")
                print(f"DEBUG: Current state length - Documents: {len(final_state.get('documents', []))}")
                # Updates the local state tracker with the modifications made by the current node
                final_state.update(state_update)
                
        print("\n" + "="*50)
        print(" FINAL GENERATED RESPONSE:")
        # Extracts the final response string directly from the compiled state
        print(final_state.get("generation", "⚠️ No generation payload found in state."))
        
        if final_state.get("documents"):
            print("\n📚 REFERENCED SOURCE DOCUMENTS:")
            for i, doc in enumerate(final_state["documents"]):
                source = doc.metadata.get('source', 'Unknown Document')
                print(f"   {i+1}. {source}")
        print("="*50)
        
    else:
        # INVOKE MODE: Classic monolithic execution (returns everything at once)
        try:
            final_state = app.invoke(initial_state, config=config)
            print("\n" + "="*50)
            print("🏁 FINAL GENERATED RESPONSE:")
            pprint.pprint(final_state.get("generation"))
            
            # Displays cited source documents to ensure auditability and prevent hallucinations
            if "documents" in final_state and final_state["documents"]:
                print("\n REFERENCED SOURCE DOCUMENTS:")
                for i, doc in enumerate(final_state["documents"]):
                    source = getattr(doc, 'metadata', {}).get('source', 'Unknown Document')
                    print(f"   {i+1}. {source}")
            print("="*50)
        except Exception as e:
            print(f"Error encountered during graph execution: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ambiguous French query used to actively evaluate the 'rewrite' feedback loop node
    test_query = "Truc pour ma voiture"
    
    # Toggle stream_mode=False if you prefer standard single-payload invocation
    run_contract_assistant(test_query, stream_mode=True)