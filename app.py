import os
import sys
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# Import bootstrap for environment integrity
import src.bootstrap  

# --- STREAMLIT CONFIGURATION & CACHING ---
st.set_page_config(
    page_title="AI Contract Analyzer | Senior MLE Portfolio", 
    page_icon="⚖️", 
    layout="wide"
)

# 1. Path management to locate the 'src' directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# App imports
from src.parser import ContractParser 
from src.agent import app 
from src.ingest import ContractIngester  

# Robust resource caching with TTL to prevent resource corruption on Windows
@st.cache_resource(show_spinner=False, ttl=3600)
def get_parser():
    return ContractParser()

@st.cache_resource(show_spinner=False, ttl=3600)
def get_ingester():
    return ContractIngester()

contract_parser = get_parser()
user_ingester = get_ingester()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📤 Contract Benchmarker")
    uploaded_file = st.file_uploader("Upload your target contract (PDF)", type="pdf")
    
    if uploaded_file:
        # Validate PDF to prevent empty file crashes
        if uploaded_file.size == 0:
            st.error("❌ The uploaded PDF file is empty.")
            st.stop()
            
        if "user_contract" not in st.session_state:
            with st.spinner("Parsing and extracting document metrics..."):
                # Use tempfile and handle system locks for Windows compatibility
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                try:
                    # High-fidelity document conversion
                    user_contract_md = contract_parser.convert_to_markdown(temp_path)
                    st.session_state["user_contract"] = user_contract_md
                except Exception as parse_error:
                    st.error(f"❌ Error during PDF parsing: {parse_error}")
                finally:
                    # Immediate removal of temporary file to release system lock
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
          
        st.success("✅ Contract loaded successfully into context!")
        
        if st.button("🗑️ Clear Active Contract"):
            if "user_contract" in st.session_state:
                del st.session_state["user_contract"]
            # Reset chat history on contract reset
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("📄 Reference Repository")
    if os.path.exists("./data"):
        pdf_list = [f for f in os.listdir("./data") if f.endswith('.pdf')]
        for pdf in pdf_list:
            st.caption(f"• {pdf}")
    else:
        st.warning("Landing directory '/data' not detected.")

    st.divider()
    st.subheader("🛠️ Tech Stack")
    st.markdown("""
    - **Orchestration**: LangGraph (Self-RAG)
    - **LLM Engine**: Mistral AI
    - **Vector Search**: FAISS
    - **Parsing**: PyMuPDF4LLM
    - **Interface**: Streamlit
    """)

# --- MAIN INTERFACE ---
st.title("⚖️ AI Contract Analyzer & Benchmarker")
st.caption("Agentic workflow orchestration for deep insurance clause comparison")

# --- AUDIT & COMPARISON SECTION ---
if "user_contract" in st.session_state:
    with st.container(border=True):
        st.subheader("📊 Active Benchmark Comparison")
        st.write("Target document staged. The orchestrator is ready to benchmark your clauses against the baseline repository.")
        
        if st.button("🚀 Run Comparative Audit"):
            with st.status("Agent analyzing contract gaps...", expanded=True) as status:
                st.write("🔍 Running structural chunking and semantic extraction on uploaded asset...")
                
                user_chunks = user_ingester.process_text(st.session_state['user_contract'], "user_upload.pdf")
                user_texts = [chunk.page_content for chunk in user_chunks]
                
                # Prevent Context Overflow (Limit to 12k chars)
                MAX_CONTEXT_CHARS = 12000
                joined_text = "\n\n".join(user_texts)
                user_contract_summary = joined_text[:MAX_CONTEXT_CHARS]
    
                # Optimized system prompt
                comparison_query = f"""[MISSION] Compare mon contrat actuel avec la base de référence sur les garanties, franchises et exclusions.
[MON CONTRAT] : {user_contract_summary}"""
                
                try:
                    final_state = app.invoke({
                        "question": comparison_query, 
                        "loop_count": 0,
                        "documents": [] 
                    })
                    status.update(label="✅ Comparative audit complete!", state="complete", expanded=False)
                    st.markdown("### 📋 Audit Benchmark Report")
                    st.markdown(final_state.get("generation", "⚠️ The agent failed to generate a report."))
                except Exception as audit_error:
                    status.update(label="❌ Audit failed", state="error", expanded=True)
                    st.error(f"Error during audit execution: {audit_error}")

st.divider()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
prompt = st.chat_input("Ask a question (e.g., 'Quelles sont les exclusions pour le vol ?')")

if prompt and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = ""
        raw_docs = []
        initial_state = {"question": prompt, "loop_count": 0, "documents": []}

        # Global Try/Catch for Graph execution flow
        try:
            with st.status("🧠 Intelligent agent executing routing graph...", expanded=True) as status:
                
                # Stream LangGraph execution
                for step in app.stream(initial_state):
                    for node, values in step.items():
                        st.write(f"📍 Transitioned through node: **{node.upper()}**")

                        # Loop protection check
                        MAX_ITERATIONS = 6
                        if isinstance(values, dict) and values.get("loop_count", 0) > MAX_ITERATIONS:
                            raise Exception("Maximum iterations reached: potential infinite loop detected.")

                        if "generation" in values:
                            final_answer = values["generation"]
                        if "documents" in values:
                            raw_docs = values["documents"]

                status.update(label="✅ Analysis complete", state="complete", expanded=False)

            if not final_answer:
                final_answer = " Agent finished without generating a response."

            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

            # Display sources with safe metadata extraction
            if raw_docs:
                st.markdown("---")
                st.markdown("### 📚 Sources used")
                for i, doc in enumerate(raw_docs):
                    source_name = doc.metadata.get("source", "Unknown Baseline") if hasattr(doc, 'metadata') else "Unknown Baseline"
                    content_preview = doc.page_content[:300] if hasattr(doc, 'page_content') else str(doc)[:300]
                    st.markdown(f"**{i+1}. {source_name}**")
                    st.caption(f"{content_preview}...")
                    st.divider()

        except Exception as graph_error:
            st.error(f" An error occurred during graph execution: {graph_error}")
        
        finally:
            # Maintain chat history window
            MAX_CHAT_HISTORY = 10
            if len(st.session_state.messages) > MAX_CHAT_HISTORY:
                st.session_state.messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
            
            st.session_state.processing = False
            st.rerun()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()