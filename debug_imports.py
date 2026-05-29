import sys
import time

print("🟢 [STEP 1] Starting import tracking...")
sys.stdout.flush()

try:
    import streamlit as st
    print("✅ Streamlit imported successfully.")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ Streamlit failed: {e}")

try:
    print("⏳ Loading src.parser (Docling / PyMuPDF)...")
    sys.stdout.flush()
    from src.parser import ContractParser
    print("✅ src.parser imported successfully.")
    sys.stdout.flush()
except Exception as e:
    print(f"❌ src.parser failed: {e}")


print("🟢 [STEP 2] Surgical analysis of src/agent.py...")
sys.stdout.flush()

try:
    print("⏳ Testing LangGraph import (StateGraph)...")
    sys.stdout.flush()
    from langgraph.graph import StateGraph, END
    print("✅ LangGraph OK.")
except Exception as e:
    print(f"❌ LangGraph failed: {e}")

try:
    print("⏳ Testing Mistral import (langchain_mistralai)...")
    sys.stdout.flush()
    from langchain_mistralai import ChatMistralAI
    print(" Mistral OK.")
except Exception as e:
    print(f" Mistral failed: {e}")

try:
    print("Testing internal modules import (database, ingest, etc.)...")
    sys.stdout.flush()
    # Testing if ChromaDB (hidden behind database) is causing the crash
    from src.database import get_vectorstore
    print(" src.database OK.")
except Exception as e:
    print(f" src.database failed: {e}")

print(" End of Step 2.")

# try:
#     print(" Loading src.agent (LangGraph / Mistral / ChromaDB)...")
#     sys.stdout.flush()
#     from src.agent import app
#     print(" src.agent imported successfully.")
#     sys.stdout.flush()
# except Exception as e:
#     print(f" src.agent failed: {e}")

# print(" Script finished without crashing.")