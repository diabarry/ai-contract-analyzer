import os
import shutil

import src.bootstrap

from dotenv import load_dotenv

from src.parser import ContractParser
from src.ingest import ContractIngester
from langchain_community.vectorstores import FAISS 
from src.database import get_embeddings

# Load environment variables (e.g., API keys, configuration)
load_dotenv()

if os.getenv("HF_TOKEN"):
    print("✅ Hugging Face token loaded")

FAISS_PATH = "faiss_index"


def run_pipeline(data_path="./data"):
    """
    Orchestrates the ingestion pipeline: parses PDFs, cleans text, 
    generates embeddings, and saves the vector index locally.
    """
    embedder = get_embeddings()
    vectorstore = None

    print("\n STARTING FULL INGESTION PIPELINE")

    # =====================================================
    # CLEAN PREVIOUS FAISS INDEX
    # =====================================================
    # Remove old index to ensure a fresh state for the current run
    if os.path.exists(FAISS_PATH):
        print("🗑️ Removing previous FAISS index...")
        shutil.rmtree(FAISS_PATH)

    # =====================================================
    # INIT COMPONENTS
    # =====================================================
    parser = ContractParser()
    ingester = ContractIngester()
    vectorstore = None

    # =====================================================
    # LOAD PDF FILES
    # =====================================================
    # Retrieve all PDF files from the target directory
    pdf_files = [
        f for f in os.listdir(data_path)
        if f.endswith(".pdf")
    ]

    print(f"📚 PDF detected: {len(pdf_files)}")
    total_chunks = 0

    # =====================================================
    # INGESTION LOOP
    # =====================================================
    for pdf_file in pdf_files:
        try:
            print(f"\n📄 Processing: {pdf_file}")
            full_path = os.path.join(data_path, pdf_file)

            # =============================================
            # PARSING
            # =============================================
            # Convert raw PDF into structured Markdown
            md_text = parser.convert_to_markdown(full_path)

            if not md_text.strip():
                print("⚠️ Empty parsed document")
                continue

            # =============================================
            # CHUNKING
            # =============================================
            # Split document into clean, meaningful semantic chunks
            chunks = ingester.process_text(md_text, pdf_file)

            if not chunks:
                print("⚠️ No chunks generated")
                continue

            print(f"✂️ Generated chunks: {len(chunks)}")

            # =============================================
            # VECTOR INDEXING
            # =============================================
            if vectorstore is None:
                # Initialize FAISS index with the first batch of documents
                print("🏗️ Initializing FAISS index with first batch...")
                vectorstore = FAISS.from_documents(documents=chunks, embedding=embedder)
            else:
                # Add subsequent chunks to the existing index
                vectorstore.add_documents(documents=chunks)

            total_chunks += len(chunks)

        except Exception as e:
            print(f"❌ Failed processing {pdf_file}")
            print(e)

    # =====================================================
    # SAVE FINAL INDEX
    # =====================================================
    if vectorstore is not None:
        print("\n💾 Saving FAISS index...")
        vectorstore.save_local(FAISS_PATH)
        print("\n======================================")
        print("✅ INGESTION COMPLETED")
        print(f"📦 Total indexed chunks: {total_chunks}")
        print("======================================")
    else:
        print("\n⚠️ Ingestion aborted: No documents were successfully processed.")

if __name__ == "__main__":
    run_pipeline()