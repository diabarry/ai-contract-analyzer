import os
import warnings
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# Suppress warning messages for cleaner console output
warnings.filterwarnings("ignore")


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Initializes and returns the embedding model using HuggingFace.
    The lru_cache decorator ensures the model is loaded into memory only once.
    """

    # Using a multilingual model optimized for sentence similarity in various languages
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu"  # Force CPU execution for wider deployment compatibility
        },
        encode_kwargs={
            "normalize_embeddings": True,  # Ensures cosine similarity calculations
            "batch_size": 8                # Optimized for memory and inference speed
        }
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    """
    Loads the FAISS vector database from the local disk.
    Returns the vectorstore object or None if the index does not exist.
    """

    embeddings = get_embeddings()

    # Path to the directory where the vector index is persisted
    db_path = "faiss_index"

    # Check if the persisted index exists on disk before attempting to load
    if os.path.exists(db_path):
        return FAISS.load_local(
            db_path,
            embeddings,
            # Required since LangChain update for local index loading
            allow_dangerous_deserialization=True 
        )

    # Return None if no index is found (e.g., first-time setup)
    return None