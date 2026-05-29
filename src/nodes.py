import time
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retriever import ContractRetriever

from pydantic import BaseModel, Field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

# =========================================================
# PATH MANAGEMENT
# =========================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# SAFE RETRY WRAPPER
# =========================================================

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
)
def safe_llm_invoke(chain, payload):
    """Executes chain invocation with exponential backoff for resilience."""
    return chain.invoke(payload)


# =========================================================
# PYDANTIC STRUCTURE
# =========================================================

class GradeDocuments(BaseModel):
    """Schema for document relevance classification."""
    binary_score: str = Field(
        description="Document relevance score: yes or no"
    )


# =========================================================
# LLM INITIALIZATION
# =========================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    timeout=180,
    max_retries=8,
)

grader_llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    timeout=120,
    max_retries=5,
)

retriever = ContractRetriever()


# =========================================================
# RETRIEVE NODE
# =========================================================
def retrieve_node(state):
    """Handles document retrieval, including special source-specific search."""
    print("--- 🔍 RETRIEVING DOCUMENTS ---")
    question = state["question"]
    question_lower = question.lower()
    
    # Special mode: Search by specific filename if requested in the query
    if any(ext in question_lower for ext in [".pdf"]) and any(k in question_lower for k in ["contrat", "que dit"]):
        import re
        match = re.search(r'([\w\-_]+\.pdf)', question)
        if match:
            filename = match.group(1)
            print(f"📄 SOURCE SEARCH MODE: {filename}")
            docs = retriever.search_by_source(filename)
            return {"documents": docs, "question": question}

    # Standard retrieval mode
    try:
        documents = retriever.search(question, k=8, score_threshold=1.0)
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        documents = []

    print(f"📄 Retrieved docs count: {len(documents)}")
    return {"documents": documents, "question": question}


# =========================================================
# DOCUMENT GRADING NODE
# =========================================================
def grade_documents_node(state):
    """Filters retrieved documents based on their relevance to the question."""
    print("--- ⚖️ GRADING DOCUMENTS ---")
    question, documents = state["question"], state["documents"]
    
    if not documents:
        return {"documents": [], "question": question}

    structured_llm = grader_llm.with_structured_output(GradeDocuments)
    
    # Prompt kept in French as per legal domain requirements
    system_prompt = "Tu es un expert assurance. Réponds UNIQUEMENT 'yes' ou 'no' si le doc est utile."
    grade_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Q: {question}\nDoc: {document}")])
    grader_chain = grade_prompt | structured_llm

    relevant_docs = []
    for d in documents:
        # Minimal pre-filtering to save tokens
        if len(d.page_content.strip()) < 120: continue
            
        try:
            score = grader_chain.invoke({"question": question, "document": d.page_content[:3500]})
            if score.binary_score.lower() == "yes":
                relevant_docs.append(d)
        except Exception as e:
            print(f"⚠️ Grading failure: {e}")

    return {"documents": relevant_docs, "question": question}

# =========================================================
# QUERY REWRITE NODE
# =========================================================

def rewrite_node(state):
    """Transforms user query into optimized keywords for better vector retrieval."""
    print("--- 🔄 QUERY REWRITE ---")

    question = state["question"]
    loop_count = state.get("loop_count", 0)

    # Prompt kept in French for domain-specific vocabulary optimization
    system_prompt = """
Tu es un expert retrieval assurance.

Transforme la question utilisateur en requête vectorielle courte.

RÈGLES :
- uniquement mots-clés
- pas de phrases
- pas d'explications
- pas de bullet points
- maximum 12 mots
- garder uniquement concepts assurance importants
- Si la question demande un résumé ou contenu global d’un document,
fais une synthèse concise des informations présentes dans les extraits.

Exemple :
Question:
"Mon vélo est-il couvert s'il est volé dans ma cave ?"

Réponse:
vol vélo cave effraction local fermé garantie vol
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        new_question = chain.invoke({
            "question": question
        }).strip()
    except Exception:
        new_question = question

    # Fallback to original query if rewrite is too short
    if len(new_question) < 5:
        new_question = question
        
    if ".pdf" in question.lower():
        return {
            "question": question,
            "loop_count": loop_count + 1
        }    

    print(f"👉 Rewritten query: {new_question}")

    return {
        "question": new_question,
        "loop_count": loop_count + 1
    }

# =========================================================
# GENERATION NODE
# =========================================================
def generate_node(state):
    """Synthesizes the final answer using retrieved, graded documents."""
    print("--- ✍️ GENERATING RESPONSE ---")
    question = state["question"]
    
    # Final context cleaning: select top 3 meaningful documents
    documents = [d for d in state["documents"] if len(d.page_content.strip()) > 120][:3]
    
    if not documents:
        return {"generation": "Information non trouvée dans les documents.", "documents": []}

    context = "\n\n".join([f"[SOURCE: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in documents])
    
    # Prompt kept in French for legal persona adherence
    template = """Tu es un expert juridique assurance. Utilise le contexte pour répondre.
    RÈGLES: Cite les sources, ne pas inventer, si absent dis "Information non trouvée",donne une réponse concise, pas de connaissance externe.
    CONTEXTE: {context}
    QUESTION: {question}
    RÉPONSE:"""
    
    chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
    
    try:
        response = safe_llm_invoke(chain, {"context": context[:12000], "question": question})
    except Exception as e:
        response = "Erreur lors de la génération."
        
    return {"generation": response, "documents": documents}