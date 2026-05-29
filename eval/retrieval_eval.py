import numpy as np
import sys
import time
from pathlib import Path

from sqlalchemy import case
import src.bootstrap
from rapidfuzz import fuzz

# --- PATH MANAGEMENT ---
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import ContractRetriever

# --- GOLDEN TEST BENCHMARK SET ---
# Keeping values in French to secure accurate sub-string verification against localized indices

np.random.seed(42)
test_cases = [
    {
        "question": "Quelle est la franchise par défaut pour la garantie 'Dommages collision' ?",
        "expected_source": "cga-assurance-vehicules-fr-2024-3.pdf",
        "expected_text": "franchise s'élève à CHF 500.– par sinistre"
    },
    {
        "question": "Quel est le délai de déclaration pour un sinistre 'Bris de machines' ?",
        "expected_source": "CG-BDM-GA6E21H.pdf",
        "expected_text": "déclarer par écrit tout sinistre dès que vous en avez connaissance et au plus tard dans les 5 jours ouvrés"
    },
    {
        "question": "Quelle loi régit le contrat de garantie de loyer (3A KAUTION) ?",
        "expected_source": "ConditionsGnralesCGA.pdf",
        "expected_text": "dispositions de la loi fédérale sur le contrat d'assurance (LCA)"
    },
    {
        "question": "Quelle est la limite de garantie pour les 'Documents professionnels' en cas de sinistre ?",
        "expected_source": "DG-100-Pro-AC-GA5M66H.pdf",
        "expected_text": "frais de reconstitution sont garantis à hauteur de 15 000 €"
    },
    {
        "question": "Le contrat Auto couvre-t-il les dommages suite à une avalanche ?",
        "expected_source": "cga-assurance-vehicules-fr-2024-3.pdf",
        "expected_text": "Europ Assistance ne peut être tenue pour responsable [...] suite à des événements tels que [...] avalanches"
    },
    {
        "question": "Quels sont les droits d'accès aux données personnelles selon la loi informatique et libertés ?",
        "expected_source": "CG-BDM-GA6E21H.pdf",
        "expected_text": "accéder aux informations vous concernant, les faire rectifier, vous opposer à leur communication"
    },
    {
        "question": "Comment est calculée l'indemnisation pour le matériel informatique professionnel ?",
        "expected_source": "DG-100-Pro-AC-GA5M66H.pdf",
        "expected_text": "indemnité est calculée selon la valeur de remplacement au jour du sinistre"
    }
]

def clean(text):
    """
    Normalizes string chunks for case and whitespace insensitive validation.
    """
    if not text:
        return ""
    return " ".join(text.lower().split())

def evaluate_retrieval(k=5):
    """
    Computes baseline Information Retrieval (IR) performance metrics (Hit Rate, MRR, Latency).
    """
    retriever = ContractRetriever()
    
    # Vectorstore collection metadata integrity check
    try:
        count = retriever.vectorstore.index.ntotal
        print(f"DEBUG: Active vectors in FAISS index: {count}")
    except Exception as e:
        print(f"⚠️ Failed to access FAISS index: {e}")
        return

    hits = []
    reciprocal_ranks = []
    latencies = []

    print(f"🧪 Launching IR metrics evaluation across {len(test_cases)} target cases...")

    for case in test_cases:
        start_time = time.perf_counter()
        results = retriever.search(case["question"], k=k)
        if not results:
            print(f"⚠️ Aucun document retourné pour : {case['question']}")
            hits.append(0)
            reciprocal_ranks.append(0)
            continue
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)

        rank = 0
        found = False
        
        expected_clean = clean(case["expected_text"])
        
        for i, doc in enumerate(results):
            content_clean = clean(doc.page_content)
            
    # Hybrid validation logic: match source file name OR target substring text
            

            
            similarity = fuzz.partial_ratio(expected_clean, content_clean)
            metadata_source = (
            doc.metadata.get("source")
            or doc.metadata.get("file_path")
            or doc.metadata.get("filename")
            or ""
            ).lower()
            source_match = case["expected_source"].lower() in metadata_source
            
            content_match = similarity >= 85  # Threshold empirique à ajuster selon les cas
            
            if source_match or content_match:
                found = True
                rank = i + 1
                break

        hits.append(1 if found else 0)
        reciprocal_ranks.append(1 / rank if found else 0)

    # --- PERFORMANCE METRICS COMPUTATION ---
    avg_hit_rate = np.mean(hits)
    avg_mrr = np.mean(reciprocal_ranks)
    avg_latency = np.mean(latencies) * 1000  # Converted to milliseconds
    p95_latency = np.percentile(latencies, 95) * 1000

    print("\n" + "="*40)
    print(f"📊 RETRIEVAL PIPELINE EVALUATION REPORT (k={k})")
    print("-"*40)
    print(f"✅ (Recall@{k}) : {avg_hit_rate:.2%}")
    print(f"🎯 Mean Reciprocal Rank (MRR) : {avg_mrr:.3f}")
    print(f"⚡ Average Latency            : {avg_latency:.2f} ms")
    print(f"📉 P95 Latency                : {p95_latency:.2f} ms")
    print("="*40)

    if avg_hit_rate < 1.0:
        print("\n🔍 Failure Analysis - Missed Ground Truth Queries:")
        for i, hit in enumerate(hits):
            if hit == 0:
                print(f"❌ Missed Target: '{test_cases[i]['question']}'")
                print("\n--- Retrieved Documents ---")
                for doc in results[:3]:
                    print("\nSOURCE:")
                    print(doc.metadata)
                    print("\nCONTENT:")
                    print(doc.page_content[:500])
                

if __name__ == "__main__":
    # Staging default evaluation depth parameter
    evaluate_retrieval(k=10)