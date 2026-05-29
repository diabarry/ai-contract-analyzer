import time
import src.bootstrap
import gc
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset

# 1. Path management to locate the 'src' directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 2. Ragas Ecosystem Imports (Compliant with v0.4+ specifications)
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI 
from langchain_core.rate_limiters import InMemoryRateLimiter
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# 3. Dynamic Agent Import
from src.agent import app

load_dotenv()

# Environment integrity check
mistral_key = os.getenv("MISTRAL_API_KEY")

if not mistral_key:
    raise ValueError("Error: MISTRAL_API_KEY not found in environment. Please check your .env file!")

import nest_asyncio
nest_asyncio.apply()

# Strip whitespace and quotes for robust API key handling
mistral_key = mistral_key.strip().strip('"').strip("'")
os.environ["MISTRAL_API_KEY"] = mistral_key

# Debug info
print(f" Mistral key loaded (Length: {len(mistral_key)} chars, Prefix: {mistral_key[:5]}...)")

# --- GOLDEN EVALUATION DATASET ---
# A curated set of ground truth data for assessing RAG performance

test_questions = [

    # ========================================================
    # DOCUMENT 1 : LEGAL PROTECTION (FORTUNA)
    # ========================================================
    {
        "question": "Quel est le délai dont je dispose pour révoquer mon contrat d'assurance de protection juridique après réception de la police ?",
        "ground_truth": "Le preneur d'assurance a le droit de se retirer du contrat d'assurance par écrit ou sous forme textuelle dans les 14 jours à compter de la réception de la police."
    },

    {
        "question": "Quel est le montant maximal des prestations versées par Fortuna pour un litige avec la variante de produit Top ?",
        "ground_truth": "Dans la variante de produit Top, Fortuna verse des prestations à concurrence d'un montant total maximal de CHF 1 000 000.- par litige."
    },

    {
        "question": "La protection juridique circulation intervient-elle si le conducteur conduit avec une alcoolémie de 1,6 ‰ ?",
        "ground_truth": "Il n'y a pas de couverture d'assurance si, au moment de la survenance du cas juridique, le conducteur présente une concentration d'alcool dans le sang de 1,5 ‰ ou de 0,75 mg/l ou plus."
    },

    # ========================================================
    # DOCUMENT 2 : 100% PRO ARTISANS-COMMERÇANTS" (GENERALI)
    # ========================================================

    {
        "question": "Comment sont définis les frais de relogement pour un professionnel dont les locaux sont inutilisables suite à un sinistre ?",
        "ground_truth": "Il s'agit du surcoût éventuel assumé par l'Assuré pour la location de locaux afin de maintenir l'activité professionnelle, lorsqu'à la suite d'un sinistre, les locaux professionnels assurés ne peuvent pas être occupés pendant le temps de la remise en état."
    },

    {
        "question": "Quel est le montant minimum de la franchise applicable pour les biens à usage professionnel en cas de catastrophe naturelle ?",
        "ground_truth": "La franchise est égale à 10 % du montant des dommages matériels directs non assurables subis par l'Assuré, sans pouvoir être inférieure à 1 140 euros (sauf pour les mouvements de terrain différentiels liés à la sécheresse où le minimum est fixé à 3 050 euros)."
    },

    {
        "question": "Quel est le plafond de garantie pour la disparition des espèces, fonds et valeurs s'ils sont placés dans un coffre-fort verrouillé ?",
        "ground_truth": "En cas d'effraction des caisses ou coffres les contenant, le montant maximum de garantie pour les espèces, fonds et valeurs en coffre est fixé aux Dispositions Particulières."
    },

    # ========================================================
    # DOCUMENT 3 : GENERALI AUTO
    # ========================================================

    {
        "question": "Ma responsabilité civile automobile est-elle couverte si j'utilise ma voiture pour remorquer bénévolement le véhicule en panne d'un ami ?",
        "ground_truth": "Oui, par extension à la garantie responsabilité civile obligatoire, la Responsabilité Civile est garantie s'il remorque bénévolement un autre véhicule en panne. Les dommages subis par le véhicule tracteur et/ou remorqué sont exclus."
    },

    {
        "question": "L'assurance auto couvre-t-elle le vol de mon véhicule si j'ai laissé les clés sur le contact alors qu'il était garé dans la rue ?",
        "ground_truth": "Non, l'Assureur ne garantit pas les vols commis alors que le véhicule se trouvait hors d'un garage individuel clos, alors que les clés de contact ou de fermeture se trouvaient à l'intérieur ou sur le véhicule."
    },

    {
        "question": "Quel est le montant de l'indemnité forfaitaire journalière en cas d'immobilisation de mon véhicule garanti au titre de l'option Rupture d'activité ?",
        "ground_truth": "Le montant de l'indemnité forfaitaire d'immobilisation est de 150 € par jour d'immobilisation, applicable après un délai de carence de 3 jours."
    },

    # ========================================================
    # DOCUMENT 4 : GENERALI RESIDENCE
    # ========================================================

    {
        "question": "Quelle mesure de prévention liée à l'eau dois-je prendre si je m'absente de ma résidence principale pendant plus de 5 jours en hiver ?",
        "ground_truth": "Du 1er novembre au 31 mars, en cas d'inoccupation des locaux supérieure à 5 jours consécutifs, vous devez interrompre la circulation d'eau dans toutes les conduites par la fermeture du robinet d'arrêt général, sauf en cas d'impossibilité technique ou de locaux mis hors gel."
    },

    {
        "question": "À concurrence de quel montant maximal l'assistance prend-elle en charge l'intervention d'un serrurier si je perds les clés de mon domicile ?",
        "ground_truth": "Generali Assistance recherche un serrurier, le dépêche au Domicile et prend en charge ses frais d'intervention à concurrence de 150 euros TTC. Le coût des réparations reste à la charge de l'Assuré."
    },

    {
        "question": "Si ma maison est totalement inhabitable suite à un incendie, dans quelles limites l'assistance Generali prend-elle en charge mes frais d'hôtel ?",
        "ground_truth": "L'assistance prend en charge les frais d'hébergement (chambre d'hôtel et petit-déjeuner), à concurrence de 60 euros TTC par nuit et par Bénéficiaire, pendant 10 nuits consécutives maximum."
    },

    {
        "question": "L'assurance scolaire couvre-t-elle mon enfant s'il se blesse au cours d'une bagarre à l'école ?",
        "ground_truth": "Non, sont exclus les accidents survenus au cours de la participation à une rixe ou une bagarre, sauf cas de légitime défense."
    }

]

def generate_rag_responses(questions):
    """Generates agent responses and retrieves context for the evaluation dataset."""
    print(f"🤖 Agent online: Generating evaluation payloads for {len(questions)} cases...")
    results = []

    for idx, item in enumerate(questions):
        try:
            print(f"\n🧪 Test case {idx + 1}/{len(questions)}")
            print(f"❓ Question: {item['question']}")

            # Invoke the agent graph
            response = app.invoke({
                "question": item["question"],
                "loop_count": 0,
                "documents": []
            })

            if not isinstance(response, dict):
                print("⚠️ Invalid LangGraph response")
                continue

            answer = str(response.get("generation", "No response generated")).strip()
            docs = response.get("documents", [])

            # Extract context from retrieved documents
            contexts = []
            for doc in docs:
                try:
                    if hasattr(doc, "page_content"):
                        contexts.append(str(doc.page_content)[:2000])
                except Exception:
                    continue

            if not contexts:
                contexts = ["No context retrieved"]
            
            results.append({
                "question": str(item["question"]),
                "answer": answer,
                "contexts": contexts,
                "ground_truth": str(item["ground_truth"])
            })
            time.sleep(2)
            print("✅ Sample added")

        except Exception as e:
            print(f"Error generating sample: {e}")

    print(f"\n📦 Samples generated: {len(results)}")
    return results

def run_full_evaluation():
    """Compiles the dataset and executes the Ragas evaluation suite."""
    samples = generate_rag_responses(test_questions)
    
    if not samples:
       raise ValueError("Error: No valid samples generated by the RAG pipeline.")

    # Data cleaning with Pandas
    df_clean = pd.DataFrame(samples)
    df_clean['answer'] = df_clean['answer'].fillna("I don't know.")
    
    # Structure contexts as a list of strings
    df_clean['contexts'] = df_clean['contexts'].apply(
        lambda x: [str(doc) for doc in x] if isinstance(x, list) else []
    )

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        "question": df_clean["question"].tolist(),
        "answer": df_clean["answer"].tolist(),
        "contexts": df_clean["contexts"].tolist(),
        "ground_truth": df_clean["ground_truth"].tolist()
    })
    
    # 4. Initialize Ragas Evaluation Components
    embedding_model = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    )
    
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="mistral-small-latest",
            openai_api_key=mistral_key,
            openai_api_base="https://api.mistral.ai/v1",
            temperature=0,
            max_retries=10,
            request_timeout=60
        )
    )
    
    # 5. Define Evaluation Config
    run_config = RunConfig(
        timeout=420,
        max_retries=10, 
        max_wait=60,
        max_workers=1 # Sequential execution for API stability
    )