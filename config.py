"""
Configuration centrale du projet. Toutes les valeurs sont surchargeables via variables
d'environnement (voir .env.example) pour rester portable d'une machine à l'autre.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


# --- Ollama ---
# NB: sur certaines machines Windows avec Hyper-V/WSL2, le port par défaut 11434
# est réservé par le système (plage de ports exclue) et Ollama ne peut pas s'y lier.
# Dans ce cas, on démarre Ollama sur un autre port via `OLLAMA_HOST=127.0.0.1:12500 ollama serve`
# et on renseigne la même valeur ici.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
# Fenêtre de contexte demandée explicitement à Ollama : sa valeur par défaut est courte selon
# les versions, et un prompt plus long est tronqué *sans erreur* — le modèle ne voit alors
# qu'une partie des chunks récupérés.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

# --- LLM juge (évaluation Ragas) ---
# Distinct du générateur pour limiter le biais d'auto-évaluation (voir README).
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL", "qwen2.5:7b")

# --- Embeddings ---
# Corpus en anglais, questions en français : il faut un modèle multilingue. Les modèles e5
# attendent les préfixes "query: " / "passage: " (sinon leurs performances chutent).
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
_is_e5 = "e5" in EMBEDDING_MODEL_NAME.lower()
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: " if _is_e5 else "")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: " if _is_e5 else "")

# --- Chunking ---
# Mesuré en tokens du tokenizer d'embeddings. multilingual-e5 tronque au-delà de 512 tokens :
# 400 laisse la place au préfixe et au fil d'Ariane de section.
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# --- Vector store (ChromaDB) ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "dbt_airflow_docs")

# --- Corpus / ingestion artifacts ---
CORPUS_DIR = ROOT_DIR / "corpus"
CHUNKS_PATH = ROOT_DIR / "ingestion" / "chunks.json"

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))
# Recherche vectorielle large, puis reranking cross-encoder et conservation des TOP_K meilleurs.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))
# Cross-encoder multilingue léger ; chaîne vide = pas de reranking.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
# Seuil optionnel sur le score du meilleur chunk (score du reranker s'il est actif, similarité
# cosinus sinon) : en dessous, on refuse sans appeler le LLM. Vide = désactivé. À calibrer avec
# `python evaluation/run_eval.py --retrieval-only` (distributions dans/hors périmètre).
RELEVANCE_THRESHOLD = _optional_float("RELEVANCE_THRESHOLD")

# --- MLflow ---
# NB: MLflow >= 3.x a mis le backend "filesystem" (./mlruns) en mode maintenance et recommande
# un backend SQLite, même en local (voir https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlruns.db'}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-genai-eval")

# --- Observabilité (Langfuse, optionnelle) ---
# Le SDK Langfuse lit lui-même LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY et LANGFUSE_BASE_URL
# (chargées depuis .env par load_dotenv ci-dessus). Sans clés, le traçage est désactivé ;
# LANGFUSE_TRACING_ENABLED=false le coupe sans retirer les clés (run DPO de plusieurs heures...).
LANGFUSE_ENABLED = (bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
                    and os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() != "false")

# --- Entraînement (DPO) ---
TRAINING_DIR = ROOT_DIR / "training"
TRAINING_DATA_DIR = TRAINING_DIR / "data"
