"""
Configuration centrale du projet. Toutes les valeurs sont surchargeables via variables
d'environnement (voir .env.example) pour rester portable d'une machine à l'autre.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent

# --- Ollama ---
# NB: sur certaines machines Windows avec Hyper-V/WSL2, le port par défaut 11434
# est réservé par le système (plage de ports exclue) et Ollama ne peut pas s'y lier.
# Dans ce cas, on démarre Ollama sur un autre port via `OLLAMA_HOST=127.0.0.1:12500 ollama serve`
# et on renseigne la même valeur ici.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")

# --- Embeddings ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# --- Vector store (ChromaDB) ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "dbt_airflow_docs")

# --- Corpus / ingestion artifacts ---
CORPUS_DIR = ROOT_DIR / "corpus"
CHUNKS_PATH = ROOT_DIR / "ingestion" / "chunks.json"

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))

# --- MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-genai-eval")
