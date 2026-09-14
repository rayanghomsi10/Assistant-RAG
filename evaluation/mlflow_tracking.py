"""
Suivi d'expérimentations avec MLflow : logue chaque run (config de chunking, modèle
utilisé, scores Ragas obtenus) pour comparer objectivement les versions du pipeline.
"""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, EMBEDDING_MODEL_NAME,
    OLLAMA_LLM_MODEL, TOP_K,
)


def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_eval_run(scores: dict, run_name: str | None = None, extra_params: dict | None = None):
    """Logue une run d'évaluation : paramètres du pipeline + scores Ragas moyens.

    scores: dict de métriques Ragas -> valeur moyenne (ex. {"faithfulness": 0.87, ...})
    """
    init_mlflow()
    with mlflow.start_run(run_name=run_name):
        params = {
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "llm_model": OLLAMA_LLM_MODEL,
            "top_k": TOP_K,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in scores.items() if v is not None})
        print(f"Run MLflow loguée. Lance `mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}` pour visualiser.")
