"""
Génération des embeddings pour les chunks, avec un modèle local (sentence-transformers).
Réutilisé aussi bien pour indexer les chunks (ingestion) que pour vectoriser une question
au moment de la recherche (retrieval) — même modèle des deux côtés, sinon les vecteurs
ne sont pas comparables.
"""
import sys
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Charge le modèle d'embedding une seule fois (mise en cache)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=len(texts) > 20, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


if __name__ == "__main__":
    # Petit smoke test manuel
    sample = ["Un modèle dbt est une requête SELECT.", "Airflow orchestre des DAGs."]
    vecs = embed_texts(sample)
    print(f"Modèle: {EMBEDDING_MODEL_NAME}")
    print(f"{len(vecs)} vecteurs générés, dimension = {len(vecs[0])}")
