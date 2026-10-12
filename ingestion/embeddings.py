"""
Embeddings et reranking, avec des modèles locaux (sentence-transformers).

Le même modèle d'embeddings sert à indexer les chunks et à vectoriser les questions — sinon les
vecteurs ne sont pas comparables. vector_store.py vérifie que l'index a été construit avec le
modèle configuré.

Le corpus est en anglais et les questions en français : le modèle doit être multilingue (en V1,
all-MiniLM-L6-v2, entraîné sur de l'anglais, faisait du retrieval cross-lingue sans y être
préparé). Les modèles e5 attendent des préfixes distincts pour les requêtes et les passages.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (EMBEDDING_MODEL_NAME, EMBEDDING_PASSAGE_PREFIX, EMBEDDING_QUERY_PREFIX,
                    RERANKER_MODEL_NAME)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Charge le modèle d'embedding une seule fois (mise en cache)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _encode(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=len(texts) > 20, normalize_embeddings=True)
    return vectors.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Vectorise des passages (chunks à indexer)."""
    return _encode([EMBEDDING_PASSAGE_PREFIX + t for t in texts])


def embed_queries(texts: list[str]) -> list[list[float]]:
    """Vectorise des requêtes (questions)."""
    return _encode([EMBEDDING_QUERY_PREFIX + t for t in texts])


def embed_query(query: str) -> list[float]:
    return embed_queries([query])[0]


def count_truncated(texts: list[str]) -> int:
    """Nombre de passages plus longs que la longueur maximale du modèle (tronqués à l'encodage)."""
    model = get_embedding_model()
    limit = model.max_seq_length
    tok = model.tokenizer
    return sum(len(tok.encode(EMBEDDING_PASSAGE_PREFIX + t)) > limit for t in texts)


@lru_cache(maxsize=1)
def get_reranker():
    """Cross-encoder de reranking, ou None si désactivé (RERANKER_MODEL_NAME vide)."""
    if not RERANKER_MODEL_NAME:
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL_NAME, max_length=512)


def rerank_scores(query: str, passages: list[str]) -> list[float] | None:
    """Scores de pertinence (question, passage) du cross-encoder ; None si pas de reranker."""
    reranker = get_reranker()
    if reranker is None or not passages:
        return None
    return [float(s) for s in reranker.predict([(query, p) for p in passages])]


if __name__ == "__main__":
    # Petit smoke test manuel : une question française doit être plus proche du passage
    # anglais qui y répond que d'un passage hors sujet.
    import numpy as np
    q = embed_query("Comment pousser une valeur dans un XCom ?")
    p = embed_passages(["Use xcom_push to push a value into XCom from a task.",
                        "dbt models are select statements."])
    print(f"Modèle: {EMBEDDING_MODEL_NAME} (dimension {len(q)})")
    print(f"similarités : pertinent={np.dot(q, p[0]):.3f}  hors sujet={np.dot(q, p[1]):.3f}")
