"""
Étape 3 du pipeline : base vectorielle (ChromaDB, en local, sans Docker) + reranking.

Indexe les chunks générés par ingestion/chunking.py et expose la recherche :
  1. recherche vectorielle large (RETRIEVAL_CANDIDATES chunks, similarité cosinus, HNSW) ;
  2. reranking par cross-encoder multilingue, conservation des TOP_K meilleurs.

La configuration qui a produit l'index (modèle d'embeddings, préfixes, paramètres de chunking,
empreinte du corpus) est enregistrée dans les métadonnées de la collection. Elle est vérifiée à
chaque recherche (un index construit avec un autre modèle d'embeddings donnerait des résultats
absurdes sans aucune erreur) et relue par l'évaluation pour étiqueter les runs MLflow avec la
configuration réellement indexée, et non celle du .env au moment de l'évaluation.

Utilisable seule (sans LLM) pour valider le retrieval avant de brancher la génération.
"""
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_PASSAGE_PREFIX,
    EMBEDDING_QUERY_PREFIX,
    RERANKER_MODEL_NAME,
    RETRIEVAL_CANDIDATES,
    TOP_K,
)
from ingestion.chunking import load_chunks
from ingestion.embeddings import count_truncated, embed_passages, embed_query, rerank_scores

INDEX_META_PREFIX = "index_"


class IndexNotReadyError(RuntimeError):
    """Index absent ou incompatible avec la configuration courante."""


@lru_cache(maxsize=1)
def get_client():
    import chromadb  # import local : les tests unitaires (CI) n'ont pas besoin de ChromaDB
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection():
    return get_client().get_or_create_collection(
        name=CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
    )


def get_index_metadata() -> dict:
    """Configuration enregistrée lors de l'indexation (clés sans le préfixe interne)."""
    meta = get_collection().metadata or {}
    return {k[len(INDEX_META_PREFIX):]: v for k, v in meta.items() if k.startswith(INDEX_META_PREFIX)}


def index_chunks(chunks_path: Path = CHUNKS_PATH) -> int:
    """Charge les chunks, les vectorise et reconstruit la collection de zéro."""
    params, chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"Aucun chunk trouvé dans {chunks_path}")

    texts = [c["text"] for c in chunks]
    print(f"Génération des embeddings pour {len(texts)} chunks ({EMBEDDING_MODEL_NAME}) ...")
    vectors = embed_passages(texts)
    truncated = count_truncated(texts)
    if truncated:
        print(f"  !! {truncated} chunk(s) dépassent la longueur max du modèle et seront tronqués "
              f"— réduire CHUNK_SIZE_TOKENS")

    meta = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "query_prefix": EMBEDDING_QUERY_PREFIX,
        "passage_prefix": EMBEDDING_PASSAGE_PREFIX,
        "n_chunks": len(chunks),
        "n_truncated": truncated,
        "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **{k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))},
    }

    # Reconstruction propre : supprimer puis recréer la collection (métadonnées comprises).
    client = get_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:  # collection inexistante
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", **{INDEX_META_PREFIX + k: v for k, v in meta.items()}},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"],
                    "section": c.get("section", ""), "url": c.get("url", "")} for c in chunks],
    )
    return len(chunks)


def ensure_index_ready() -> None:
    """Vérifie que l'index existe et a été construit avec le modèle d'embeddings configuré."""
    collection = get_collection()
    if collection.count() == 0:
        raise IndexNotReadyError(
            "L'index ChromaDB est vide. Lance `python ingestion/chunking.py` puis "
            "`python retrieval/vector_store.py`."
        )
    meta = get_index_metadata()
    indexed_with = meta.get("embedding_model")
    if not indexed_with:
        raise IndexNotReadyError(
            "L'index ne contient pas de métadonnées de configuration (construit par une version "
            "antérieure du projet). Relance `python retrieval/vector_store.py`."
        )
    if indexed_with != EMBEDDING_MODEL_NAME:
        raise IndexNotReadyError(
            f"L'index a été construit avec {indexed_with!r} mais EMBEDDING_MODEL_NAME vaut "
            f"{EMBEDDING_MODEL_NAME!r}. Relance `python retrieval/vector_store.py`."
        )


def index_is_stale(chunks_path: Path = CHUNKS_PATH) -> bool:
    """Vrai si l'index est vide ou ne correspond pas à chunks.json / au modèle configuré."""
    if get_collection().count() == 0:
        return True
    meta = get_index_metadata()
    if meta.get("embedding_model") != EMBEDDING_MODEL_NAME:
        return True
    if chunks_path.exists():
        params, chunks = load_chunks(chunks_path)
        if meta.get("corpus_sha") != params.get("corpus_sha") or meta.get("n_chunks") != len(chunks):
            return True
        if any(meta.get(k) != v for k, v in params.items()):
            return True
    return False


def search(query: str, k: int = TOP_K, candidates: int = RETRIEVAL_CANDIDATES,
           rerank: bool = True) -> list[dict]:
    """Recherche vectorielle puis reranking. Retourne les k meilleurs chunks, triés.

    Chaque résultat : id, text, source, section, url, similarity (cosinus), rerank_score
    (None sans reranker) et score (le critère de tri : rerank_score si dispo, sinon similarity).
    """
    ensure_index_ready()
    collection = get_collection()
    n = max(k, candidates if rerank and RERANKER_MODEL_NAME else k)
    results = collection.query(query_embeddings=[embed_query(query)], n_results=n)

    hits = []
    for i in range(len(results["ids"][0])):
        md = results["metadatas"][0][i]
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": md["source"],
            "section": md.get("section", ""),
            "url": md.get("url", ""),
            "similarity": 1.0 - results["distances"][0][i],
            "rerank_score": None,
        })

    scores = rerank_scores(query, [h["text"] for h in hits]) if rerank else None
    if scores is not None:
        for h, s in zip(hits, scores, strict=True):
            h["rerank_score"] = s
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    for h in hits:
        h["score"] = h["rerank_score"] if h["rerank_score"] is not None else h["similarity"]
    return hits[:k]


# Compatibilité avec l'API de la V1.
similarity_search = search


def main():
    n = index_chunks()
    print(f"{n} chunks indexés dans ChromaDB (collection '{CHROMA_COLLECTION_NAME}', dossier {CHROMA_PERSIST_DIR})")
    print(f"Configuration enregistrée avec l'index : {get_index_metadata()}")

    # Validation manuelle du retrieval seul, avant toute intervention du LLM. La validation
    # chiffrée se fait avec `python evaluation/run_eval.py --retrieval-only`.
    test_queries = [
        "À quoi sert Jinja dans un projet dbt ?",
        "Qu'est-ce qu'un XCom dans Airflow ?",
        "Quelle est la différence entre view et table comme matérialisation dbt ?",
        # Témoin hors corpus : les scores doivent être nettement plus bas.
        "Quel est le chiffre d'affaires de TotalEnergies en 2024 ?",
    ]
    print("\n--- Vérification manuelle du retrieval (sans LLM) ---")
    for q in test_queries:
        print(f"\nQ: {q}")
        for hit in search(q, k=3):
            rr = f" rerank={hit['rerank_score']:.2f}" if hit["rerank_score"] is not None else ""
            print(f"  [cos={hit['similarity']:.3f}{rr}] {hit['section'][:70]}")


if __name__ == "__main__":
    main()
