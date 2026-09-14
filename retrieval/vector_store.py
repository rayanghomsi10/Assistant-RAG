"""
Étape 4 de la feuille de route : base vectorielle (ChromaDB, en local, sans Docker).

Indexe les chunks générés par ingestion/chunking.py et expose une fonction de recherche
par similarité, utilisable seule (sans LLM) pour valider le retrieval avant de brancher
la génération — c'est le piège à éviter n°1 selon le guide.
"""
import json
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, CHUNKS_PATH, TOP_K
from ingestion.embeddings import embed_texts, embed_query


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # ANN: HNSW, similarité cosinus
    )


def index_chunks(chunks_path: Path = CHUNKS_PATH) -> int:
    """Charge les chunks depuis le JSON produit par chunking.py, les vectorise et les indexe."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"Aucun chunk trouvé dans {chunks_path}")

    collection = get_collection()

    # Ré-indexation propre : on repart d'une collection vide pour éviter les doublons
    # si ce script est relancé après un changement de chunking.
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    print(f"Génération des embeddings pour {len(texts)} chunks ...")
    vectors = embed_texts(texts)

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    return len(chunks)


def similarity_search(query: str, k: int = TOP_K) -> list[dict]:
    """Recherche brute par similarité (sans LLM). Retourne les k chunks les plus proches."""
    collection = get_collection()
    query_vector = embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return hits


def main():
    n = index_chunks()
    print(f"{n} chunks indexés dans ChromaDB (collection '{CHROMA_COLLECTION_NAME}', dossier {CHROMA_PERSIST_DIR})")

    # Questions test manuelles pour valider le retrieval seul, comme demandé par le guide
    # avant de brancher le LLM.
    test_queries = [
        "Comment fonctionne le chevauchement (overlap) dans le chunking dbt ?",
        "Qu'est-ce qu'un XCom dans Airflow ?",
        "Quelle est la différence entre view et table comme matérialisation dbt ?",
    ]
    print("\n--- Vérification manuelle du retrieval (sans LLM) ---")
    for q in test_queries:
        print(f"\nQ: {q}")
        for hit in similarity_search(q, k=3):
            preview = hit["text"][:100].replace("\n", " ")
            print(f"  [{hit['distance']:.3f}] {hit['source']} :: {preview}...")


if __name__ == "__main__":
    main()
