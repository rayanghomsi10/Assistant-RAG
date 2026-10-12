"""Test d'intégration de l'index ChromaDB avec des embeddings factices (sans modèle)."""
import json
import re
import zlib

import numpy as np
import pytest

pytest.importorskip("chromadb")

from retrieval import vector_store as vs  # noqa: E402


def fake_embed(texts):
    vecs = []
    for t in texts:
        v = np.zeros(64)
        for w in re.findall(r"\w+", t.lower()):
            v[zlib.crc32(w.encode()) % 64] += 1
        vecs.append((v / (np.linalg.norm(v) or 1)).tolist())
    return vecs


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(vs, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(vs, "embed_passages", fake_embed)
    monkeypatch.setattr(vs, "embed_query", lambda q: fake_embed([q])[0])
    monkeypatch.setattr(vs, "count_truncated", lambda texts: 0)
    monkeypatch.setattr(vs, "rerank_scores", lambda q, passages: None)
    vs.get_client.cache_clear()
    chunks = [
        {"id": "airflow_04_xcoms.md::chunk_000", "source": "airflow_04_xcoms.md", "chunk_index": 0,
         "section": "XComs", "url": "u", "text": "xcom push pull tasks communicate", "n_tokens": 5},
        {"id": "dbt_01_sql_models.md::chunk_000", "source": "dbt_01_sql_models.md", "chunk_index": 0,
         "section": "SQL models", "url": "u", "text": "sql model select statement", "n_tokens": 4},
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps({"params": {"corpus_sha": "abc", "chunk_size_tokens": 400}, "chunks": chunks}))
    vs.index_chunks(path)
    yield path
    vs.get_client.cache_clear()


def test_index_metadata_and_search(index):
    meta = vs.get_index_metadata()
    assert meta["embedding_model"] == vs.EMBEDDING_MODEL_NAME
    assert meta["corpus_sha"] == "abc" and meta["n_chunks"] == 2
    hits = vs.search("xcom push", k=1)
    assert hits[0]["source"] == "airflow_04_xcoms.md" and hits[0]["section"] == "XComs"
    assert hits[0]["score"] == hits[0]["similarity"]
    assert not vs.index_is_stale(index)


def test_stale_index_detected(index, monkeypatch):
    data = json.loads(index.read_text())
    data["params"]["corpus_sha"] = "changed"
    index.write_text(json.dumps(data))
    assert vs.index_is_stale(index)
    monkeypatch.setattr(vs, "EMBEDDING_MODEL_NAME", "other-model")
    with pytest.raises(vs.IndexNotReadyError):
        vs.ensure_index_ready()


def test_reranker_reorders(index, monkeypatch):
    monkeypatch.setattr(vs, "rerank_scores", lambda q, passages: [10.0 if "sql" in p else 0.0 for p in passages])
    monkeypatch.setattr(vs, "RERANKER_MODEL_NAME", "fake")
    hits = vs.search("xcom push", k=2)
    assert hits[0]["source"] == "dbt_01_sql_models.md" and hits[0]["score"] == 10.0
