"""
Étape 5 du pipeline : exposition du système via une API REST FastAPI.

Lancer avec : uvicorn api.main:app --reload --port 8000
Documentation Swagger auto-générée sur : http://127.0.0.1:8000/docs
"""
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_PATH, LANGFUSE_ENABLED, OLLAMA_LLM_MODEL, TOP_K
from rag_chain import tracing
from rag_chain.chain import answer_question
from retrieval.vector_store import IndexNotReadyError, ensure_index_ready, index_chunks, index_is_stale

log = logging.getLogger("uvicorn.error")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Au premier démarrage (volume chroma_db vide) ou après un nouveau chunking, l'index n'existe
    # pas ou ne correspond plus à chunks.json : on le (re)construit plutôt que de servir des
    # réponses vides ou issues d'un ancien corpus.
    if CHUNKS_PATH.exists() and index_is_stale():
        log.info("Index ChromaDB absent ou obsolète : indexation de %s ...", CHUNKS_PATH)
        log.info("%d chunks indexés", index_chunks())
    try:
        ensure_index_ready()
    except IndexNotReadyError as exc:
        log.warning("Index non prêt : %s", exc)
    log.info("Traçage Langfuse %s", "actif" if LANGFUSE_ENABLED else "désactivé (pas de clés)")
    yield
    tracing.flush()


app = FastAPI(
    title="Assistant RAG & Évaluation GenAI",
    description="API de questions-réponses sur la documentation dbt et Apache Airflow, "
                "avec recherche vectorielle, reranking et génération augmentée (RAG) 100% locale.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="La question à poser au corpus")
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="Nombre de chunks transmis au LLM")


class Passage(BaseModel):
    source: str
    section: str = ""
    score: float | None = Field(default=None, description="Critère de tri : score du reranker, sinon cosinus")
    similarity: float | None = Field(default=None, description="Similarité cosinus de la recherche vectorielle")
    text: str
    cited: bool = Field(description="Le document de ce passage est cité dans la réponse")


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool = Field(description="Le système a répondu qu'il ne trouvait pas l'information")
    refused_by_threshold: bool = Field(default=False, description="Refus décidé par le seuil, sans appel au LLM")
    cited_sources: list[str] = Field(description="Documents cités dans la réponse")
    retrieved_sources: list[str] = Field(description="Documents dont des passages ont été fournis au LLM")
    passages: list[Passage] = Field(default_factory=list, description="Passages fournis au LLM, dans l'ordre")
    model: str
    latency_seconds: float
    trace_url: str | None = Field(default=None, description="Trace Langfuse de la requête, si le traçage est actif")


@app.get("/", include_in_schema=False)
def ui():
    """Interface web de test (api/static/index.html)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    try:
        ensure_index_ready()
        index_ok = True
    except IndexNotReadyError:
        index_ok = False
    return {"status": "ok" if index_ok else "degraded", "index_ready": index_ok, "model": OLLAMA_LLM_MODEL,
            "tracing": LANGFUSE_ENABLED}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    start = time.time()
    try:
        with tracing.attributes(tags=["api"]):
            result = answer_question(request.question, k=request.top_k)
    except IndexNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de contacter le serveur Ollama local : {exc}",
        ) from exc
    except Exception as exc:  # timeout du modèle, erreur d'inférence, etc.
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {exc}") from exc

    cited = set(result["cited_sources"])
    passages = [Passage(source=h["source"], section=h.get("section", ""), score=h.get("score"),
                        similarity=h.get("similarity"), text=text, cited=h["source"] in cited)
                for h, text in zip(result.get("hits", []), result.get("contexts", []), strict=False)]
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        refused=result["refused"],
        refused_by_threshold=result.get("refused_by_threshold", False),
        cited_sources=result["cited_sources"],
        retrieved_sources=result["retrieved_sources"],
        passages=passages,
        model=result["model"],
        latency_seconds=round(time.time() - start, 2),
        trace_url=tracing.trace_url(result.get("trace_id")),
    )
