"""
Étape 6 de la feuille de route : exposition via une API FastAPI.

Lancer avec : uvicorn api.main:app --reload --port 8000
Documentation Swagger auto-générée sur : http://127.0.0.1:8000/docs
"""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K
from rag_chain.chain import answer_question

app = FastAPI(
    title="Assistant RAG & Évaluation GenAI",
    description="API de questions-réponses sur la documentation dbt et Apache Airflow, "
                "avec recherche vectorielle et génération augmentée (RAG) 100% locale.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="La question à poser au corpus")
    top_k: int = Field(default=TOP_K, ge=1, le=20, description="Nombre de chunks à récupérer")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    start = time.time()
    try:
        result = answer_question(request.question, k=request.top_k)
    except ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de contacter le serveur Ollama local : {exc}",
        )
    except Exception as exc:  # timeout du modèle, erreur d'inférence, etc.
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {exc}")

    latency = time.time() - start
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        latency_seconds=round(latency, 2),
    )
