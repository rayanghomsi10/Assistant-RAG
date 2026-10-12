FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch en version CPU : les embeddings et le reranker tournent sur CPU dans le conteneur
# (le LLM tourne dans Ollama, sur l'hôte). Évite ~2 Go de bibliothèques CUDA inutiles.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Chunks construits au build (télécharge seulement le tokenizer du modèle d'embeddings).
# L'index ChromaDB est construit au premier démarrage de l'API s'il est absent ou obsolète.
RUN python ingestion/chunking.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
