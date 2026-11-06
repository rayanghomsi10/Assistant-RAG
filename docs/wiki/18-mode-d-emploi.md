# 18. Mode d'emploi

## 18.1 Installation

```
python -m venv venv
venv\Scripts\activate                 # Windows
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

Si le port 11434 est réservé par Windows (Hyper-V, WSL2) : `OLLAMA_HOST=127.0.0.1:12500 ollama serve`, et reporter le port dans `OLLAMA_BASE_URL`. Vérifier les plages réservées avec `netsh interface ipv4 show excludedportrange protocol=tcp`.

Pour vérifier le code : `pip install -r requirements-dev.txt`, puis `ruff check .` et `pytest -q`.

## 18.2 Construire et interroger

```
python ingestion/build_corpus.py      # optionnel : le corpus est versionné
python ingestion/chunking.py
python retrieval/vector_store.py      # indexation + 4 questions témoins
python rag_chain/chain.py             # deux questions de démonstration
uvicorn api.main:app --reload --port 8000   # Swagger : http://127.0.0.1:8000/docs
                                            # interface : http://127.0.0.1:8000
docker compose --project-directory langfuse up -d   # traces : http://localhost:3000
```

Avec Docker : `docker compose up -d --build`, puis `curl http://127.0.0.1:8600/health`.

## 18.3 Évaluer

```
python evaluation/dataset.py                        # valider le jeu
python evaluation/run_eval.py --retrieval-only
python evaluation/run_eval.py --run-name baseline_v2 --repeats 2
python evaluation/compare_runs.py baseline_v2 autre_run
mlflow ui --backend-store-uri sqlite:///mlruns.db   # http://127.0.0.1:5000
python evaluation/run_conversation_eval.py --run-name memoire_v1   # relances
```

## 18.4 Entraîner par DPO

```
python training/generate_questions.py --per-chunk 3
python training/build_preferences.py --samples 4 --limit 20   # essai
python training/build_preferences.py --samples 4              # complet ; relancé, il reprend
python training/sample_pairs.py --n 30                        # paires à relire avant d'entraîner
# Colab / Kaggle : training/dpo_colab.ipynb (--limit 20, puis complet) → dpo-lora.zip
python training/export_ollama.py --adapter training/outputs/dpo-lora
ollama create rag-assistant-dpo -f training/outputs/ollama/Modelfile
python evaluation/run_eval.py --run-name dpo_v1 --model rag-assistant-dpo --repeats 2
python evaluation/compare_runs.py baseline_v2 dpo_v1
```

Le détail de chaque étape, les temps observés et la lecture des statistiques sont dans `training/README.md`.

## 18.5 Documentation

```
python docs/build_pdf.py              # régénère docs/Assistant-RAG_documentation_cours.pdf depuis le wiki
```

Le wiki (`docs/wiki/`) est la source ; le PDF en est une sortie. Les conventions sont dans `docs/wiki/SCHEMA.md`.
