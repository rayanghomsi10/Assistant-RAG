# Assistant RAG & Évaluation d'IA Générative

Système de questions-réponses (RAG) sur la documentation officielle **dbt** et **Apache Airflow**,
avec une boucle d'évaluation mesurée (Ragas) et un suivi d'expérimentations (MLflow).
Stack 100% gratuite et locale : aucun appel à une API payante, tout tourne sur la machine
(LLM via Ollama, embeddings via sentence-transformers, base vectorielle ChromaDB).

## Pourquoi ce projet

Un LLM seul hallucine et a une connaissance figée à sa date d'entraînement. Le RAG résout ça en
injectant, au moment de la requête, des passages de documents pertinents dans le prompt — le
modèle génère alors une réponse ancrée dans une source vérifiable. Ce projet ne se contente pas
de faire fonctionner un chatbot : il mesure objectivement la qualité des réponses générées
(fidélité au contexte, pertinence, précision du retrieval) avec le framework Ragas, et suit ces
scores dans le temps avec MLflow pour comparer les versions du pipeline.

## Architecture

```
                         ┌─────────────────────┐
   corpus/*.pdf  ───────▶│  ingestion/          │
  (dbt + Airflow docs)   │  chunking.py         │  chunks (~500 tokens, overlap 50)
                         │  embeddings.py        │  vecteurs (sentence-transformers, local)
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  retrieval/          │
                         │  vector_store.py      │  ChromaDB (persistent, local, HNSW)
                         └──────────┬───────────┘
                                    ▼
        question ──────▶ ┌─────────────────────┐
                         │  rag_chain/chain.py   │  retriever → prompt → Ollama (llama3.1:8b) → réponse
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐        ┌───────────────────────────┐
                         │  api/main.py          │        │  evaluation/               │
                         │  FastAPI POST /query  │        │  run_ragas_eval.py          │──▶ MLflow
                         └─────────────────────┘        │  (juge = Ollama local)      │
                                                          └───────────────────────────┘
```

## Stack technique

| Brique | Outil | Notes |
|---|---|---|
| Modèle de langage | Ollama + Llama 3.1 8B (Q4_K_M) | Génération, exécuté en local sur GPU |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Vectorisation des chunks et des questions |
| Base vectorielle | ChromaDB (persistent client, sans Docker) | Recherche par similarité cosinus (HNSW) |
| Orchestration RAG | LangChain (LCEL) | retriever → prompt template → LLM |
| API | FastAPI | `POST /query`, doc Swagger auto-générée |
| Évaluation | Ragas | faithfulness, answer_relevancy, context_precision — juge = Ollama local |
| Suivi d'expérimentations | MLflow | historique des scores entre versions du pipeline |
| Conteneurisation | Docker | API conteneurisée, ChromaDB en volume local |

## Corpus

10 documents PDF issus de la documentation officielle **dbt** (modèles SQL, Jinja/macros,
data tests, modèles incrémentaux, matérialisations) et **Apache Airflow** (DAGs, opérateurs,
scheduler, XComs, bonnes pratiques) — un seul domaine cohérent (outillage data engineering),
comme recommandé pour une V1. Voir `corpus/*.pdf`.

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

cp .env.example .env           # puis ajuster si besoin (voir note ci-dessous sur le port Ollama)
```

Installer [Ollama](https://ollama.com) et récupérer le modèle :
```bash
ollama pull llama3.1:8b
```

> **Note Windows / Hyper-V** : sur une machine où Hyper-V ou WSL2 est actif, le port par défaut
> d'Ollama (11434) peut être réservé par le système (`netsh interface ipv4 show
> excludedportrange protocol=tcp`), provoquant une erreur `bind: An attempt was made to access a
> socket in a way forbidden by its access permissions`. Dans ce cas, démarrer Ollama sur un autre
> port : `OLLAMA_HOST=127.0.0.1:12500 ollama serve`, et renseigner `OLLAMA_BASE_URL` (et
> `OLLAMA_BASE_URL_DOCKER` pour Docker) en conséquence dans `.env`. Le même problème peut
> survenir avec le port 8000 de l'API — utiliser un autre port (`uvicorn api.main:app --port
> 8600`, ou `HOST_PORT` dans `.env` pour Docker) le cas échéant. Vérifier les plages réservées
> avec `netsh interface ipv4 show excludedportrange protocol=tcp`.

## Utilisation

```bash
# 1. Ingestion : PDF -> chunks
python ingestion/chunking.py

# 2. Indexation dans ChromaDB + vérification manuelle du retrieval (sans LLM)
python retrieval/vector_store.py

# 3. Test de la chaîne RAG complète en ligne de commande
python rag_chain/chain.py

# 4. Lancer l'API
uvicorn api.main:app --reload --port 8000
# Swagger : http://127.0.0.1:8000/docs

# 5. Boucle d'évaluation (Ragas + MLflow)
python evaluation/run_ragas_eval.py --run-name "ma_config"
mlflow ui --backend-store-uri sqlite:///mlruns.db   # puis ouvrir http://127.0.0.1:5000
```

### Avec Docker

```bash
docker compose up -d --build
curl http://127.0.0.1:8600/health
```
L'API tourne dans le conteneur (port hôte 8600 par défaut, voir `HOST_PORT` dans `.env`),
ChromaDB persiste dans `./chroma_db` (monté en volume), et le conteneur contacte Ollama sur la
machine hôte via `OLLAMA_BASE_URL_DOCKER` (voir `.env.example`).

Exemple d'appel API :
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment pousser une valeur dans un XCom ?"}'
```

## Résultats d'évaluation

Configuration de référence : chunking 500 tokens / overlap 50, `all-MiniLM-L6-v2`, `llama3.1:8b`
(Q4_K_M), top-k = 4, jeu de test de 15 questions de référence (`evaluation/test_dataset.json`).

| Métrique | Score moyen | Ce qu'elle mesure |
|---|---|---|
| Faithfulness | **0.818** | La réponse est-elle factuellement cohérente avec les chunks récupérés (pas d'hallucination) ? |
| Answer relevancy | **0.581** | La réponse répond-elle réellement à la question posée ? |
| Context precision | **0.987** | Les chunks les plus pertinents sont-ils bien classés en tête du retrieval ? |

*(run `baseline_chunk500_overlap50_v2_fixed_dataset`, 15/15 questions évaluées sans erreur — voir
`evaluation/last_eval_results.csv` pour le détail par question.)*

**Lecture des résultats** : le retrieval est excellent (context precision quasi parfaite — les
bons chunks remontent bien en tête), ce qui valide le choix de chunking et d'embeddings. La
faithfulness est bonne (0.82) : les réponses inspectées manuellement dans
`last_eval_results.csv` sont bien ancrées dans le contexte récupéré, y compris sur des réponses
factuellement correctes et vérifiées (ex. `xcom_push`, fréquence du scheduler). L'answer
relevancy est en revanche systématiquement plus basse (0.58 en moyenne, parfois 0.22-0.37 sur des
réponses par ailleurs correctes et fidèles à 100%) — en creusant, ce n'est pas la qualité des
réponses qui est en cause mais la métrique elle-même : Ragas calcule l'answer relevancy en
générant des questions hypothétiques à partir de la réponse via le LLM juge, puis compare leur
similarité à la question d'origine. Avec un juge local de 8B de paramètres (le même modèle que le
générateur, faute d'API payante), ces questions générées dérivent souvent de la formulation
d'origine — c'est un biais connu de la métrique quand le juge est un petit modèle local, pas un
défaut du pipeline RAG lui-même. Un vrai cas de refus correct existe aussi dans le jeu de test
(question hors périmètre, corpus financier) : le système répond honnêtement "je ne trouve pas
cette information", ce qui est le comportement voulu, mais fait mécaniquement chuter les deux
métriques à 0 sur cette ligne faute de contenu à évaluer.

Résultats détaillés par question : `evaluation/last_eval_results.csv`. Historique des runs et
comparaison entre configurations : `mlflow ui --backend-store-uri sqlite:///mlruns.db`.

## Limites connues

- **Chunking simple** : découpage par taille fixe (500 tokens / overlap 50), pas de chunking
  sémantique ou structurel. C'est un choix volontaire de V1 (voir feuille de route du projet) —
  premier axe d'itération naturel pour améliorer `context_precision`.
- **Corpus restreint** : 10 documents, un seul domaine (dbt/Airflow). Le système répond "je ne
  trouve pas cette information" en dehors de ce périmètre par construction du prompt.
- **LLM juge = LLM générateur** : Ragas utilise le même modèle (`llama3.1:8b` via Ollama) comme
  juge que comme générateur, faute d'API payante. Un juge plus puissant (ou un juge différent du
  générateur) donnerait des scores plus fiables et moins sujets à un biais d'auto-évaluation —
  on observe concrètement ce biais sur `answer_relevancy` (voir section Résultats).
- **Construction du jeu de test** : une première version d'une question de référence
  (idempotence des tâches Airflow) avait été écrite à partir de connaissances générales sur
  Airflow plutôt que vérifiée contre le corpus réellement récupéré, qui ne couvre pas ce sujet en
  détail. Le système a correctement répondu "je ne trouve pas cette information" (comportement
  honnête, sans hallucination) plutôt que d'inventer une réponse plausible — mais cela a révélé
  une question de référence mal ancrée, corrigée depuis. Ça illustre un piège réel : le jeu de
  questions/réponses de référence doit être vérifié phrase par phrase contre le corpus effectif,
  pas rédigé de mémoire.
- **Concurrence Ragas vs Ollama local** : Ragas lance ses appels au LLM juge en parallèle
  (`max_workers` par défaut = 16), mais un serveur Ollama local sert les requêtes en série
  (`OLLAMA_NUM_PARALLEL=1`) — la configuration par défaut provoque des `TimeoutError` sur une
  partie des questions. Corrigé en réduisant la concurrence et en augmentant le timeout
  (`RunConfig(max_workers=2, timeout=300)` dans `run_ragas_eval.py`).
- **Latence** : ~2-8s par requête une fois le modèle chargé en VRAM (~40-70s au premier appel, le
  temps de charger le modèle) sur un GPU 8 Go. Sans GPU, la latence serait nettement plus élevée.
- **Compatibilité ragas 0.4.3 / langchain-community** : `ragas` importe de façon inconditionnelle
  un module VertexAI que `langchain-community` (en cours de dépréciation) a retiré. Contournement
  documenté et appliqué dans `evaluation/_ragas_compat.py` (aucun impact fonctionnel, le projet
  n'utilise jamais VertexAI).
- **ChromaDB plutôt que Qdrant** : choix fait pour rester à zéro dépendance Docker côté base
  vectorielle. `docker-compose.yml` conteneurise l'API ; ChromaDB persiste en fichiers locaux
  montés en volume.
- **`docker run -v` sous Git Bash (Windows)** : passer des volumes en ligne de commande
  (`docker run -v $(pwd)/...`) sous Git Bash déclenche une conversion de chemin MSYS qui corrompt
  le chemin de destination dans le conteneur. `docker compose up` (les chemins viennent du YAML,
  pas de la ligne de commande) n'a pas ce problème — c'est la méthode recommandée sur cette
  configuration.

## Structure du dépôt

```
rag-genai-eval/
├── README.md
├── config.py                  # configuration centrale (env vars)
├── corpus/                    # 10 PDF (documentation dbt + Airflow)
├── ingestion/
│   ├── chunking.py            # PDF -> chunks (taille fixe + overlap)
│   └── embeddings.py          # sentence-transformers (local)
├── retrieval/
│   └── vector_store.py        # ChromaDB : indexation + recherche par similarité
├── rag_chain/
│   └── chain.py                # LangChain LCEL : retriever -> prompt -> LLM
├── api/
│   └── main.py                 # FastAPI (POST /query, GET /health)
├── evaluation/
│   ├── test_dataset.json       # 15 questions/réponses de référence
│   ├── run_ragas_eval.py       # boucle d'évaluation Ragas
│   ├── mlflow_tracking.py      # log des runs dans MLflow
│   └── last_eval_results.csv   # résultats détaillés du dernier run
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
