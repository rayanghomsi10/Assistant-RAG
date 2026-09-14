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

Le corpus est reproductible : `ingestion/build_corpus.py` télécharge les pages, en extrait le
contenu principal (navigation, pieds de page et artefacts de mise en page retirés) et génère les
PDF. Les PDF produits contiennent bien une couche de texte extractible (aucun scan), condition
nécessaire à l'ingestion.

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
# 0. (optionnel) Reconstruire le corpus depuis les docs en ligne — les PDF sont déjà versionnés
python ingestion/build_corpus.py

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
`evaluation/results/baseline_chunk500_overlap50_v2_fixed_dataset.csv` pour le détail par question.)*

**Lecture des résultats** : le retrieval est excellent (context precision quasi parfaite — les
bons chunks remontent bien en tête), ce qui valide le choix de chunking et d'embeddings. La
faithfulness est bonne (0.82) : les réponses inspectées manuellement dans le CSV de résultats
sont bien ancrées dans le contexte récupéré, y compris sur des réponses
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

Résultats détaillés par question : `evaluation/results/<nom_du_run>.csv` (un fichier par
expérience). Historique des runs et comparaison : `mlflow ui --backend-store-uri sqlite:///mlruns.db`.

### Itération : effet de la taille de chunk

Deuxième configuration testée pour mesurer l'impact du chunking, à corpus, prompt, modèle et jeu
de test identiques — seuls la taille de chunk et le chevauchement changent :

| Config | Chunks indexés | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|---|
| **500 tokens / overlap 50** (retenue) | 126 | **0.818** | 0.581 | **0.987** |
| 300 tokens / overlap 30 | 221 | 0.748 | **0.629** | 0.963 |

**Conclusion** : des chunks plus petits améliorent légèrement l'answer relevancy (+0.05 — des
passages plus courts et plus ciblés donnent des réponses plus directes), mais dégradent la
faithfulness (−0.07) et la context precision (−0.02). L'explication la plus probable : en
découpant à 300 tokens, un passage explicatif est plus souvent coupé en deux, le modèle reçoit
des fragments incomplets et comble les trous — exactement ce que la faithfulness pénalise. La
configuration **500/50 est donc retenue** comme référence, la fidélité au contexte primant sur
la formulation dans un système RAG dont l'intérêt est précisément l'ancrage dans les sources.

## Limites connues

- **Chunking simple** : découpage par taille fixe (500 tokens / overlap 50), pas de chunking
  sémantique ou structurel. C'est un choix volontaire de V1. La context precision étant déjà à
  0.987, le gain restant n'est pas côté retrieval mais côté génération : un chunking respectant
  les frontières de sections (plutôt qu'une coupe à taille fixe au milieu d'une explication)
  viserait surtout à faire monter la faithfulness — c'est ce que suggère l'itération 300/30
  ci-dessus, où des coupes plus fréquentes l'ont dégradée.
- **Corpus restreint** : 10 documents, un seul domaine (dbt/Airflow). Le système répond "je ne
  trouve pas cette information" en dehors de ce périmètre par construction du prompt.
- **Citations reformulées** : sur certaines réponses, le modèle présente entre guillemets un
  passage qui n'est pas verbatim dans le contexte — le fond reste correct et ancré, mais la
  citation est reformulée. C'est une des causes concrètes d'une faithfulness à 0.82 plutôt que
  plus haute. Piste d'amélioration : interdire explicitement les guillemets dans le prompt
  système, ou n'autoriser que des citations vérifiées par correspondance exacte avec le contexte.
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
- **Latence** : mesurée entre ~2s et ~12s par requête selon la longueur de la réponse, une fois
  le modèle chargé en VRAM (comptez ~40-70s au tout premier appel, le temps du chargement) sur un
  GPU 8 Go. Sans GPU, la latence serait nettement plus élevée.
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
├── config.py                   # configuration centrale (surchargeable par variables d'env)
├── .env.example                # modèle de configuration locale
├── corpus/                     # 10 PDF (documentation dbt + Airflow)
├── ingestion/
│   ├── chunking.py             # PDF -> chunks (taille fixe + overlap)
│   ├── embeddings.py           # sentence-transformers (local)
│   └── chunks.json             # artefact d'ingestion (généré par chunking.py)
├── retrieval/
│   └── vector_store.py         # ChromaDB : indexation + recherche par similarité
├── rag_chain/
│   └── chain.py                # LangChain LCEL : retriever -> prompt -> LLM
├── api/
│   └── main.py                 # FastAPI (POST /query, GET /health)
├── evaluation/
│   ├── test_dataset.json       # 15 questions/réponses de référence
│   ├── run_ragas_eval.py       # boucle d'évaluation Ragas
│   ├── mlflow_tracking.py      # log des runs dans MLflow
│   ├── _ragas_compat.py        # correctif d'import ragas/langchain-community
│   └── results/                # résultats détaillés, un CSV par run
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
└── requirements.txt
```

Non versionnés (voir `.gitignore`) : `venv/`, `chroma_db/` (index vectoriel, reconstructible via
`retrieval/vector_store.py`), `mlruns.db` (historique MLflow local), `.env`, `corpus/raw_text/`.
