# Assistant RAG & Évaluation d'IA Générative

[![CI/CD](https://github.com/rayanghomsi10/Assistant-RAG/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rayanghomsi10/Assistant-RAG/actions/workflows/ci-cd.yml)

Système de questions-réponses (RAG) sur la documentation officielle **dbt** et **Apache Airflow**,
avec un protocole d'évaluation fiable (métriques déterministes, Ragas avec juge indépendant,
intervalles de confiance), un suivi d'expérimentations MLflow, et une boucle d'alignement du
générateur par **DPO** (Direct Preference Optimization) sur des préférences vérifiables.

Stack gratuite et locale pour l'inférence : LLM via Ollama, embeddings et reranker via
sentence-transformers, base vectorielle ChromaDB. Seul l'entraînement DPO demande un GPU de
16 Go (Colab ou Kaggle gratuits).

## Pourquoi ce projet

Un LLM seul hallucine et a une connaissance figée. Le RAG injecte au moment de la requête des
passages de documents pertinents dans le prompt, pour ancrer la réponse dans une source
vérifiable. Ce projet ne s'arrête pas au chatbot qui fonctionne : il mesure la qualité des
réponses, distingue ce qui relève du retrieval et de la génération, sépare le signal du bruit
statistique, et n'améliore le modèle qu'avec un signal de récompense qu'on peut vérifier.

## Architecture

```
  sources officielles (dépôts dbt-labs/docs.getdbt.com et apache/airflow, versions figées)
        │  ingestion/build_corpus.py      MDX / reStructuredText -> Markdown propre (corpus/*.md)
        ▼
  ingestion/chunking.py                   chunks par sections + fil d'Ariane (≤ 400 tokens)
        ▼
  retrieval/vector_store.py               multilingual-e5 -> ChromaDB (top 20)
        │                                 -> reranker cross-encoder multilingue (top 4)
        ▼
  rag_chain/chain.py                      prompt (sources citées, refus explicite) -> Ollama
        │
        ├── api/main.py                   FastAPI POST /query, GET /health
        │
        ├── evaluation/                   métriques déterministes + Ragas (juge distinct)
        │                                 + IC bootstrap -> MLflow ; compare_runs.py
        │
        └── training/                     questions -> réponses on-policy -> récompense
                                          vérifiable -> paires -> DPO QLoRA -> Ollama
```

## Stack technique

| Brique | Outil | Notes |
|---|---|---|
| Générateur | Ollama + Llama 3.1 8B (Q4_K_M) | `num_ctx` explicite (8192) : sinon le prompt peut être tronqué sans erreur |
| Embeddings | `intfloat/multilingual-e5-small` | Questions en français, corpus en anglais : modèle multilingue, préfixes `query:` / `passage:` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingue, léger ; recherche vectorielle large (20) puis top 4 |
| Base vectorielle | ChromaDB (client persistant) | La configuration d'indexation est stockée dans la collection et vérifiée à chaque recherche |
| Orchestration | LangChain (LCEL) | Un seul prompt, partagé par l'inférence et l'entraînement |
| API | FastAPI | Sources citées vs récupérées, (ré)indexation automatique au démarrage |
| Évaluation | Métriques déterministes + Ragas | Juge `qwen2.5:7b`, distinct du générateur ; IC bootstrap à 95 % |
| Suivi | MLflow (SQLite) | Configuration réellement indexée, commit git, empreinte du jeu d'éval, artefacts |
| Alignement | TRL (DPO) + PEFT (QLoRA) | Llama 3.1 8B Instruct, sur GPU Colab/Kaggle ; export vers Ollama |
| CI/CD | GitHub Actions + GHCR | Lint, tests, corpus reproductible, image Docker publiée |

## Corpus

10 pages de la documentation officielle : **dbt** (modèles SQL, Jinja et macros, data tests,
modèles incrémentaux, matérialisations) et **Apache Airflow** (Dags, opérateurs, scheduler,
XComs, bonnes pratiques). `ingestion/build_corpus.py` part des **sources** de ces pages (MDX du
dépôt `dbt-labs/docs.getdbt.com`, reStructuredText du dépôt `apache/airflow` au tag `3.3.1`), à
des versions figées dans le script : le corpus est reproductible au caractère près, ce que la CI
vérifie à chaque push. Les fragments réutilisables, FAQ et exemples de code inclus par ces pages
sont résolus ; les titres de sections sont conservés pour le chunking.

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows (source venv/bin/activate sous Linux/macOS)
pip install -r requirements.txt
cp .env.example .env           # puis ajuster si besoin (voir la note sur les ports ci-dessous)

ollama pull llama3.1:8b        # générateur
ollama pull qwen2.5:7b         # juge Ragas (distinct du générateur)
```

> **Note Windows / Hyper-V** : sur une machine où Hyper-V ou WSL2 est actif, le port par défaut
> d'Ollama (11434) peut être réservé par le système (`netsh interface ipv4 show
> excludedportrange protocol=tcp`), provoquant l'erreur `bind: An attempt was made to access a
> socket in a way forbidden by its access permissions`. Démarrer alors Ollama sur un autre port
> (`OLLAMA_HOST=127.0.0.1:12500 ollama serve`) et renseigner `OLLAMA_BASE_URL` (et
> `OLLAMA_BASE_URL_DOCKER` pour Docker) dans `.env`. Même problème possible avec le port 8000 de
> l'API : utiliser `--port 8600`, ou `HOST_PORT` dans `.env` pour Docker.

## Utilisation

```bash
# 0. (optionnel) Reconstruire le corpus depuis les sources — il est déjà versionné
python ingestion/build_corpus.py

# 1. Chunking par sections (télécharge le tokenizer d'embeddings)
python ingestion/chunking.py

# 2. Indexation ChromaDB + vérification manuelle du retrieval (sans LLM)
python retrieval/vector_store.py

# 3. Chaîne RAG complète en ligne de commande
python rag_chain/chain.py

# 4. API (Swagger : http://127.0.0.1:8000/docs)
uvicorn api.main:app --reload --port 8000

# 5. Évaluation
python evaluation/run_eval.py --retrieval-only               # retrieval seul, quelques secondes
python evaluation/run_eval.py --run-name baseline_v2         # complet (génération + Ragas)
python evaluation/compare_runs.py baseline_v2 autre_run      # comparaison appariée
mlflow ui --backend-store-uri sqlite:///mlruns.db            # http://127.0.0.1:5000
```

Exemple d'appel API :
```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "Comment pousser une valeur dans un XCom ?"}'
```

### Avec Docker

```bash
docker compose up -d --build
curl http://127.0.0.1:8600/health
```
L'image construit les chunks au build ; au premier démarrage, l'API indexe dans `./chroma_db`
(volume) et réindexe si les chunks ou le modèle d'embeddings ont changé. Le conteneur contacte
Ollama sur l'hôte via `OLLAMA_BASE_URL_DOCKER`. L'image publiée par la CD est disponible sur
`ghcr.io/rayanghomsi10/assistant-rag`.

## Protocole d'évaluation

Jeu : `evaluation/eval_dataset.json`, **85 questions** — 73 dans le périmètre (les 10 documents
sont couverts) et 12 hors périmètre, dont des pièges proches du domaine (Prefect, Dagster,
Snowflake, Kafka Connect…). Chaque question du périmètre porte des **preuves recopiées mot pour
mot du corpus** ; `evaluation/dataset.py` vérifie qu'elles s'y trouvent (en CI à chaque push).
Ce jeu est tenu à l'écart de l'entraînement DPO.

| Famille | Métriques | Juge |
|---|---|---|
| Retrieval | `evidence_recall`, `evidence_hit`, `source_hit`, `mrr` | Aucun : présence des preuves dans les chunks récupérés |
| Génération, vérifiable | `false_refusal`, `correct_refusal`, `citations_valid`, `cites_expected_source`, `quotes_faithful`, `code_support` | Aucun : règles déterministes (`rag_chain/checks.py`) |
| Génération, sémantique | `faithfulness`, `answer_relevancy`, `context_precision` (Ragas) | `qwen2.5:7b`, distinct du générateur ; périmètre uniquement |

Chaque moyenne est donnée avec un **intervalle de confiance bootstrap à 95 %** sur les questions.
`compare_runs.py` compare deux runs par **bootstrap apparié** (différences question par
question) et indique si l'écart est significatif. `--repeats N` relance génération et juge pour
mesurer la variance entre exécutions. `--retrieval-only` donne aussi la distribution des scores
du meilleur chunk dans et hors périmètre, pour calibrer un éventuel seuil de refus avant LLM
(`RELEVANCE_THRESHOLD`, désactivé par défaut).

## Résultats

Les résultats de la V2 (corpus, index, jeu d'évaluation et juge nouveaux) sont produits avec
`python evaluation/run_eval.py --run-name baseline_v2 --repeats 2` ; ils ne sont pas comparables
à ceux de la V1, archivés dans `evaluation/results/v1/` (voir « Historique »). Moyenne et
intervalle de confiance à 95 %, détail dans `evaluation/results/baseline_v2/`.

| Run | evidence_hit | false_refusal | quotes_faithful | faithfulness | answer_relevancy | correct_refusal |
|---|---|---|---|---|---|---|
| `baseline_v2` | 0,973 [0,932 ; 1,000] | 0,000 [0,000 ; 0,000] | 1,000 [1,000 ; 1,000] | 0,864 [0,802 ; 0,921] | 0,926 [0,909 ; 0,939] | 1,000 [1,000 ; 1,000] |
| `dpo_v1` | à mesurer | | | | | |

## Alignement du générateur par DPO

**Objectif** : corriger les défauts mesurés du générateur — citations reformulées présentées
comme textuelles, sources citées hors contexte, refus à tort, réponses non ancrées — sans
l'entraîner contre un juge LLM bruité.

**Pipeline** (`training/`, détails dans `training/README.md`) :

1. `generate_questions.py` — questions d'entraînement générées à partir de chaque chunk (le
   chunk source est conservé), questions hors périmètre rédigées à la main, **filtre anti-fuite**
   contre le jeu d'évaluation (recouvrement lexical et similarité d'embeddings).
2. `build_preferences.py` — pour chaque question, le pipeline réel récupère le contexte et le
   modèle déployé génère plusieurs réponses (données *on-policy*) ; la **récompense vérifiable**
   (`reward.py`) les note ; la meilleure réponse valide et la moins bonne forment une paire.
   Contraintes dures : pas de refus quand le passage source est dans le contexte, refus hors
   périmètre, pas de document cité hors contexte, pas de citation inventée. Composantes :
   implication des phrases par le contexte (NLI multilingue), source citée, identifiants de code
   présents dans le contexte, concision.
3. `train_dpo.py` / `dpo_colab.ipynb` — DPO avec TRL et QLoRA sur Llama 3.1 8B Instruct (le
   même modèle que `llama3.1:8b`), avec le gabarit de chat d'Ollama et le prompt exact de
   l'inférence ; GPU 16 Go (Colab ou Kaggle).
4. `export_ollama.py` — adaptateur appliqué à `llama3.1:8b` (Modelfile `ADAPTER`), ou modèle
   fusionné puis quantifié ; évaluation avec `run_eval.py --model rag-assistant-dpo` et
   comparaison appariée avec `baseline_v2`.

**Règle de décision** : le modèle DPO n'est adopté que si `compare_runs.py` montre une
amélioration significative sur au moins une métrique visée (`quotes_faithful`, `faithfulness`,
`false_refusal`, `citations_valid`) **et** aucune dégradation significative de
`answer_relevancy` ni de `correct_refusal`.

## CI/CD

`.github/workflows/ci-cd.yml` : à chaque push et pull request, lint (ruff), tests unitaires sans
modèle ni GPU, validation du jeu d'évaluation contre le corpus, reconstruction du corpus depuis
les sources figées (aucun écart toléré), build de l'image Docker. Sur `main` et sur les tags
`v*`, l'image est publiée sur GitHub Container Registry puis démarrée pour vérifier que l'API
construit son index et répond sur `/health`. `.github/workflows/training-smoke.yml` exécute un
entraînement DPO minuscule sur CPU quand le code d'entraînement change.

## Limites connues

- **Juge LLM local** : Ragas reste jugé par un modèle de 7B. Il est désormais distinct du
  générateur (moins de biais d'auto-évaluation), mais ses scores restent bruités : d'où les
  métriques déterministes en parallèle et les intervalles de confiance.
- **La récompense DPO ne mesure pas la pertinence de la réponse** : seulement son ancrage et le
  respect des règles. Risque : apprendre des réponses fidèles mais évasives. Garde-fous :
  longueur minimale, `answer_relevancy` dans la règle de décision, inspection manuelle d'un
  échantillon de paires.
- **Questions d'entraînement synthétiques** : générées par LLM, donc moins variées que de vraies
  questions d'utilisateurs.
- **Adaptateur QLoRA sur une base Q4_K_M** : l'adaptateur est entraîné sur une base quantifiée en
  NF4 et appliqué par Ollama sur une base quantifiée autrement ; Ollama signale que le résultat
  peut s'écarter. L'évaluation tranche ; sinon, voie « fusion » de `export_ollama.py`.
- **Corpus restreint** : 10 documents, un seul domaine. Hors périmètre, le système refuse par
  construction du prompt.
- **Taille du jeu d'évaluation** : 85 questions donnent des intervalles de confiance de l'ordre
  de ±0,05 à ±0,10 sur des métriques binaires ; de petits écarts resteront non significatifs.
- **Concurrence Ragas vs Ollama local** : Ragas parallélise par défaut (16 workers) alors qu'un
  Ollama local sert en série ; `RunConfig(max_workers=2, timeout=300)` évite les `TimeoutError`.
- **Compatibilité ragas 0.4.3 / langchain-community** : contournement d'un import VertexAI
  retiré, dans `evaluation/_ragas_compat.py` (sans impact fonctionnel).
- **`docker run -v` sous Git Bash (Windows)** : la conversion de chemins MSYS corrompt les
  volumes passés en ligne de commande ; utiliser `docker compose up`.

## Historique : ce que la V1 a appris

La V1 (corpus PDF tiré du HTML des sites, `all-MiniLM-L6-v2`, chunks fixes de 500 tokens,
15 questions, juge = générateur) affichait faithfulness 0,818, answer relevancy 0,581 et
context precision 0,987. Son analyse a motivé la V2 :

- le modèle d'embeddings n'était pas multilingue alors que les questions sont en français et le
  corpus en anglais — suspect au moins aussi probable que le juge pour l'answer relevancy basse ;
- une context precision de 0,987 sur 10 documents avec des questions qui nomment l'outil ne
  prouvait pas grand-chose (la question hors périmètre y obtenait même 1,0) ;
- la comparaison 500/50 contre 300/30 tokens reposait sur une exécution de 15 questions, où des
  questions basculaient de 1,0 à 0,0 : l'écart de −0,07 n'était pas distinguable du bruit ;
- la question de refus valait 0 par construction et tirait les moyennes vers le bas ;
- le texte extrait des PDF contenait des artefacts de navigation et des phrases coupées ;
- MLflow étiquetait les runs avec la configuration du `.env` au moment de l'évaluation, pas
  celle qui avait produit l'index.

## Structure du dépôt

```
assistant-rag/
├── config.py                    # configuration centrale (surchargeable par variables d'env)
├── corpus/                      # 10 documents Markdown, reconstruits depuis les sources
├── ingestion/
│   ├── build_corpus.py          # MDX / reStructuredText -> Markdown (versions figées)
│   ├── chunking.py              # chunking par sections
│   └── embeddings.py            # embeddings multilingues + reranker
├── retrieval/vector_store.py    # ChromaDB + reranking + métadonnées d'index
├── rag_chain/
│   ├── chain.py                 # prompt unique, génération
│   └── checks.py                # vérifications déterministes (éval et récompense)
├── api/main.py                  # FastAPI
├── evaluation/
│   ├── eval_dataset.json        # 85 questions avec preuves
│   ├── dataset.py               # validation contre le corpus
│   ├── metrics.py               # métriques déterministes, bootstrap
│   ├── run_eval.py              # boucle d'évaluation + MLflow
│   ├── compare_runs.py          # comparaison appariée
│   └── results/                 # un dossier par run (v1/ : archives)
├── training/                    # pipeline DPO (voir training/README.md)
├── tests/                       # tests unitaires (sans modèle ni GPU) + smoke test DPO
├── .github/workflows/           # CI/CD
├── Dockerfile, docker-compose.yml
└── requirements.txt, requirements-dev.txt, requirements-train.txt
```

Non versionnés : `venv/`, `chroma_db/`, `ingestion/chunks.json`, `mlruns.db`, `.env`,
`training/data/`, `training/outputs/`.
