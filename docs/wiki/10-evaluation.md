# 10. Évaluation

Le dossier `evaluation/` mesure la qualité du système et, surtout, dit si une différence entre deux versions est réelle. C'est la partie du projet qui a le plus changé entre la V1 et la V2.

## 10.1 Le jeu d'évaluation

`eval_dataset.json` contient **85 questions** : 73 dans le périmètre, réparties sur les 10 documents (entre 6 et 10 par document), et 12 hors périmètre. Chaque question du périmètre a une réponse de référence en français, ses documents attendus et des **preuves** : des extraits recopiés mot pour mot du corpus (98 au total).

```
{
  "id": "af_xcom_02",
  "type": "in_scope",
  "question": "Comment pousse-t-on une valeur dans un XCom depuis une tâche Airflow ?",
  "ground_truth": "Avec la méthode xcom_push de l'instance de tâche, par exemple
                   task_instance.xcom_push(key=..., value=...). …",
  "expected_sources": ["airflow_04_xcoms.md"],
  "evidence": ["task_instance.xcom_push(key=",
               "auto-push their results into an XCom key called return_value"]
}
```

Les questions hors périmètre mêlent des sujets éloignés (« Quelle est la capitale de l'Australie ? ») et des **pièges proches du domaine** : Prefect, Dagster, Kafka Connect, l'auto-suspend de Snowflake, le prix de dbt Cloud, NiFi. Ces derniers sont les plus utiles : la recherche ramène des passages qui « ressemblent », et le modèle est tenté de répondre.

`dataset.py` valide le jeu (`validate_dataset`) : identifiants et questions uniques, types connus, documents attendus présents dans le corpus, et **chaque preuve retrouvée dans un des documents attendus** après normalisation. Le test `test_eval_dataset_evidence_is_in_corpus` et un job CI échouent si une preuve disparaît du corpus.

> [!TIP]
> **POURQUOI DES PREUVES**
> En V1, une réponse de référence avait été écrite de mémoire, sur un sujet que le corpus ne couvrait pas ; le système avait correctement refusé, et cette bonne réponse avait été comptée comme une erreur. Les preuves rendent impossible ce type d'erreur, et permettent de mesurer le retrieval sans aucun LLM.

## 10.2 Les métriques

### Retrieval (questions du périmètre, sans LLM)

|Métrique|Définition|
|---|---|
|`evidence_recall`|Part des preuves présentes dans au moins un des 4 chunks récupérés.|
|`evidence_hit`|1 si au moins une preuve est présente.|
|`source_hit`|1 si au moins un document attendu figure parmi les documents récupérés.|
|`mrr`|1 / rang du premier chunk contenant une preuve, 0 si aucun.|

### Génération, par règles

|Métrique|Définition|
|---|---|
|`answered`|Question du périmètre à laquelle le système a répondu (pas de refus).|
|`false_refusal`|Refus_alors que la preuve était dans le contexte_: le modèle avait de quoi répondre. Plus bas = mieux.|
|`has_citation`,|Au moins une source citée ; aucune source citée hors contexte ; le document attendu est cité.|
|`citations_valid`, `cites_expected_source`|Calculées sur les réponses non refusées.|
|`quotes_faithful`|Toutes les citations entre guillemets sont verbatim dans le contexte.|
|`code_support`|Part des identifiants de code présents dans le contexte.|
|`correct_refusal`, `clean_refusal`|Questions hors périmètre : le système refuse ; et refuse sans dérouler une réponse inventée.|

### Génération, par Ragas (questions du périmètre uniquement)

|Métrique|Définition|
|---|---|
|`faithfulness`|Le juge décompose la réponse en affirmations et vérifie chacune contre le contexte : part des affirmations étayées.|
|`answer_relevancy`|Le juge génère des questions à partir de la réponse ; similarité cosinus moyenne entre ces questions et la question posée.|
|`context_precision`|Le juge évalue si chaque passage récupéré est utile à la réponse de référence ; précision pondérée par le rang.|

Le juge est `qwen2.5:7b`, distinct du générateur. Les embeddings de Ragas sont ceux du projet, avec le préfixe `query:` des deux côtés : `answer_relevancy` compare deux questions entre elles. La concurrence de Ragas est limitée (`max_workers=2`, `timeout=300` s) parce qu'un Ollama local sert les requêtes une par une ; avec les 16 workers par défaut, une partie des requêtes expirait en V1. Les questions hors périmètre sont **exclues** de Ragas : un refus correct y valait 0 par construction et faisait baisser les moyennes.

## 10.3 Le déroulement d'un run

`run_eval.py` enchaîne :

1. chargement et validation du jeu (un jeu invalide arrête tout) ;

2. pour chaque question : recherche (mise en cache entre répétitions, car déterministe), métriques de retrieval, puis génération (ou refus par seuil) et métriques par règles ;

3. Ragas, une fois par répétition ;

4. agrégation : moyenne par question sur les répétitions, puis moyenne et intervalle de confiance bootstrap sur les questions ; écart-type entre répétitions si `--repeats` > 1 ;

5. rapport sur les scores du meilleur chunk dans et hors périmètre, avec un seuil de refus suggéré (exactitude équilibrée maximale) ;

6. écriture de `results/<run>/per_question.csv` et `summary.json`, puis journalisation MLflow.

```
python evaluation/run_eval.py --retrieval-only               # quelques secondes, sans LLM
python evaluation/run_eval.py --run-name baseline_v2 --repeats 2
python evaluation/run_eval.py --run-name dpo_v1 --model rag-assistant-dpo
python evaluation/compare_runs.py baseline_v2 dpo_v1
```

Options : `--model` (générateur à évaluer), `--repeats`, `--retrieval-only`, `--skip-ragas`, `--limit` (N premières questions), `--no-mlflow`.

## 10.4 Comparer deux runs

`compare_runs.py` charge les résultats par question des deux runs, ne garde que les questions communes et applique un **bootstrap apparié** (5 000 tirages) sur les différences B − A. Pour chaque métrique, il affiche les deux moyennes, la différence, son intervalle de confiance, une p-valeur approchée et un verdict : « B meilleur », « B moins bon » ou « non significatif ». Pour `false_refusal`, une baisse est une amélioration.

```
metric             A      B   delta  ci_low  ci_high  p_value  n  verdict
answered       1.000  0.000  -1.000  -1.000   -1.000    0.000 73  B moins bon
correct_refusal 1.000 1.000   0.000   0.000    0.000    …     12  non significatif
```

Extrait simplifié d'une sortie produite par les tests, avec un générateur simulé qui refuse tout : ce ne sont pas des résultats du système.

## 10.5 Suivi MLflow

`mlflow_tracking.py` enregistre pour chaque run : la configuration **lue dans l'index** (préfixe `index_` : modèle d'embeddings, empreinte du corpus, paramètres de chunking), le générateur, le juge, top-k, candidats, reranker, seuil, `num_ctx`, nombre de répétitions, empreinte du jeu d'évaluation, mode ; les métriques avec leurs bornes d'intervalle ; les tags `git_commit` et `git_dirty` ; et en artefacts le CSV, le résumé et le jeu d'évaluation. En V1, la taille de chunk enregistrée était celle du `.env` au moment de l'évaluation, pas forcément celle de l'index réellement interrogé.

`_ragas_compat.py` contourne un import cassé : Ragas 0.4.3 importe un module VertexAI que les versions récentes de `langchain-community` ont supprimé. Le module enregistre un module factice avant l'import de Ragas ; sans effet fonctionnel puisque le projet n'utilise pas VertexAI.

## 10.6 Fonctions

|Fonction|Rôle|
|---|---|
|**`dataset.load_eval_dataset`**, **`load_corpus_texts`**|Chargement du jeu et du corpus.|
|**`dataset.validate_dataset`** `(items, corpus) → [erreurs]`|Validation complète ; liste vide si tout est correct.|
|**`dataset.norm`**,**`evidence_in`**,|Normalisation partagée avec `checks.py`; test de présence d'une preuve ; empreinte|
|**`dataset_fingerprint`**|du fichier.|
|**`metrics.retrieval_metrics`** `(item, hits) → dict`|Les quatre métriques de retrieval.|
|**`metrics.generation_metrics`** `(item, answer, hits, evidence_hit)` `→ dict`|Métriques par règles (via `check_answer`), selon le type de question.|
|**`metrics.bootstrap_ci`** `(values, n_boot=2000) → (moyenne,` `bas, haut, n)`|Intervalle percentile, valeurs manquantes ignorées.|
|**`metrics.paired_bootstrap`** `(a, b, n_boot=5000) → dict`|Différence appariée, intervalle, p-valeur.|
|**`metrics.suggest_threshold`** `(in_scores, out_scores) → dict`|Seuil qui maximise l'exactitude équilibrée.|
|**`run_eval.run_pipeline`**,**`run_ragas`**, **`summarize`**,**`print_summary`**, **`retrieval_report`**,**`safe_name`**,**`main`**|Étapes du run décrites ci-dessus.|
|**`compare_runs.load_run`**,**`compare`**, **`main`**|Chargement d'un run par nom ou chemin ; tableau comparatif.|
|**`mlflow_tracking.git_state`**, **`init_mlflow`**,**`log_eval_run`**|Commit et état du dépôt ; initialisation ; journalisation d'un run.|

## 10.7 Choix et alternatives

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Un juge plus fort** : un modèle de 70B en local (trop gros pour un GPU de 8 Go) ou une API (GPT-4o, Claude) pour Ragas. Scores plus fiables, au prix d'un coût et d'une dépendance externe. C'est l'amélioration la plus rentable si un budget existe.
>
> - **Autres frameworks** : DeepEval, TruLens, ARES, ou les évaluateurs de LangSmith. Mêmes familles de métriques, avec leurs propres biais de juge.
>
> - **Évaluation humaine** : noter 30 réponses à la main (exactitude, complétude) et mesurer l'accord avec le juge. La seule façon de savoir si le juge est fiable sur ce domaine.
>
> - **Exactitude de la réponse contre la référence** (answer correctness de Ragas, ou un score sémantique) : la réponse de référence existe déjà dans le jeu mais n'est utilisée que par context_precision.
>
> - **Jeu plus grand** : 85 questions donnent des intervalles d'environ ±0,05 à ±0,10 sur des métriques binaires. Pour détecter des écarts de 0,03, il en faudrait plusieurs centaines.
