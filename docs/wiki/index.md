# Assistant RAG : documentation et cours du projet

Assistant qui répond aux questions sur la documentation officielle de dbt et d'Apache Airflow,
cite ses sources et refuse quand la réponse n'y est pas ; protocole d'évaluation statistique ;
alignement du générateur par DPO. Conventions de ce wiki : [SCHEMA.md](SCHEMA.md). Historique des
modifications : [log.md](log.md).

## Partie I : notions

| Page | Résumé |
|---|---|
| [1. Introduction](01-introduction.md) | Le problème, le projet en une page, état réel, comment lire ce document. |
| [2. Notions fondamentales](02-notions-fondamentales.md) | LLM, RAG, embeddings, chunking, reranking, évaluation, bootstrap, DPO, LoRA, NLI. |

## Partie II : architecture

| Page | Résumé |
|---|---|
| [3. Vue d'ensemble](03-vue-d-ensemble.md) | Les quatre chaînes, artefacts, arborescence du dépôt, stack, variables de configuration. |

## Partie III : les modules

| Page | Résumé |
|---|---|
| [4. Construction du corpus](04-construction-du-corpus.md) | `build_corpus.py` : sources MDX / RST à versions figées vers Markdown reproductible. |
| [5. Chunking](05-chunking.md) | Découpage par sections avec fil d'Ariane, taille mesurée par le tokenizer d'embeddings. |
| [6. Embeddings et reranker](06-embeddings-et-reranker.md) | `multilingual-e5-small` avec préfixes, cross-encodeur mMiniLM. |
| [7. Base vectorielle et recherche](07-base-vectorielle-et-recherche.md) | ChromaDB, 20 candidats puis reranking, contrôle de fraîcheur de l'index. |
| [8. Chaîne RAG et vérifications](08-chaine-rag-et-verifications.md) | Prompt unique, génération, seuil de refus, contrôles déterministes des réponses. |
| [9. API](09-api.md) | FastAPI : `/query`, `/health`, interface web, codes d'erreur. |
| [10. Évaluation](10-evaluation.md) | Jeu de 85 questions avec preuves, métriques, bootstrap, comparaison appariée, MLflow. |
| [11. Alignement par DPO](11-alignement-par-dpo.md) | Questions synthétiques, récompense vérifiable, paires, QLoRA, export Ollama, règle de décision. |
| [12. Tests](12-tests.md) | 66 tests sans modèle ni GPU, fichier par fichier. |
| [13. Conteneurisation et CI/CD](13-conteneurisation-et-ci-cd.md) | Image Docker, workflow GitHub Actions, corpus reproductible. |

## Partie IV : recul et résultats

| Page | Résumé |
|---|---|
| [14. Historique : de la V1 à la V2](14-historique-de-la-v1-a-la-v2.md) | Ce que mesurait la V1 et ce que sa relecture a montré. |
| [15. Récapitulatif des décisions](15-recapitulatif-des-decisions.md) | Chaque choix structurant, l'alternative écartée et la raison. |
| [16. Limites et points non mesurés](16-limites-et-points-non-mesures.md) | Ce qui reste à mesurer, limites de conception. |
| [17. Pistes d'évolution](17-pistes-d-evolution.md) | Évolutions classées par gain / effort, avec leur état. |
| [18. Mode d'emploi](18-mode-d-emploi.md) | Installation et commandes : construire, interroger, évaluer, entraîner, documenter. |
| [19. Résultats mesurés et état d'avancement](19-resultats-mesures-et-etat-d-avancement.md) | Baseline V2 avec intervalles de confiance, analyse des échecs, défauts révélés, état du DPO. |
| [20. Observabilité, interface et conversation](20-observabilite-interface-et-conversation.md) | Langfuse, interface de test, mémoire de conversation, compléments, robustesse et révision des paires DPO. |

## Annexes

| Page | Résumé |
|---|---|
| [A. Glossaire](A-glossaire.md) | Définitions des termes techniques. |
| [B. Questions de révision](B-questions-de-revision.md) | Questions corrigées pour vérifier sa compréhension. |
