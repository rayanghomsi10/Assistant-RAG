# 12. Tests

66 tests répartis en 11 fichiers, conçus pour tourner en quelques secondes sans modèle, sans base vectorielle réelle ni GPU, sauf deux fichiers qui demandent ChromaDB ou PyTorch.

|Fichier|Tests|Ce qui est vérifié|
|---|---|---|
|`test_build_corpus.py`|5|Rôles RST, titres, directives, commentaires, inclusions d'exemples ; composants MDX (constantes, termes, fichiers, FAQ, fragments avec props) ; suppression des sections de navigation. Sources simulées.|
|`test_chunking.py`|4|Sections et `#` dans le code ; regroupement, recoupage et fil d'Ariane ; tous les chunks du vrai corpus sous la taille maximale ; empreinte du corpus.|
|`test_checks.py`|8|Détection des refus, normalisation des citations de documents, citations verbatim et inventées, identifiants de code ; guillemet orphelin, bloc de code et espace de la typographie française.|
|`test_dataset.py`|4|Toutes les preuves sont dans le corpus ; composition du jeu ; rejet d'une preuve inventée et des doublons.|
|`test_metrics.py`|7|Métriques de retrieval (rang, rappel), faux refus, bootstrap, test apparié (effet détecté, absence d'effet), seuil.|
|`test_pipeline.py`|4|Run d'évaluation complet avec recherche et LLM simulés ; mode retrieval seul ; détection d'une régression par `compare_runs`; API (`/health`, `/query`, validation).|
|`test_vector_store.py`|3|Indexation ChromaDB réelle avec embeddings factices : métadonnées, recherche, détection d'un index obsolète ou d'un autre modèle, reranking.|
|`test_training.py`|18|Récompense (cas valides, 4 contraintes dures, hors périmètre, composantes, NLI simulé), sélection des paires (avec écart sur le fond) et re-sélection hors ligne, disjonction et filtre anti-fuite, déduplication, découpage, export Ollama, gabarit de chat identique à celui d'Ollama.|
|`test_train_dpo_smoke.py`|2|Avec un Llama minuscule : tokenisation par TRL (BOS en tête, `<\|eot_id\|>` en fin de réponse), seuls les adaptateurs LoRA entraînables ; deux pas d'entraînement et sauvegarde de l'adaptateur, uniquement si un GPU est présent.|
|`test_conversation.py`|7|Prompt identique sans historique ; ordre et troncature de l'historique ; pas d'appel au LLM sans historique ; jeu de conversations valide ; filtres des compléments (NLI simulé).|
|`test_preferences_resume.py`|4|Reprise après une interruption, sans paire en double ; refus de reprendre si les paramètres changent ; ancien fichier protégé ; fichier des candidats ; traçage inactif sans clés.|

> [!TIP]
> **POURQUOI DES FONCTIONS INJECTABLES**
> Les fonctions qui dépendent d'un service externe prennent la dépendance en paramètre (`fetch` pour le téléchargement, `count_tokens` pour le tokenizer, `nli` pour le modèle NLI) ou sont remplacées par `monkeypatch` dans les tests (`search`, `generate`). C'est ce qui permet de tester la logique en isolation, rapidement, en CI.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Tests d'intégration avec de vrais modèles** (embeddings et reranker sur CPU, un petit LLM via Ollama dans la CI) : plus lents, ils vérifieraient les tailles réelles en tokens et la qualité minimale du retrieval.
>
> - **Seuil de non-régression du retrieval en CI** : échouer si `evidence_hit` passe sous une valeur de référence. Nécessite les modèles d'embeddings en CI.
>
> - **Couverture de code** (pytest-cov) affichée en CI.
