# 20. Observabilité, interface et conversation

Ces fonctionnalités ont été ajoutées après la mesure de la baseline, pour observer le système et en faire un outil de travail : on peut suivre chaque requête, tester à la main, poursuivre une conversation et recevoir des mises en garde. Elles ne changent pas le prompt de la réponse principale quand il n'y a pas d'historique : les résultats du chapitre 19 restent valables.

## 20.1 Observabilité avec Langfuse

MLflow (section 10.5) garde des runs agrégés : une configuration, des moyennes, des fichiers. Il ne montre pas ce qui s'est passé pour _une_ question. Langfuse enregistre une **trace** par question : la `recherche` (passages, scores du reranker), l'appel au LLM (prompt exact, réponse, latence, tokens) et le résultat. C'est l'outil qui permet de répondre à « pourquoi cette réponse ? ».

|**Origine**|**Trace**|**Regroupement et scores**|
|---|---|---|
|API, interface|`question-rag`|Étiquette api ; session = identifiant de conversation.|
|`run_eval.py`|`eval:<id>`|Session = nom du run ; métriques déterministes et Ragas attachées comme scores, pour filtrer les réponses en échec.|
|`build_preferences.py`|`dpo:<id>`|Session par run ; récompenses de la paire comme scores.|
|`run_conversation_eval.py`|`conversation:<id>`|Session = nom du run ; question reformulée.|

Exemple réel d'une trace d'API : `question-rag` contient `recherche` (retriever) puis `génération`, qui contient l'appel `ChatOllama` : modèle llama3.1:8b, 1 324 tokens en entrée, 94 en sortie, 14,9 s.

|**Fonction de rag_chain/tracing.py**|**Rôle**|
|---|---|
|`is_active()`|Clés présentes et serveur joignable (vérifié une fois, 3 s au plus).|
|`observe(name, as_type)`|Décorateur : trace une fonction (`recherche`, question, compléments).|
|`langchain_config(run_name)`|Rattache un appel LangChain au LLM à la trace en cours.|
|`trace(...)`, `attributes(...)`|Ouvre une trace ; étiquette et regroupe en session.|
|`set_output`, `score`, `score_trace`|Sortie et scores de la trace en cours ou d'une trace terminée (Ragas, après coup).|
|`current_trace_id`, `trace_url`, `flush`|Identifiant et lien de la trace ; envoi des données en fin de script.|

> [!TIP]
> **POURQUOI CE CHOIX**
>
> **Optionnel et sans dépendance cachée.** Sans clés, aucune fonction n'agit et `langfuse` n'est pas importé : les tests et la CI ne dépendent d'aucun serveur. La vérification de disponibilité a été ajoutée après un incident : avec des clés mais des conteneurs arrêtés, chaque lot de traces attendait l'expiration de son envoi, et une évaluation restait bloquée plusieurs minutes à la fin. Désormais le traçage se désactive avec un avertissement.

**Serveur local.** Le dossier `langfuse/` contient le `docker-compose.yml` officiel (Langfuse v4 : interface, worker, Postgres, ClickHouse, Redis, MinIO), un override sans redémarrage automatique, qui fait aussi entrer les nouveaux comptes dans l'organisation du projet, et `init_env.py`, qui génère des secrets aléatoires à la place des mots de passe par défaut (`langfuse/.env`, non versionné). Vérification : `python -m rag_chain.tracing`.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> **Langfuse Cloud** : rien à faire tourner, mais les questions et `passages` partent sur un serveur tiers. **Le traçage MLflow** (`mlflow.langchain.autolog`) : déjà installé, le plus léger, interface moins riche pour les sessions et les scores. **Phoenix** (Arize) ou **OpenTelemetry + Jaeger** : standards ouverts, plus génériques. **LangSmith** : payant au-delà d'un quota et non local.

## 20.2 Interface web de test

`api/static/index.html` est servie sur / par l'API. Elle montre tout ce qu'il faut pour juger une réponse : badge de refus (en distinguant le refus par seuil, sans appel au LLM), sources citées et sources récupérées mais non citées, `passages` exacts fournis au modèle avec leur score de reranking et leur cosinus, question reformulée, compléments, modèle, latence et lien vers la trace Langfuse.

|**Champ de l'API**|**Sens**|**Rôle**|
|---|---|---|
|`history`|requête|Échanges précédents {question, answer}, du plus ancien au plus récent.|
|`conversation_id`|requête|Identifiant de conversation, devient la session Langfuse.|
|`complements`|requête|Active la recherche de compléments (faux par défaut).|
|`standalone_question`|réponse|Question reformulée qui a servi à la recherche.|
|`passages`|réponse|Passages fournis au LLM : source, section, scores, texte, cité ou non.|
|`complements`, `trace_url`|réponse|Compléments vérifiés ; lien de la trace.|

> [!TIP]
> **POURQUOI CE CHOIX**
>
> **Une page statique, sans framework.** Pas d'étape de build ni de second processus : elle est servie par l'API, y compris dans l'image Docker. C'est un outil de test pour voir ce que le système a lu et décidé, pas un produit fini.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> **Streamlit** ou **Gradio** : rapides à écrire en Python, mais un second serveur. **Chainlit** : spécialisé dans les interfaces de chat pour LLM, avec affichage des étapes intermédiaires. Un front **React** : pour un produit, surdimensionné pour tester.

## 20.3 Mémoire de conversation

Sans mémoire, chaque question est traitée seule. Une relance comme « Laquelle est utilisée par défaut ? » ne contient pas son sujet : la recherche vectorielle n'a rien sur quoi s'appuyer. La mémoire sert d'abord la **recherche**, puis la génération.

1. L'interface envoie l'historique avec chaque question ; l'API ne stocke rien.

2. `condense_question` demande au LLM de reformuler la relance en question autonome, à partir des `HISTORY_MAX_TURNS` derniers échanges (3 par défaut). Garde-fou : une sortie vide ou trop longue (le modèle a répondu au lieu de reformuler) est remplacée par la question d'origine.

3. La recherche utilise la question reformulée, affichée dans l'interface.

4. La génération reçoit les derniers échanges entre le message système et la question (`MessagesPlaceholder`), les réponses étant tronquées à 1 500 caractères. Les sources restent les passages de la question courante.

5. Sans historique, le prompt est identique au caractère près : évaluation et DPO ne sont pas affectés.

**Mesure.** `evaluation/conversations.json` contient 18 relances, chacune rattachée à une question du jeu d'évaluation dont les preuves sont connues. L'historique contient la question précédente et sa réponse de référence. On compare la recherche avec la relance brute, reformulée, et avec la question écrite en entier (plafond).

|**Métrique**|**Relance brute**|**Reformulée**|**Question complète**|**Écart**|
|---|---|---|---|---|
|`evidence_hit`|0,444 [0,22 ; 0,67]|**0,778 [0,56 ; 0,94]**|1,000 [1,00 ; 1,00]|+0,333|
|`evidence_recall`|0,444 [0,22 ; 0,67]|**0,750 [0,53 ; 0,94]**|0,972 [0,92 ; 1,00]|+0,306|
|`mrr`|0,389 [0,19 ; 0,61]|**0,671 [0,46 ; 0,87]**|0,866 [0,74 ; 0,97]|+0,282|
|`source_hit`|0,722 [0,50 ; 0,94]|**1,000 [1,00 ; 1,00]**|1,000 [1,00 ; 1,00]|+0,278|

Les quatre écarts sont significatifs (bootstrap apparié). Sur les 4 relances où la reformulation ne retrouve pas la preuve, une a changé de sens (le modèle a recopié la question précédente) ; les trois autres sont correctes mais formulées autrement que le passage, ce qui montre aussi la sensibilité du retrieval à la formulation.

> [!TIP]
> **POURQUOI CE CHOIX**
>
> **Historique côté client.** L'API reste sans état : rien à stocker, rien à purger, plusieurs instances possibles. L'identifiant de conversation sert seulement à regrouper les traces.

> [!WARNING]
> **ATTENTION**
>
> 18 relances et une seule exécution : un ordre de grandeur, pas une mesure fine. L'historique de la mesure utilise les réponses de référence, pas des réponses générées, et l'ancrage des réponses générées avec historique n'est pas encore évalué. La reformulation coûte un appel au LLM par relance.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> **Concaténer les dernières questions** à la requête de recherche : gratuit, mais bruité. **Mémoire côté serveur** (sessions, Redis) : nécessaire pour reprendre une conversation plus tard. **Résumé glissant** de la conversation : pour les longues discussions, au-delà de la fenêtre de contexte. **LangGraph** et ses points de sauvegarde : pour des agents à plusieurs étapes.

## 20.4 Compléments : mises en garde vérifiées (expérimental)

Objectif : signaler ce qu'un ingénieur risque d'ignorer, comme le ferait un collègue (« attention, les colonnes de unique_key ne doivent pas être nulles »). L'ajouter au prompt principal aurait trois défauts : les 4 passages sont choisis pour répondre, pas pour trouver les pièges, donc le modèle puiserait dans sa mémoire ; la faithfulness, l'answer relevancy et la concision de la récompense DPO baisseraient sans qu'on sache si le système s'améliore ; et le prompt unique de l'évaluation et du DPO serait modifié. D'où un module séparé, `rag_chain/complements.py`.

|**Étape**|**Ce qui est fait**|
|---|---|
|1 · Recherche dédiée|40 candidats reclassés par le reranker ; ne garde que les passages contenant un encadré (Note, Warning, Tip, Info…, conservés par build_corpus.py) ou issus du guide de bonnes pratiques d'Airflow, absents du contexte de la réponse, de score ≥ 0.|
|2 · Génération|Un appel au LLM en JSON : au plus 3 compléments typés (mise en garde, autre façon de faire, à savoir), chacun avec son document. Une liste vide est la réponse attendue le plus souvent.|
|3 · Filtre par programme|Document parmi ces passages ; citations et identifiants de code vérifiés (checks.py) ; pas de question à la place d'une information.|
|4 · Filtre NLI|Probabilité d'implication moyenne ≥ 0,5 sur les phrases (modèle de la récompense DPO, déplacé dans rag_chain/nli.py). Si le NLI ne se charge pas, rien n'est affiché.|

**Observé sur 8 questions.** Aucun complément pour 5 d'entre elles. Pour les 3 autres, des compléments utiles et tirés du texte (l'avertissement sur `dbt-snowflake` avant la version 1.10.6 pour les modèles incrémentaux, les self-checks et l'environnement de test pour les Dags), mais aussi des platitudes ancrées (« il est important de connaître la structure du Dag »). Coût : 5 à 10 s par question. Avant le filtre NLI, le modèle produisait des conseils génériques présentés comme tirés de la documentation.

> [!WARNING]
> **ATTENTION**
>
> Le NLI vérifie qu'une phrase découle d'un passage, pas qu'elle est utile ni liée à la question. Les seuils (score du reranker ≥ 0, implication ≥ 0,5) ne sont pas calibrés et aucune mesure n'existe encore : la fonction reste expérimentale, et désactivée par défaut dans l'API.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> **Mesurer d'abord** : un jeu de questions dont la mise en garde attendue est connue (rappel des mises en garde, précision des compléments), comme pour la mémoire. **Questions de suivi cliquables** : moins risqué, aucune nouvelle affirmation. **Marquer les mises en garde à l'indexation** (métadonnée du chunk) plutôt qu'à la volée. **Juge d'utilité** par LLM : utile pour trier, mais bruité.

## 20.5 Robustesse du pipeline DPO

**Le goulot était le NLI, pas le LLM.** Un chronométrage étape par étape a montré qu'une génération prenait environ 11 s et la notation d'une réponse de 31 mots environ 45 s : PyTorch était installé sans CUDA et le modèle NLI tournait sur le processeur. Avec la roue CUDA de PyTorch, la notation prend environ 1 s sur une RTX 4060. `build_preferences.py` affiche maintenant le périphérique du NLI au démarrage.

**Reprise après interruption.** Le script écrit sa progression question par question (`preferences_progress.jsonl`) et chaque paire immédiatement. Relancé, il reprend où il s'était arrêté et retire une paire écrite juste avant l'interruption, pour ne pas l'avoir en double. Il refuse de reprendre si les paramètres ou les questions ont changé (`--restart` repart de zéro). Quatre tests couvrent ces cas (`tests/test_preferences_resume.py`).

> [!IMPORTANT]
> **LA LEÇON**
>
> Le premier run de nuit a été coupé à 3 h 16 par un redémarrage de Windows Update, après environ 28 questions sur 418 : tout était perdu. Un traitement de plusieurs heures doit sauvegarder au fil de l'eau. Et avant d'optimiser, mesurer : l'intuition désignait le LLM, le chronométrage a désigné le NLI.

## 20.6 Révision de la sélection des paires

Le premier run a produit 164 paires. Avant d'entraîner, la procédure impose de relire un échantillon : c'est cette relecture qui a fait revoir la sélection.

|**Constat**|**Paires**|
|---|---|
|Relecture de 30 paires : la réponse retenue est vraiment meilleure|9|
|Relecture : contenu équivalent, seule la citation du document diffère|13|
|Relecture : la réponse retenue est moins bonne ou douteuse (mauvais opérateur, tautologie, erreur factuelle, exemple exact de la documentation écarté)|8|
|Sur les 164 : rejetée pour une contrainte dure (citation inventée, faux refus, trop courte)|37|
|Sur les 164 : écart de fond inférieur à 0,05, la citation fait toute la différence|48|
|Sur les 164 : réponse retenue moins ancrée que la rejetée selon le NLI|8|

**Cause.** La citation pèse 0,30 dans la récompense (source citée 0,20, présence d'une citation 0,10), alors que l'écart minimal pour former une paire est de 0,20 : une citation suffit à créer une paire, même quand la réponse retenue est moins bonne sur le fond.

- **Écart sur le fond.** Quand la réponse rejetée respecte les contraintes dures, l'écart pondéré sur l'ancrage, le code et la concision doit atteindre `--min-substance` (0,1). La rejetée devient la moins bonne réponse qui remplit ces conditions, pas forcément la pire.

- **Toutes les réponses candidates sont enregistrées** (`preferences_candidates.jsonl`) : `training/select_pairs.py` refait la sélection avec d'autres seuils sans régénérer des heures de réponses. Le premier run ne gardait que la paire, d'où la nécessité de tout recommencer.

- **Contrôle des citations corrigé** : un guillemet orphelin dans une réponse décalait l'appariement, un bloc de code était lu comme une citation, et l'espace avant un point d'exclamation (typographie française) faisait échouer la comparaison. Environ 5 rejets sur 29 venaient de ces défauts.

- **3 questions par chunk au lieu de 2**, pour compenser les paires que la nouvelle règle écarte.

> [!TIP]
> **POURQUOI CE CHOIX**
>
> **On corrige la sélection, pas la règle de décision.** La règle d'adoption du modèle (section 11.5) ne change pas, et la nouvelle sélection est fixée avant tout entraînement : on ne choisit pas le critère d'après un résultat. La correction du contrôle des citations rend la métrique `quotes_faithful` un peu plus tolérante ; la baseline, déjà à 1, n'est pas affectée.

> [!WARNING]
> **ATTENTION**
>
> La relecture porte sur 30 paires et mérite une seconde lecture. Le NLI vérifie l'ancrage, pas la pertinence : une réponse bien ancrée qui répond à côté de la question peut encore être retenue.

**Second run.** Sur 600 questions, 144 paires (l'ancienne règle en aurait formé 224 : 80 ne tenaient qu'à la citation). Relecture de 30 paires : 19 nettement justifiées (12 sur le fond, 7 sur une règle du prompt, comme une paraphrase mise entre guillemets), 8 où les deux réponses disent presque la même chose mais où le NLI les note très différemment (0,99 contre 0,37), 3 où la rejetée paraît plus complète. Relever le seuil ne corrige pas ces cas : à 0,3, il reste 101 paires et 7 des 11 cas douteux. La limite restante vient de la variabilité du NLI sur des paraphrases, pas du seuil.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> **Baisser le poids des citations** dans la récompense. **Ajouter la citation par post-traitement** plutôt que l'apprendre, puisqu'elle se vérifie mécaniquement. **Ajouter une composante de pertinence** (similarité entre la question et la réponse), contre les réponses ancrées mais à côté du sujet. **Stabiliser le NLI** : noter chaque phrase contre une fenêtre de passages plutôt qu'un passage entier, ou moyenner plusieurs modèles, pour que deux paraphrases reçoivent des notes proches.
