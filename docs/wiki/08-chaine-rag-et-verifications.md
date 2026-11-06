# 8. Chaîne RAG et vérifications

`rag_chain/chain.py` (206 lignes) assemble le contexte, le prompt et l'appel au LLM. `rag_chain/checks.py` (164 lignes) vérifie mécaniquement qu'une réponse respecte les règles du prompt.

## 8.1 Le prompt

```
Tu es un assistant technique qui répond à des questions sur la documentation dbt
et Apache Airflow.
Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. N'utilise aucune
  autre connaissance.
- Si le contexte ne contient pas la réponse, réponds exactement : "Je ne trouve
  pas cette information dans les documents fournis." et rien d'autre. N'invente pas.
- Cite les documents qui justifient ta réponse par leur nom entre crochets, tel
  qu'il apparaît dans le contexte, par exemple [airflow_04_xcoms.md]. Ne cite
  jamais un document absent du contexte.
- N'utilise des guillemets que pour recopier un passage du contexte mot pour mot,
  dans sa langue d'origine (anglais). Sinon, reformule sans guillemets.
- Écris les noms de fonctions, de paramètres et de commandes exactement comme
  dans le contexte, entre accents graves (par exemple `xcom_push`).
- Réponds en français, de façon concise et précise.
Contexte :
[Document : airflow_04_xcoms.md | Section : Apache Airflow — XComs]
…
---
[Document : …]
```

Chaque règle a été écrite pour être vérifiable : une phrase de refus exacte, des citations dans un format reconnaissable, des guillemets réservés au verbatim, des identifiants entre accents graves. La règle sur les guillemets répond à un défaut observé en V1 : le modèle présentait entre guillemets des passages reformulés.

## 8.2 Fonctions de chain.py

|Fonction|Rôle|
|---|---|
|**`format_context`** `(hits) → str`|Met en forme les passages avec document et section, séparés par `---`.|
|**`build_messages`** `(question, hits) → [{role,` `content}]`|Messages exactement tels que le LLM les reçoit.**Seule définition du prompt**, réutilisée pour construire les exemples DPO.|
|**`get_llm`** `(model, temperature, seed) →` `ChatOllama`|Client Ollama avec `num_ctx` explicite.|
|**`build_chain`** `(model, temperature, seed)`|Chaîne LCEL `prompt | llm | StrOutputParser()`, mise en cache par configuration (la V1 la reconstruisait à chaque requête).|
|**`retrieve`** `(question, k)`|Appelle `search`.|
|**`generate`** `(question, hits, model, temperature, seed, history=None) → str`|Génère une réponse à partir de passages donnés (utilisé par l'évaluation et le DPO).|
|**`below_threshold`** `(hits, threshold) → bool`|Vrai si le seuil est actif et que le meilleur score est inférieur.|
|**`answer_question`** `(question, k, model, temperature, history=None, complements=False) → dict`|Point d'entrée : reformulation si historique, recherche, refus par seuil ou génération. Renvoie `answer`, `refused`, `refused_by_threshold`, `retrieved_sources`, `cited_sources`, `contexts`, `hits`, `model`, `standalone_question`, `complements`, `trace_id`.|

La distinction entre `retrieved_sources` (documents dont des passages ont été fournis) et `cited_sources` (documents réellement cités dans la réponse, et présents dans le contexte) corrige un défaut de la V1, qui présentait comme « sources » tous les documents récupérés, cités ou non.

## 8.3 Les vérifications déterministes

`check_answer(answer, contexts, retrieved_sources)` renvoie un objet `AnswerCheck` :

|Champ|Ce qui est vérifié|
|---|---|
|`refusal`, `clean_refusal`|La réponse est un refus (expressions reconnues par regex : « je ne trouve pas cette information », « le contexte ne mentionne pas »…) ; refus « net » s'il fait au plus 40 mots, c'est-à-dire sans réponse inventée à la suite.|
|`cited`, `cited_not_retrieved`|Documents cités (regex sur les noms `dbt_NN_…` / `airflow_NN_…`, extension normalisée) ; ceux qui ne faisaient pas partie du contexte.|
|`quotes`, `unsupported_quotes`|Passages entre guillemets (« », “ ”, " ") d'au moins 4 mots ; ceux qui ne figurent pas mot pour mot dans le contexte, après normalisation (casse, apostrophes, espaces, marqueurs Markdown).|
|`code_spans`, `unsupported_code`|Identifiants entre accents graves ; ceux absents du contexte (tolère `xcom_push()` pour `xcom_push` et `@task` pour `task`).|
|propriétés|`has_citation`, `citations_valid`, `quotes_faithful`, `code_support` (part des identifiants présents, 1 s'il n'y en a pas).|

Fonctions auxiliaires : `normalize`, `_strip_markdown`, `is_refusal`, `is_clean_refusal`, `cited_documents`, `extract_quotes`, `extract_code_spans`, `_in_contexts`, `_code_in_contexts`. La constante `REFUSAL_MESSAGE` est partagée avec le prompt.

> [!TIP]
> **POURQUOI DES RÈGLES PLUTÔT QU'UN JUGE**
> Ces contrôles sont exacts, reproductibles au bit près, instantanés et gratuits. Ils ne disent pas si la réponse est _bonne_, mais ils détectent de façon fiable les violations les plus graves pour un assistant documentaire : inventer une citation, citer une source qu'on n'a pas lue, inventer un nom de paramètre, refuser à tort. Ils servent deux fois : comme métriques d'évaluation et comme contraintes de la récompense DPO.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Sorties structurées** : demander une réponse JSON (`{"answer": …, "citations": [{"doc": …,` `"quote": …}]}`), imposée par le mode `format` d'Ollama. Plus facile à vérifier, mais les petits modèles perdent en qualité de rédaction sous contrainte de `format`.
>
> - **Citations par identifiant de passage** ([1], [2]) plutôt que par nom de document : plus précis (on sait quel passage justifie quoi).
>
> - **Vérification a posteriori et correction** : supprimer automatiquement les citations non vérifiées avant de renvoyer la réponse. Un garde-fou simple à ajouter dans l'API, qui masque le problème sans le corriger dans le modèle.
>
> - **Few-shot** : ajouter deux exemples de réponses modèles dans le prompt. Souvent efficace avec les petits modèles, au prix de tokens de contexte.
