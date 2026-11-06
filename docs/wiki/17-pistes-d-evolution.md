# 17. Pistes d'évolution

Classées par rapport gain / effort. Chacune peut être mesurée avec le protocole existant.

|#|Évolution|Gain attendu|Effort|
|---|---|---|---|
|1|Mesurer la baseline V2 (`--repeats` 2)|**Fait**: résultats et lecture au chapitre 19.|Fait|
|2|Comparer 2-3 modèles d'embeddings et rerankers en `--retrieval-only`|Choix fondé sur des mesures, en quelques minutes.|Faible|
|3|Recherche hybride BM25 + vecteurs (fusion RRF)|Meilleur rappel sur les identifiants exacts.|Moyen|
|4|Annoter 30 réponses à la main|Mesurer l'accord du juge avec un humain ; calibrer la confiance dans Ragas.|Moyen|
|5|Calibrer et activer `RELEVANCE_THRESHOLD`|Calibré (section 19.4) : à ne pas activer en l'état.|Faible|
|6|Parent-child retrieval|Plus de contexte pour le LLM sans perdre en précision de recherche.|Moyen|
|7|Pipeline DPO complet (petit run, puis run complet)|Moins de citations inventées et de faux refus, si la baseline en montre.|En cours|
|8|Journaliser les questions de l'API|**Fait**: traces Langfuse (section 20.1).|Fait|
|9|Agrandir le corpus (sources, snapshots, Airflow TaskFlow…)|Couverture ; questions transverses.|Moyen|
|10|Interface Streamlit / Gradio, streaming|**Fait**: interface web de test (section 20.2).|Fait|
