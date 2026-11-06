# 14. Historique : de la V1 à la V2

La V1 fonctionnait et était bien documentée. Sa relecture critique a montré que plusieurs de ses conclusions n'étaient pas étayées par ses propres mesures. Cette relecture est la meilleure leçon du projet.

## 14.1 Ce que mesurait la V1

Configuration : corpus PDF généré depuis le HTML des sites, `all-MiniLM-L6-v2`, chunks fixes de 500 tokens (tiktoken) avec 50 de chevauchement, top-4, `llama3.1:8b` comme générateur _et_ comme juge, 15 questions dont une hors périmètre.

|Configuration|Chunks|Faithfulness|Answer relevancy|Context precision|
|---|---|---|---|---|
|500 / 50 (retenue)|126|0,818|0,581|0,987|
|300 / 30|221|0,748|0,629|0,963|

## 14.2 Ce que la relecture a montré

|Constat|Explication|Réponse en V2|
|---|---|---|
|**Modèle d'embeddings non multilingue**|Questions en français, corpus en anglais, modèle entraîné sur de l'anglais. L'answer relevancy basse avait été attribuée au seul juge ; le modèle d'embeddings, utilisé aussi par Ragas pour cette métrique, était un suspect au moins aussi sérieux, jamais testé.|`multilingual-e5-small` avec préfixes.|
|**« Retrieval excellent » sur- affirmé**|0,987 de context precision sur 10 documents, avec des questions qui nomment l'outil, ne prouve presque rien. La question hors périmètre obtenait même 1,0, signe que la métrique jugée par LLM n'était pas fiable ici.|Métriques de retrieval contre des preuves, sans LLM.|
|**Comparaison 500 vs 300 non concluante**|15 questions, une exécution par configuration ; des questions passaient de 1,0 à 0,0 d'un run à l'autre. Un écart de −0,07 est dans le bruit (intervalle d'environ ±0,2).|85 questions, intervalles de confiance, bootstrap apparié, répétitions.|
|**Refus correct compté comme échec**|La question hors périmètre valait 0 par construction et faisait passer la faithfulness moyenne de 0,877 à 0,818.|Refus mesuré à part ; Ragas sur le périmètre seulement.|
|**Corpus bruité**|Artefacts de navigation dans le texte, phrases coupées par l'extraction HTML → PDF → texte.|Corpus reconstruit depuis les sources.|
|**Étiquetage MLflow erroné possible**|Paramètres lus dans le `.env` au moment de l'évaluation, pas dans l'index réellement interrogé.|Configuration stockée dans l'index et relue.|
|**Juge = générateur**|Biais d'auto-évaluation reconnu dans le README de la V1.|Juge `qwen2.5:7b`.|
|**Troncature silencieuse possible**|Fenêtre de contexte Ollama par défaut non fixée.|`num_ctx = 8192`.|
|**Sources affichées = sources récupérées**|L'API présentait tous les documents récupérés comme sources.|Sources citées et sources récupérées distinguées.|

> [!TIP]
> **LA LEÇON**
> Une métrique n'est utile que si l'on sait ce qu'elle mesure, avec quelle précision, et ce qu'elle ne mesure pas. Avant d'interpréter un écart, il faut se demander s'il dépasse le bruit ; avant d'attribuer une cause, il faut avoir éliminé les autres suspects. C'est aussi la condition pour que le DPO ait un sens : on ne peut pas optimiser un signal qu'on ne sait pas mesurer.
