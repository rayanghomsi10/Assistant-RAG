# 9. API

`api/main.py` (164 lignes) expose le système en HTTP avec FastAPI.

## 9.1 Points d'entrée

|Route|Comportement|
|---|---|
|`GET /health`|`{"status": "ok" ou "degraded", "index_ready": bool, "model": …, "tracing": bool}`. Utilisée par le healthcheck Docker et par la CD.|
|`GET /`|Interface web de test (section 20.2).|
|`POST /query`|Entrée : `question` (non vide), `top_k` (1–20) ; depuis le chapitre 20, `history`, `conversation_id`, `complements`. Sortie : réponse, refus, sources citées et récupérées, modèle, latence ; depuis, `question` reformulée, passages, compléments, lien de trace.|

Erreurs : 422 si l'entrée est invalide (`question` vide, `top_k` hors de 1–20) ; 400 si la `question` ne contient que des espaces ; 503 si l'index n'est pas prêt ou si Ollama est injoignable ; 500 pour une erreur de génération. La fonction `lifespan` s'exécute au démarrage : si `chunks.json` existe et que l'index est absent ou obsolète, elle réindexe. Un conteneur neuf avec un volume vide devient ainsi opérationnel sans intervention.

```
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "Comment pousser une valeur dans un XCom ?"}'
```

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Streaming** (Server-Sent Events) : afficher la réponse au fil de la génération ; indispensable pour une interface de chat, inutile pour une API de test.
>
> - **Endpoints asynchrones** avec `ainvoke` : meilleure tenue en charge ; ici, le goulot est le LLM local, qui sert les requêtes en série.
>
> - **Sécurité** : CORS restreint (actuellement *), clé d'API, limitation de débit. Nécessaire avant toute exposition publique.
>
> - **Observabilité** : journaliser chaque `question`, les passages, la réponse et la latence (Langfuse, OpenTelemetry). Ces journaux seraient la meilleure source de questions réelles pour l'évaluation et le DPO.

**Interface** : un front Streamlit ou Gradio de quelques dizaines de lignes pour une démonstration.
