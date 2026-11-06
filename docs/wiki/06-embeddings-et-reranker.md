# 6. Embeddings et reranker

`ingestion/embeddings.py` (86 lignes) charge les deux modèles de sentence-transformers et expose des fonctions simples d'encodage et de scoring.

## 6.1 Fonctions

|Fonction|Rôle|
|---|---|
|**`get_embedding_model`** `()`|Charge SentenceTransformer(EMBEDDING_MODEL_NAME)une seule fois (`lru_cache`).|
|**`_encode`** `(texts) → vecteurs`|Encode avec normalisation L2 ; barre de progression au-delà de 20 textes.|
|**`embed_passages`** `(texts)`|Encode des chunks avec le préfixe `"passage: "`.|
|**`embed_queries`**,**`embed_query`**|Encode des questions avec le préfixe `"query: "`.|
|**`count_truncated`** `(texts) → int`|Nombre de passages plus longs que max_seq_lengthdu modèle.|
|**`get_reranker`** `()`|Charge le `CrossEncoder`(longueur max 512), ou Nonesi désactivé.|
|**`rerank_scores`**|Score de pertinence de chaque paire (question, passage).|

`(query, passages) → [float] | None` Les imports de `sentence_transformers` sont faits à l'intérieur des fonctions : le module s'importe sans PyTorch, ce qui permet aux tests et à la CI de tourner sans installer les modèles.

## 6.2 Choix et alternatives

> [!TIP]
> **POURQUOI CES DEUX MODÈLES**
> Ils sont multilingues, petits (ils tournent sur CPU, y compris dans le conteneur Docker), gratuits et locaux. `multilingual-e5-small` est un bon compromis taille/qualité parmi les modèles multilingues ouverts ; le cross-encodeur mMARCO est entraîné précisément sur une tâche question → passage, dans plusieurs langues.
>
> CE QU'ON AURAIT PU FAIRE
>
> - **Modèles plus grands** : `multilingual-e5-base` ou `-large`, `BAAI/bge-m3` (multilingue, 8 192 tokens de contexte, vecteurs denses et creux). Meilleurs, mais plus lents sur CPU. À tester avec `run_eval.py` `--retrieval-only`, qui mesure l'effet en quelques secondes.
>
> - **Reranker plus fort** : `BAAI/bge-reranker-v2-m3`, multilingue et plus précis, environ 570 M de paramètres.
>
> - **API commerciales** (OpenAI, Cohere, Voyage) : excellentes, mais payantes et non locales, contraires au cahier des charges.
>
> - **Traduire les questions en anglais** avant la recherche : contourne le problème cross-lingue avec un modèle anglais, au prix d'un appel LLM et d'erreurs de traduction sur les termes techniques.
>
> - **Fine-tuner le modèle d'embeddings** sur des paires (question, chunk) générées : gain possible sur le domaine, mais il faut beaucoup de paires et on risque le surapprentissage sur un corpus de 10 documents.
