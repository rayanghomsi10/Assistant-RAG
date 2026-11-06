# 7. Base vectorielle et recherche

`retrieval/vector_store.py` (207 lignes) indexe les chunks dans ChromaDB, vérifie que l'index correspond à la configuration courante, et implémente la recherche en deux étapes.

## 7.1 Fonctionnement

### Indexation

`index_chunks` lit `chunks.json`, encode les textes, supprime puis recrée la collection, et y enregistre les vecteurs, les textes et les métadonnées de chaque chunk (document, rang, `section`, URL). Les métadonnées **de la collection** reçoivent la configuration d'indexation, préfixée par `index_` : modèle d'embeddings, préfixes, nombre de chunks, nombre de chunks tronqués, date, empreinte du corpus et paramètres de chunking.

### Contrôles

- `ensure_index_ready`, appelée à chaque recherche, lève `IndexNotReadyError` si l'index est vide, s'il a été construit par une ancienne version (pas de métadonnées), ou avec un autre modèle d'embeddings que celui configuré. Sans ce contrôle, changer de modèle sans réindexer donnerait des résultats absurdes _sans aucune erreur_ : les vecteurs de deux modèles différents ne sont pas comparables.

- `index_is_stale` compare l'index à `chunks.json` (empreinte du corpus, nombre de chunks, paramètres). L'API s'en sert au démarrage pour réindexer automatiquement.

### Recherche

`search(query, k=4, candidates=20, rerank=True)` récupère 20 candidats par similarité cosinus, les fait noter par le cross-encodeur, les retrie et renvoie les 4 meilleurs. Chaque résultat contient `id`, `text`, `source`, `section`, `url`, `similarity` (cosinus), `rerank_score` et `score` (le critère de tri : `score` du reranker s'il est actif, cosinus sinon). Ce `score` sert aussi au seuil de refus optionnel.

## 7.2 Fonctions

|Fonction|Rôle|
|---|---|
|**`IndexNotReadyError`**|Exception : index absent ou incompatible. L'API la traduit en HTTP 503.|
|**`get_client`**,**`get_collection`**|Client persistant ChromaDB (créé une fois, import paresseux) ; collection en espace cosinus.|
|**`get_index_metadata`** `() → dict`|Configuration enregistrée lors de l'indexation.|
|**`index_chunks`** `(chunks_path) → int`|Reconstruit l'index de zéro.|
|**`ensure_index_ready`** `()`|Vérifie présence, métadonnées et modèle d'embeddings.|
|**`index_is_stale`** `(chunks_path) → bool`|Index vide ou différent de chunks.json/ du modèle.|
|**`search`**|Recherche vectorielle puis reranking. Alias similarity_searchpour la|
|`(query, k, candidates, rerank) →` `[hit]`|compatibilité V1.|
|**`main`** `()`|Indexe et affiche les résultats de 4 questions témoins, dont une hors corpus.|

## 7.3 Choix et alternatives

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Recherche hybride BM25 + vecteurs** : BM25 (recherche lexicale) excelle sur les identifiants exacts (`on_schema_change`, `xcom_pull`) que les embeddings rapprochent mal ; on fusionne les deux classements (Reciprocal Rank Fusion). Souvent le meilleur gain pour de la documentation technique. Limite ici : BM25 ne fait pas le pont français → anglais, sauf sur les identifiants, mais c'est justement là qu'il est utile.
>
> - **Autres bases** : Qdrant (serveur, filtres riches, recherche hybride native), pgvector (si une base Postgres existe déjà), FAISS (bibliothèque en mémoire, très rapide, sans persistance de métadonnées), LanceDB (fichiers, sans serveur). À cette échelle, le choix n'a aucun effet sur la qualité ; c'est une question d'exploitation.
>
> - **Réécriture de requête** : HyDE (faire générer une réponse hypothétique et chercher avec elle), multi-query (plusieurs reformulations), décomposition des questions composées. Ajoute un appel LLM par question.
>
> - **MMR** (Maximal Marginal Relevance) : diversifier les 4 passages pour éviter quatre chunks redondants.
>
> - **GraphRAG** : extraire un graphe d'entités et de relations (Dag → tâche → opérateur) et y naviguer. Pertinent pour des questions transverses (« quels mécanismes permettent à deux tâches de communiquer ? »), lourd à construire et à évaluer.
