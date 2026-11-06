# 5. Chunking

`ingestion/chunking.py` (261 lignes) découpe les 10 documents en passages par sections, chacun préfixé de son fil d'Ariane, et enregistre les paramètres qui les ont produits.

## 5.1 Fonctionnement

L'algorithme (`CHUNKER_VERSION = "sections-v2"`) se déroule en trois temps :

1. **Découpage en sections** (`parse_sections`) : chaque titre Markdown ouvre une section ; le chemin de titres (« Apache Airflow — XComs > Custom XCom Backends ») est conservé. Les # à l'intérieur des blocs de code (commentaires Python) sont ignorés. La ligne `Source :` est extraite comme URL.

2. **Regroupement** : une petite section est fusionnée avec la précédente si elles partagent la même section de niveau 2 et si le total tient dans la taille maximale. Le titre de la sous-section fusionnée reste dans le texte. Cela évite des chunks de deux lignes, trop pauvres pour un bon vecteur.

3. **Recoupage** : une section plus longue que le budget est coupée par `RecursiveCharacterTextSplitter` (LangChain), qui essaie dans l'ordre les séparateurs paragraphe, bloc de code, ligne, phrase, mot ; avec un chevauchement de 50 tokens.

Chaque chunk commence par son fil d'Ariane, suivi d'une ligne vide :

```
Apache Airflow — XComs > Custom XCom Backends
The XCom system has interchangeable backends, and you can set which backend is
being used via the `xcom_backend` configuration option. …
```

Le fil d'Ariane rend le chunk compréhensible hors de son document (pour le LLM comme pour le lecteur) et injecte dans le vecteur le sujet de la section, souvent absent du texte lui-même.

![Figure 3. Chunking par sections (`chunk_document`).](assets/figure-3.png)

*Figure 3. Chunking par sections (`chunk_document`).*

### Mesurer la taille avec le bon tokenizer

La taille est mesurée avec le **tokenizer du modèle d'embeddings** (`get_token_counter`), pas avec celui d'OpenAI (`tiktoken cl100k_base`) comme en V1. Ce qui compte est ce que le modèle d'embeddings peut encoder : 512 tokens au maximum pour E5, au-delà le texte est tronqué. 400 tokens laissent de la place au préfixe `"passage: "` et au fil d'Ariane. Au moment de l'indexation, `count_truncated` compte les chunks qui dépasseraient malgré tout et affiche un avertissement.

### Traçabilité

`chunks.json` contient, en plus des chunks, les paramètres qui les ont produits (`chunking_params`) : empreinte SHA-256 du corpus (12 premiers caractères), version de l'algorithme, taille, chevauchement, tokenizer. Ces paramètres sont recopiés dans les métadonnées de l'index, puis dans MLflow. Chaque chunk porte un identifiant stable (`airflow_04_xcoms.md::chunk_003`), son document, son rang, sa section, l'URL source, son texte et sa taille en tokens.

## 5.2 Fonctions

|Fonction|Rôle|
|---|---|
|**`Chunk`**,**`Section`**|Structures de données (dataclasses). `Section`.bodyrenvoie le texte de la section.|
|**`get_token_counter`** `() → (str → int)`|Charge le tokenizer du modèle d'embeddings (sans le modèle), une seule fois.|
|**`load_pdf_text`** `(pdf_path) → str`|Extraction de texte PDF (pypdf), conservée pour pouvoir ajouter des PDF au corpus.|
|**`load_corpus_documents`** `(corpus_dir) → [{source, text}]`|Charge les `.md`(et `.pdf`) du corpus ; ignore les textes vides (PDF scannés).|
|**`parse_sections`** `(markdown) → (url, [`Section`])`|Découpe en sections avec chemins de titres, en ignorant le code.|
|**`chunk_document`** `(source, text, count_tokens,` `chunk_size, chunk_overlap) →` `[`Chunk`]`|Regroupement puis recoupage d'un document.|
|**`chunk_documents`** `(documents, count_tokens=None, …)`|Applique chunk_documentà tout le corpus.|
|**`corpus_fingerprint`** `(documents) → str`|Empreinte du corpus, pour détecter un index obsolète.|
|**`chunking_params`**,**`save_chunks`**, **`load_chunks`**|Paramètres de production ; écriture et lecture de `chunks.json`(l'ancien format en liste nue reste lisible).|

## 5.3 Choix et alternatives

> [!TIP]
> **POURQUOI UN CHUNKING PAR SECTIONS**
> La documentation technique est déjà découpée par ses auteurs en sections cohérentes : c'est le meilleur découpage disponible, et il ne coûte rien. L'itération de la V1 (300 tokens au lieu de 500) avait fait baisser la faithfulness, un indice que la coupe à taille fixe tronquait les explications, même si cette mesure était trop bruitée pour conclure (chapitre 14).

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Chunking sémantique** : couper là où la similarité entre phrases consécutives chute. Utile pour du texte sans structure (transcriptions, e-mails), inutile quand les titres existent.
>
> - **Parent-child (small-to-big)** : indexer de petits chunks pour une recherche précise, mais envoyer au LLM la section parente entière. Bon compromis précision/contexte ; c'est l'évolution la plus naturelle si les réponses manquent de contexte.
>
> - **Contextual retrieval** : faire écrire par un LLM, pour chaque chunk, une phrase qui le situe dans son document, et l'ajouter avant l'indexation. Le fil d'Ariane en est une version gratuite et déterministe.
>
> - **Late chunking** : encoder le document entier avec un modèle à long contexte, puis découper les vecteurs de tokens. Demande un modèle d'embeddings à long contexte.
>
> - **Propositions** : faire réécrire le texte en affirmations atomiques. Coûteux et risque de déformer la source.

> [!WARNING]
> **ATTENTION**
>
> Mesuré sur le corpus réel avec le tokenizer E5 : **207 chunks**, médiane de 311 tokens, maximum de 400, aucun chunk tronqué à l'encodage ; 202 font au moins 60 tokens et servent à générer les questions d'entraînement. Les tests utilisent un compteur approximatif injecté : ils vérifient la logique, pas la taille exacte en tokens E5.
