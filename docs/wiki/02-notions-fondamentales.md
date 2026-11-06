# 2. Notions fondamentales

Ce chapitre est un cours condensé. Chaque notion est présentée au niveau nécessaire pour comprendre les choix du projet, avec un renvoi vers le module qui l'utilise.

## 2.1 Modèle de langage, hallucination et fenêtre de contexte

Un **grand modèle de langage** (LLM) est un réseau de neurones (un transformer) entraîné à prédire le jeton suivant d'un texte. Un **jeton** (token) est un morceau de mot : « matérialisation » fait plusieurs jetons, « dbt » un ou deux. Le modèle génère une réponse jeton par jeton, chaque jeton étant tiré d'une distribution de probabilité. La **température** règle cette distribution : à 0, on prend toujours le jeton le plus probable (sortie quasi déterministe) ; plus haut, on explore davantage (sortie variée). Le projet génère à température 0 pour l'inférence et l'évaluation, et à 0,8 pour produire des réponses variées lors de la constitution des paires DPO.

Une **hallucination** est une affirmation fluide et plausible mais fausse ou non étayée. Elle vient du fait que le modèle optimise la vraisemblance du texte, pas sa vérité. Elle est particulièrement fréquente sur les détails précis : noms de paramètres, valeurs par défaut, numéros de version.

La **fenêtre de contexte** est le nombre maximal de jetons que le modèle peut lire en une fois (prompt + réponse). Llama 3.1 accepte jusqu'à 128 000 jetons, mais **Ollama** n'en alloue par défaut que quelques milliers, selon la version. Au-delà, le début du prompt est tronqué _sans erreur_. Avec un prompt système, 4 passages de 400 jetons et la question, on approche de ces limites ; c'est pourquoi le projet fixe explicitement `num_ctx = 8192` (variable `OLLAMA_NUM_CTX`).

**Quantification** : stocker les poids du modèle sur moins de bits (4 au lieu de 16) pour réduire la mémoire. `llama3.1:8b` dans Ollama est quantifié au format GGUF `Q4_K_M` : environ 5 Go au lieu de 16, au prix d'une légère perte de qualité. C'est ce qui permet de faire tourner un modèle de 8 milliards de paramètres sur un GPU de 8 Go.

## 2.2 Le RAG : génération augmentée par la recherche

Le **RAG** (Retrieval-Augmented Generation) consiste à chercher, au moment de la question, les passages de documents pertinents et à les insérer dans le prompt. Le modèle ne répond plus « de mémoire » mais en lisant une source. Trois bénéfices : la connaissance est à jour (il suffit de mettre à jour les documents), la réponse est vérifiable (on peut citer le passage), et on peut demander au modèle de refuser quand les passages ne contiennent pas la réponse.

Un RAG a deux points de défaillance distincts, et toute la conception de l'évaluation en découle :

- **Le retrieval** peut ne pas ramener le bon passage. Le meilleur générateur du monde ne peut alors pas répondre correctement.

- **La génération** peut mal exploiter un bon passage : réponse incomplète, détail inventé, citation déformée, refus alors que la réponse était là.

Mesurer uniquement la réponse finale ne permet pas de savoir lequel des deux a échoué. Le projet mesure donc le retrieval seul (sans LLM), puis la génération.

## 2.3 Embeddings et similarité cosinus

Un **embedding** est un vecteur de nombres réels (384 dimensions pour le modèle utilisé) qui représente le sens d'un texte. Un modèle d'embeddings est entraîné pour que deux textes de sens proche aient des vecteurs proches. La proximité se mesure par la **similarité cosinus** :

cos(u, v) = (u · v) / (‖u‖ · ‖v‖)

Elle vaut 1 pour deux vecteurs de même direction, 0 pour deux vecteurs orthogonaux. Si les vecteurs sont normalisés (norme 1), la similarité cosinus est simplement leur produit scalaire ; le projet normalise systématiquement (`normalize_embeddings=True`). ChromaDB, configurée en espace `cosine`, renvoie une _distance_ égale à 1 − similarité ; le code la reconvertit en similarité.

Un modèle d'embeddings **multilingue** place une phrase française et sa traduction anglaise à des positions voisines. C'est indispensable ici : les questions sont en français, le corpus en anglais. Le modèle de la V1 (allMiniLM-L6-v2) avait été entraîné sur de l'anglais ; la recherche cross-lingue fonctionnait en partie par chance, sur les mots partagés (`xcom_push`, « dbt »). La V2 utilise `intfloat/multilingual-e5-small`.

Les modèles de la famille **E5** ont été entraînés avec des préfixes : `"query: "` devant les questions, `"passage: "` devant les documents. Les omettre dégrade nettement leurs performances. C'est une asymétrie volontaire : une question et le passage qui y répond ne se ressemblent pas, le préfixe indique au modèle quel rôle joue chaque texte.

## 2.4 Index vectoriel et recherche approchée

Trouver les k vecteurs les plus proches d'une question parmi N vecteurs demande N calculs en recherche exacte. Pour de grands corpus, on utilise une **recherche approchée** (ANN). ChromaDB utilise **HNSW** (Hierarchical Navigable Small World) : un graphe à plusieurs niveaux où chaque vecteur est relié à ses voisins ; la recherche descend les niveaux en suivant les arêtes vers la question. C'est très rapide, au prix d'une petite probabilité de manquer un voisin exact.

Avec quelques centaines de chunks, ce projet n'a pas besoin d'ANN : une recherche exacte (un produit matriciel numpy) serait instantanée. ChromaDB est utilisée pour sa persistance sur disque, ses métadonnées et la facilité de passer à l'échelle, pas pour sa vitesse.

## 2.5 Le chunking

On n'indexe pas des documents entiers : un document de 7 000 mots dilue son sens dans un seul vecteur, et ne tiendrait pas dans le prompt. On le découpe en **chunks** (passages). La taille est un compromis :

- trop petits, ils coupent les explications en deux, et le modèle reçoit des fragments sans le contexte nécessaire ;

- trop grands, ils mélangent plusieurs sujets (vecteur moins précis) et consomment la fenêtre de contexte ;

- au-delà de la longueur maximale du modèle d'embeddings (512 jetons pour E5), le texte est **tronqué silencieusement** à l'encodage : la fin du chunk n'influence pas son vecteur.

Le **chevauchement** (overlap) répète la fin d'un chunk au début du suivant, pour qu'une phrase coupée existe entière dans au moins un chunk. Le chunking peut être à taille fixe (simple, aveugle à la structure), structurel (suit les titres et paragraphes) ou sémantique (coupe là où le sens change, mesuré par embeddings). La V2 fait du chunking structurel par sections, avec un repli à taille fixe pour les sections trop longues (chapitre 5).

## 2.6 Bi-encodeur et cross-encodeur : le reranking

Un modèle d'embeddings est un **bi-encodeur** : la question et chaque passage sont encodés _séparément_, puis comparés. C'est ce qui permet de pré-calculer les vecteurs des passages et de chercher vite. Mais la comparaison est grossière : les deux textes ne se « voient » jamais.

Un **cross-encodeur** lit la paire (question, passage) _ensemble_, dans un seul passage du transformer, et produit un score de pertinence. Il est bien plus précis, mais il faut l'exécuter pour chaque paire : impossible sur tout le corpus. D'où la recherche en deux étapes, standard en production :

![Figure 1. Recherche en deux étapes (`retrieval/vector_store.py`, fonction `search`).](assets/figure-1.png)

*Figure 1. Recherche en deux étapes (`retrieval/vector_store.py`, fonction `search`).*

## 2.7 Prompting et ancrage

Le **prompt système** fixe les règles du modèle. Celui du projet impose : répondre uniquement à partir du contexte, refuser avec une phrase exacte si l'information manque, citer les documents par leur nom entre crochets, n'utiliser des guillemets que pour une citation mot pour mot, écrire les identifiants de code exactement comme dans le contexte. Chaque règle est **vérifiable par un programme** (chapitre 8) : c'est ce qui permet de mesurer le respect des règles sans juge et, plus tard, d'en faire une récompense pour l'entraînement.

Un prompt n'est qu'une demande : un modèle de 8B ne la respecte pas toujours. Le DPO (section 2.10) sert précisément à transformer ces règles en comportement appris.

## 2.8 Évaluer un système RAG

On peut évaluer à trois niveaux, du plus objectif au plus subjectif :

1. **Retrieval, contre une vérité terrain** : si l'on sait quel passage contient la réponse, on vérifie s'il a été récupéré et à quel rang. Le projet stocke pour chaque question des _preuves_ recopiées du corpus ; un chunk « contient la preuve » s'il en contient le texte.

2. **Génération, par règles** : refus, sources citées, citations textuelles, identifiants de code. Déterministe, reproductible, gratuit.

3. **Génération, par un LLM juge** : le framework **Ragas** demande à un LLM de décomposer la réponse en affirmations et de vérifier chacune contre le contexte (_faithfulness_), de régénérer des questions à partir de la réponse et de mesurer leur similarité avec la question posée (_answer relevancy_), ou de juger la pertinence de chaque passage récupéré (_context precision_). Plus riche sémantiquement, mais bruité et biaisé, surtout avec un petit juge local.

Le **MRR** (Mean Reciprocal Rank) vaut 1/rang du premier résultat pertinent, moyenné sur les questions : 1 si le bon passage est premier, 0,5 s'il est deuxième, 0 s'il est absent.

Un piège classique est le **biais d'auto-évaluation** : si le juge est le même modèle que le générateur, il tend à valider ses propres formulations. La V1 utilisait le même modèle pour les deux rôles ; la V2 utilise `qwen2.5:7b` comme juge, d'une autre famille que Llama.

## 2.9 Statistiques : intervalle de confiance, bootstrap, test apparié

Une moyenne calculée sur 15 ou 85 questions est une **estimation** : avec d'autres questions du même type, on aurait obtenu un autre chiffre. L' **intervalle de confiance à 95 %** donne une plage plausible pour la vraie valeur. Pour une métrique binaire (0 ou 1) de moyenne p sur n questions, l'écart-type de la moyenne vaut environ √(p(1−p)/n) : avec p = 0,8 et n = 15, cela fait 0,10, donc un intervalle d'environ ±0,20. Un écart de 0,07 entre deux configurations évaluées sur 15 questions ne dit donc rien.

Le **bootstrap** estime cet intervalle sans hypothèse sur la distribution : on tire avec remise n questions parmi les n, on recalcule la moyenne, on répète 2 000 fois, et on prend les quantiles 2,5 % et 97,5 % des moyennes obtenues (méthode des percentiles).

Pour **comparer deux versions**, il faut les évaluer sur les mêmes questions et raisonner sur les _différences question par question_ : c'est le **test apparié**. Les questions difficiles le sont pour les deux versions ; travailler sur les différences élimine cette variabilité commune et rend le test beaucoup plus sensible qu'une comparaison de deux moyennes. Le projet applique un bootstrap sur les différences : si l'intervalle de la différence exclut 0, l'écart est significatif.

Il existe une seconde source de variance : à juge et générateur fixés, relancer l'évaluation ne donne pas exactement les mêmes scores (échantillonnage du juge, non-déterminisme du GPU). L'option `--repeats` mesure cette variance entre exécutions.

## 2.10 Alignement par préférences : RLHF, PPO, DPO

Après le pré-entraînement et le fine-tuning supervisé, on **aligne** un modèle sur des préférences : préférer la réponse A à la réponse B pour un même prompt. L'approche historique, le **RLHF** (Reinforcement Learning from Human Feedback), entraîne d'abord un _modèle de récompense_ sur des paires annotées, puis optimise le LLM contre cette récompense avec **PPO**, un algorithme d'apprentissage par renforcement, en le pénalisant (divergence KL) s'il s'éloigne trop du modèle d'origine. C'est puissant mais lourd : quatre modèles en mémoire (politique, référence, récompense, critique), réglage délicat, instabilité.

**DPO** (Direct Preference Optimization, Rafailov et al., 2023) montre qu'on peut obtenir la même solution optimale sans modèle de récompense ni boucle de renforcement, avec une simple perte de classification sur les paires (y<sub>w</sub> préférée, y<sub>l</sub> rejetée) :

```
L = − log σ(β [ (log πθ(yw|x) − log πref(yw|x)) − (log πθ(yl|x) − log πref(yl|x)) ])
```

πθ est le modèle entraîné, πref le modèle de référence (gelé), σ la sigmoïde, β règle l'écart autorisé par rapport à la référence. Intuitivement, la perte pousse le modèle à augmenter la probabilité relative de la réponse préférée et à diminuer celle de la réponse rejetée, _par rapport à ce que faisait la référence_. La quantité β(log πθ − log πref) s'interprète comme une « récompense implicite » ; TRL journalise son exactitude (part des paires où la préférée l'emporte) et sa marge.

Le DPO reste une méthode d'apprentissage par préférences : la qualité du résultat dépend entièrement de la qualité des paires. Si la réponse « préférée » n'est pas réellement meilleure, le modèle apprend le mauvais comportement. C'est pourquoi le projet construit les paires à partir d'une **récompense vérifiable** plutôt que d'un juge LLM (chapitre 11).

## 2.11 LoRA et QLoRA

Ré-entraîner les 8 milliards de paramètres d'un modèle demande des dizaines de Go de mémoire GPU. **LoRA** (Low-Rank Adaptation, Hu et al., 2021) gèle les poids d'origine W0 et n'apprend qu'une correction de faible rang :

W = W0 + (α / r) · B A, avec A ∈ ℝ<sup>r×d</sup> , B ∈ ℝ<sup>d×r</sup> , r ≪ d

Avec r = 16 sur des matrices de dimension 4 096, on entraîne une fraction de pour cent des paramètres. L'adaptateur obtenu (quelques dizaines de Mo) s'applique au modèle de base ou s'y fusionne.

**QLoRA** (Dettmers et al., 2023) charge en plus le modèle de base en 4 bits (format NF4, adapté à la distribution des poids) et calcule en 16 bits. Un modèle de 8B tient alors dans environ 6 Go, ce qui rend l'entraînement possible sur un GPU T4 de 16 Go (Colab gratuit). Avantage supplémentaire avec DPO : le modèle de référence est simplement le même modèle _adaptateurs désactivés_, sans seconde copie en mémoire.

## 2.12 NLI : vérifier qu'une phrase découle d'un texte

L' **inférence en langage naturel** (NLI) classe une paire (prémisse, hypothèse) en _implication_, _contradiction_ ou _neutre_. Un modèle NLI répond donc à la question « cette phrase de la réponse découle-t-elle de ce passage ? ». C'est une version spécialisée, légère et déterministe de ce que fait la faithfulness de Ragas avec un LLM. Le projet utilise un modèle NLI multilingue (mDeBERTa-v3, entraîné sur XNLI et d'autres jeux) dans la récompense DPO : les phrases de la réponse sont en français, les passages en anglais.
