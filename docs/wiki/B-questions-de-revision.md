# B. Questions de révision

Essaie de répondre avant de lire la correction.

### 1. Pourquoi mesurer le retrieval séparément de la génération ?

Parce qu'un RAG a deux points de défaillance. Une mauvaise réponse peut venir d'un passage non récupéré (le générateur n'y peut rien) ou d'un bon passage mal exploité. Sans mesure séparée, on ne sait pas quoi corriger. Le projet mesure le retrieval sans LLM, grâce aux preuves.

### 2. Pourquoi le modèle d'embeddings de la V1 posait-il problème ?

Il était entraîné sur de l'anglais alors que les questions sont en français et le corpus en anglais. La recherche cross-lingue n'était pas maîtrisée, et Ragas utilisait le même modèle pour calculer answer_relevancy.

### 3. À quoi servent les préfixes « query: » et « passage: » ?

Les modèles E5 ont été entraînés avec eux : ils indiquent si le texte est une question ou un document. Les omettre dégrade la qualité de la recherche.

### 4. Pourquoi deux étapes de recherche ?

Le bi-encodeur trie rapidement tout l'index mais grossièrement ; le cross-encodeur est précis mais trop lent pour tout l'index. On combine un rappel large (20 candidats) et un tri fin (4 retenus).

### 5. Que se passerait-il si l'on changeait de modèle d'embeddings sans réindexer, et comment le projet l'empêche-t-il ?

Les vecteurs des questions et des passages viendraient de deux espaces différents : résultats absurdes, sans erreur. `ensure_index_ready` compare le modèle enregistré dans l'index à celui configuré et lève `IndexNotReadyError` ; l'API réindexe au démarrage si l'index est obsolète.

### 6. Pourquoi mesurer la taille des chunks avec le tokenizer d'embeddings ?

Parce que c'est lui qui limite l'encodage (512 tokens pour E5) : au-delà, la fin du chunk est ignorée dans le vecteur. Un autre tokenizer donne des comptes différents.

### 7. Un écart de 0,07 de faithfulness entre deux configurations évaluées sur 15 questions est-il significatif ?

Non. Pour une métrique de l'ordre de 0,8 sur 15 questions, l'écart-type de la moyenne est d'environ 0,10, soit un intervalle d'environ ±0,2. Il faut plus de questions, un test apparié et idéalement des répétitions.

### 8. Pourquoi un test apparié est-il plus sensible qu'une comparaison de deux moyennes ?

Les deux versions sont évaluées sur les mêmes questions ; la difficulté propre à chaque question affecte les deux de la même façon et disparaît dans la différence. La variance restante est bien plus faible.

### 9. Qu'est-ce qu'un faux refus, et pourquoi ne compte-t-on que ceux où la preuve était dans le contexte ?

Un refus à une question du périmètre. Si la preuve n'avait pas été récupérée, refuser était le bon comportement : l'erreur vient du retrieval, pas du générateur.

### 10. Pourquoi exclure les questions hors périmètre de Ragas ?

Un refus correct n'a pas d'affirmations à vérifier : Ragas lui donne 0, ce qui pénalise le bon comportement et fait baisser les moyennes. Le refus est mesuré par `correct_refusal`.

### 11. Pourquoi utiliser un juge différent du générateur ?

Pour limiter le biais d'auto-évaluation : un modèle tend à juger favorablement ses propres formulations.

### 12. Quelle est la différence entre DPO et PPO ?

PPO entraîne d'abord un modèle de récompense puis optimise le LLM contre lui par renforcement, en générant pendant l'entraînement. DPO optimise directement une perte de classification sur des paires fixes, sans modèle de récompense ni génération pendant l'entraînement : plus simple et plus stable, mais limité aux paires disponibles.

### 13. Que représente β dans la perte DPO ?

L'intensité de la contrainte qui retient le modèle près du modèle de référence. Plus β est grand, moins le modèle s'en écarte.

### 14. Pourquoi la récompense n'utilise-t-elle pas un juge LLM ?

Un juge de 7-8B est bruité et biaisé ; optimiser contre lui pousse le modèle à exploiter ses failles (reward hacking). Les contraintes dures sont vérifiées par programme et ne peuvent pas être trompées ; le grounding utilise un modèle NLI dédié.

### 15. Pourquoi écarter une question d'entraînement dont le chunk source n'a pas été récupéré ?

On ne saurait pas si un refus serait correct : la paire serait ambiguë et pourrait apprendre au modèle à répondre sans source.

### 16. À quoi sert le filtre anti-fuite ?

À garantir que les questions d'entraînement ne sont pas des quasi-copies des questions d'évaluation. Sinon, une amélioration après DPO pourrait refléter de la mémorisation et non une meilleure capacité.

### 17. Pourquoi le gabarit de chat d'entraînement reproduit-il celui d'Ollama plutôt que celui de Hugging Face ?

Le modèle doit être entraîné sur le format exact qu'il recevra en production. Le gabarit Hugging Face de Llama 3.1 ajoute des lignes de date au message système ; Ollama non.

### 18. Pourquoi l'entraînement DPO ne peut-il pas tourner dans la CI ?

Depuis la version 1.15, TRL calcule les log-probabilités avec un noyau Triton, qui exige un GPU CUDA. Les runners gratuits de GitHub n'ont pas de GPU ; le test CPU vérifie tout ce qui précède l'entraînement.

### 19. Que vérifie le job CI « Corpus » ?

Qu'en reconstruisant le corpus depuis les sources figées, on obtient exactement les fichiers versionnés : le corpus est reproductible et n'a pas été modifié à la main.

### 20. Quel est le premier travail à faire avant de lancer le DPO, et pourquoi ?

Mesurer la baseline V2. Si les métriques que le DPO vise (citations, faux refus, faithfulness) sont déjà au plafond, l'entraînement n'a rien à corriger ; sinon, la baseline sert de référence pour la comparaison appariée.
