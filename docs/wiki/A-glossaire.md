# A. Glossaire

|Terme|Définition|
|---|---|
|Adaptateur (LoRA)|Petit ensemble de poids appris qui modifie un modèle gelé ; s'applique au modèle de base ou s'y fusionne.|
|ANN|Approximate Nearest Neighbors : recherche approchée des plus proches voisins.|
|Answer relevancy|Métrique Ragas : la réponse répond-elle à la question posée ? Mesurée via des questions régénérées.|
|Bi-encodeur|Modèle qui encode séparément la question et le passage ; permet l'indexation préalable.|
|Bootstrap|Rééchantillonnage avec remise pour estimer la variabilité d'une statistique.|
|BM25|Score de recherche lexicale fondé sur la fréquence des termes ; référence des moteurs de recherche classiques.|
|Chunk|Passage de document indexé et récupéré comme une unité.|
|Context precision|Métrique Ragas : les passages utiles sont-ils bien classés en tête ?|
|Cross-encodeur|Modèle qui lit ensemble la question et le passage pour noter leur pertinence ; utilisé en reranking.|
|DPO|Direct Preference Optimization : alignement sur des paires de préférences par une perte de classification, sans modèle de récompense explicite.|
|Embedding|Vecteur qui représente le sens d'un texte.|
|Faithfulness|Métrique Ragas : part des affirmations de la réponse étayées par le contexte.|
|Fil d'Ariane|Chemin de titres (« Document > Section > Sous-section ») placé en tête de chaque chunk.|
|GGUF, Q4_K_M|Format de modèle de llama.cpp / Ollama ; Q4_K_M est une quantification à environ 4 bits.|
|GHCR|GitHub Container Registry, registre d'images Docker lié au dépôt.|
|Grounding|Ancrage : la réponse découle-t-elle des sources ? Dans la récompense, mesuré par NLI.|
|HNSW|Graphe hiérarchique utilisé par ChromaDB pour la recherche approchée.|
|Hallucination|Affirmation plausible mais fausse ou non étayée produite par un LLM.|
|Intervalle de confiance|Plage plausible pour la vraie valeur d'une métrique, compte tenu de l'échantillon de questions.|
|LCEL|LangChain Expression Language : composition de composants par l'opérateur |.|
|MDX|Markdown enrichi de composants JSX/React, format du site de dbt.|
|MRR|Mean Reciprocal Rank : moyenne de 1 / rang du premier résultat pertinent.|
|NF4|Format de quantification 4 bits de QLoRA, adapté à la distribution des poids.|
|NLI|Natural Language Inference : implication, contradiction ou neutralité entre deux textes.|
|On-policy|Données produites par le modèle qu'on entraîne lui-même.|
|QLoRA|LoRA appliqué à un modèle chargé en 4 bits.|
|RAG|Retrieval-Augmented Generation : génération à partir de passages récupérés.|
|Reranking|Reclassement fin d'une liste de candidats, ici par un cross-encodeur.|
|Reward hacking|Le modèle optimise les failles de la récompense au lieu de l'objectif réel.|
|RLHF, PPO|Apprentissage par renforcement à partir de préférences humaines ; PPO est l'algorithme de renforcement classiquement utilisé.|
|RST|reStructuredText, format de la documentation d'Airflow (Sphinx).|
|Test apparié|Comparaison de deux versions sur les mêmes questions, à partir des différences question par question.|
|Token|Unité de texte manipulée par un modèle (morceau de mot).|
|TRL|Bibliothèque de Hugging Face pour l'apprentissage par préférences et par renforcement des LLM.|
