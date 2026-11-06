# 15. Récapitulatif des décisions

Toutes les décisions structurantes, avec l'alternative principale écartée et la raison.

|Sujet|Choix|Alternative principale|Raison du choix|
|---|---|---|---|
|**Source du corpus**|Fichiers sources MDX / RST à versions figées|HTML des sites (V1), build Sphinx/Docusaurus|Texte propre, structure conservée, reproductible au caractère près.|
|**Format du corpus**|Markdown|PDF (V1)|Le PDF n'apportait rien et dégradait le texte.|
|**Chunking**|Par sections + fil d'Ariane, recoupage au-delà de 400 tokens|Taille fixe (V1), sémantique, parent-child|La structure existe déjà ; gratuit, déterministe.|
|**Unité de taille**|Tokens du modèle d'embeddings|tiktoken cl100k (V1)|C'est la limite réelle d'encodage (512).|
|**Embeddings**|`multilingual-e5-small`|bge-m3, e5-large, API|Multilingue, CPU, gratuit ; à comparer via `--retrieval-only`.|
|**Recherche**|20 candidats cosinus → cross-encodeur → 4|Hybride BM25, HyDE|Gain de précision standard, coût faible.|
|**Base vectorielle**|ChromaDB persistante|Qdrant, pgvector, FAISS|Sans serveur ; métadonnées de collection pour la traçabilité.|
|**Intégrité de l'index**|Configuration stockée et vérifiée|Aucun contrôle (V1)|Empêche les résultats absurdes silencieux.|
|**Générateur**|`llama3.1:8b` local|API commerciale, Qwen 2.5|Gratuit, local ; base du DPO.|
|**Prompt**|Règles vérifiables, définition unique|Prompt libre, sortie JSON|Rend mesurable le respect des règles et réutilisable pour l'entraînement.|
|**Refus hors périmètre**|Par le LLM ; seuil optionnel calibré|Seuil obligatoire|Un faux rejet coûte une réponse ; le seuil n'est activé que s'il est calibré.|
|**Jeu d'évaluation**|85 questions avec preuves validées|15 questions écrites de mémoire (V1)|Ancrage vérifié, puissance statistique, retrieval mesurable.|
|**Métriques**|Règles + Ragas (juge distinct)|Ragas seul, juge = générateur (V1)|Signal exact là où c'est possible, sémantique ailleurs.|
|**Statistiques**|IC bootstrap, test apparié, répétitions|Moyennes brutes (V1)|Distinguer un effet du bruit.|
|**Méthode d'alignement**|DPO hors ligne|PPO / GRPO en ligne, SFT|Simple, stable, faisable sur un GPU gratuit.|
|**Récompense**|Vérifiable (règles + NLI)|Juge LLM|Résiste au reward hacking.|
|**Données DPO**|On-policy, questions synthétiques filtrées|Réponses d'un modèle plus fort|Corrige les erreurs réelles du modèle déployé.|
|**Entraînement**|QLoRA, gabarit Ollama, prompt identique|Fine-tuning complet|Tient sur 16 Go ; cohérence entraînement / inférence.|
|**Retour en production**|Adaptateur dans Ollama, fusion en repli|Servir avec vLLM / transformers|Reste dans la stack existante.|
|**Tests**|Sans modèle, dépendances injectées|Tests d'intégration lourds|Rapides, exécutables en CI gratuite.|
|**CI/CD**|GitHub Actions + GHCR|GitLab CI, Docker Hub|Intégré au dépôt, gratuit.|
