# 1. Introduction

Ce document décrit le projet **Assistant RAG & Évaluation d'IA générative** dans son état actuel (version 2, dépôt `rayanghomsi10/Assistant-RAG`). Il est écrit comme un cours : chaque composant est expliqué, justifié, et confronté aux alternatives qu'on aurait pu choisir.

## 1.1 Le problème

Un ingénieur data qui travaille avec dbt et Apache Airflow se pose en permanence des questions précises : « quelle est la matérialisation par défaut d'un modèle dbt ? », « pourquoi mes XComs disparaissent-ils au retry ? », « quand un Dag `@daily` est-il réellement déclenché ? ». Un modèle de langage généraliste répond à ces questions avec aplomb, mais pas toujours correctement : sa connaissance est figée à sa date d'entraînement, il ne sait pas quelle version de l'outil vous utilisez, et il peut inventer une option qui n'existe pas.

Le projet construit un assistant qui répond **uniquement à partir de la documentation officielle**, cite ses sources, et dit « je ne trouve pas cette information » quand la réponse n'est pas dans les documents. Surtout, il **mesure** si l'assistant y parvient, avec un protocole d'évaluation conçu pour distinguer une vraie amélioration du bruit statistique. Enfin, il propose une boucle d'amélioration du modèle générateur par **DPO**, fondée sur des critères vérifiables plutôt que sur l'avis d'un autre modèle.

## 1.2 Le projet en une page

Le système se décompose en quatre chaînes qui partagent les mêmes briques :

1. **Construction de la base de connaissances** : téléchargement des sources de 10 pages de documentation (5 dbt, 5 Airflow) à des versions figées, conversion en Markdown propre, découpage en passages (chunks) par sections, vectorisation, indexation dans ChromaDB.

2. **Inférence** : pour une question, recherche des 20 passages les plus proches, reclassement par un crossencoder, conservation des 4 meilleurs, génération de la réponse par Llama 3.1 8B (via Ollama) avec un prompt qui impose l'ancrage, les citations et le refus explicite. Exposée par une API FastAPI.

3. **Évaluation** : 85 questions (73 dans le périmètre, 12 hors périmètre) avec des preuves recopiées du corpus ; métriques de retrieval et de génération calculées sans juge, métriques Ragas avec un juge distinct du générateur, intervalles de confiance par bootstrap, comparaison appariée entre deux versions, suivi dans MLflow.

4. **Alignement** : génération de questions d'entraînement, réponses produites par le modèle actuel, notation par une récompense vérifiable, constitution de paires de préférences, entraînement DPO avec QLoRA sur GPU (Colab ou Kaggle), réimport dans Ollama, réévaluation.

Autour : 66 tests automatisés qui tournent sans modèle ni GPU, une image Docker, et une CI/CD GitHub Actions qui vérifie le code, le jeu d'évaluation et la reproductibilité du corpus, puis publie l'image.

> [!IMPORTANT]
> **ÉTAT RÉEL DU PROJET**
>
> Le code de la V2 est complet et testé. **La baseline V2 est mesurée** (chapitre 19) ; observabilité, interface, mémoire de conversation et compléments ont été ajoutés ensuite (chapitre 20). Le pipeline DPO a produit un premier jeu de paires, revu puis regénéré (chapitre 20). Les chiffres de la V1 (chapitre 14) ne sont pas comparables à ceux de la V2. Un chiffre non mesuré n'apparaît nulle part comme un résultat.

## 1.3 Comment lire ce document

La **partie I** pose les notions. Si tu les maîtrises, passe directement à la **partie II**, qui donne l'architecture d'ensemble et la configuration. La **partie III** détaille chaque module : son rôle, son fonctionnement, la liste de ses fonctions, les choix faits et leurs alternatives. La **partie IV** prend du recul : historique V1 → V2, tableau des décisions, limites, pistes d'évolution et mode d'emploi. Les annexes regroupent le glossaire, la référence des métriques et des questions de révision corrigées.

Trois types d'encadrés reviennent :

> [!TIP]
> **POURQUOI CE CHOIX**
> La justification d'une décision de conception.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
> Les alternatives crédibles, avec ce qu'elles auraient apporté et coûté.

> [!WARNING]
> **ATTENTION**
> Un piège, une limite ou un point non vérifié.
