# 16. Limites et points non mesurés

## 16.1 Ce qui n'a pas encore été mesuré

- **Les résultats de la V2 sont mesurés** (chapitre 19), mais pour une seule configuration : aucune variante (autre modèle d'embeddings, sans reranker, autre taille de chunk) n'a encore été comparée à la baseline.

- **Le pipeline DPO après l'étape 2** (paires construites, chapitre 20) : entraînement sur GPU, export vers Ollama, réévaluation et comparaison appariée avec `baseline_v2`.

- **La fiabilité du juge** `qwen2.5:7b` sur ce domaine : il est différent du générateur, pas forcément meilleur. Sur un essai de deux questions relancé à l'identique, la faithfulness est passée de 0,75 à 1,0, et une réponse du juge est restée illisible aux deux répétitions de la baseline.

- **L'accord avec un humain** : aucune réponse n'a été annotée à la main.

## 16.2 Limites de conception

- **Corpus restreint** : 10 pages, un seul domaine. Les questions transverses (qui demandent plusieurs documents) sont peu représentées.

- **Taille du jeu** : 85 questions donnent des intervalles d'environ ±0,05 à ±0,10 sur les métriques binaires ; les petites améliorations resteront non significatives.

- **Pertinence non mesurée par la récompense DPO** : seulement l'ancrage et le respect des règles.

- **Questions d'entraînement synthétiques** : moins variées que de vraies questions d'utilisateurs.

- **Adaptateur QLoRA appliqué à une base Q4_K_M** : écart possible entre entraînement et inférence.

- **Convertisseurs de corpus spécialisés** : ajouter une page demande de relire la sortie.

- **API non sécurisée** (CORS ouvert, pas d'authentification) : acceptable en local uniquement.

- **Détection des refus par expressions régulières** : un refus formulé autrement ne serait pas reconnu.
