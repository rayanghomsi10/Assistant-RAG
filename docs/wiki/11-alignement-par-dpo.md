# 11. Alignement par DPO

Le dossier `training/` améliore le générateur par apprentissage de préférences. Le pipeline compte quatre étapes, dont trois tournent en local avec Ollama et une sur GPU (Colab ou Kaggle).

## 11.1 Objectif et principe

Le DPO vise les défauts du générateur que l'on sait mesurer : citations entre guillemets qui ne sont pas verbatim, documents cités hors contexte, identifiants de code inventés, refus à tort, réponse au lieu de refus hors périmètre, réponses peu ancrées. Le principe : faire produire au modèle actuel plusieurs réponses à une même question, les noter avec une **récompense vérifiable**, et lui apprendre à préférer la meilleure.

![Figure 4. Construction d'une paire de préférences (`build_preferences.py`).](assets/figure-4.png)

*Figure 4. Construction d'une paire de préférences (`build_preferences.py`).*

## 11.2 Étape 1 : les questions d'entraînement

`generate_questions.py` demande au modèle juge (`qwen2.5:7b`, température 0,7, sortie JSON) de rédiger 2 questions en français par chunk d'au moins 60 tokens, en gardant l'identifiant du chunk source. S'y ajoutent les 60 questions hors périmètre de `out_of_scope_train.json`, différentes de celles de l'évaluation. Puis :

- **déduplication** (`dedupe`) : questions trop proches entre elles (Jaccard sur les mots porteurs de sens ≥ 0,8) ;

- **filtre anti‑fuite** (`LeakageFilter`) : toute question trop proche d'une question d'évaluation est écartée, par recouvrement lexical (Jaccard ≥0,5) ou par similarité d'embeddings au‑delà d'un seuil calibré sur le jeu d'évaluation (0,949 ici ; un seuil fixe de 0,88 écartait presque tout, section 19.5).

Fonctions : `content_tokens`, `jaccard`, `product_of`, `generate_for_chunk`, `LeakageFilter.keep`, `dedupe`, `main`. Sortie : `training/data/questions.jsonl`.

## 11.3 Étape 2 : la récompense et les paires

`reward.py` définit `compute_reward(answer, question_type, hits, source_document, nli)` :

|Cas|Récompense|
|---|---|
|Hors périmètre|0 si le modèle répond (`answered_out_of_scope`) ; 1 pour un refus net ; 0,5 pour un refus suivi d'un développement.|
|Périmètre, contraintes dures|0 en cas de refus (`false_refusal`, le chunk source était dans le contexte), réponse de moins de 12 mots (`too_short`), document cité hors contexte, citation inventée.|
|Périmètre, composantes|Moyenne pondérée : grounding NLI 0,40 ; document source cité 0,20 ; identifiants de code présents 0,15 ; au moins une citation 0,10 ; concision 0,15 (1 jusqu'à 180 mots, décroît jusqu'à 0 à 400). Les poids sont renormalisés si le NLI est désactivé.|

Le **grounding** (`grounding_score`) découpe la réponse en phrases (`answer_sentences`), calcule pour chacune la probabilité d'implication par le meilleur passage du contexte (`NliScorer.entailment`), et fait la moyenne. Le modèle NLI est chargé une fois (`get_nli_scorer`).

`build_preferences.py` applique la récompense : pour chaque question, il recherche les passages ; si c'est une question du périmètre dont le chunk source n'a _pas_ été récupéré, il l'écarte (on ne saurait pas si un refus serait correct) ; il génère une réponse greedy et 4 échantillonnées (`sample_answers`, doublons exacts retirés), les note, et forme une paire si la meilleure réponse valide dépasse la moins bonne d'au moins 0,2 (`select_pair`). Sortie : `preferences.jsonl` (prompt complet, chosen, rejected, détails des récompenses) et `preferences_stats.json` (questions écartées, motifs de rejet, questions sans paire).

> [!TIP]
> **POURQUOI UNE RÉCOMPENSE VÉRIFIABLE**
> Entraîner contre un juge LLM de 7-8B, c'est optimiser un signal bruité et biaisé : le modèle apprend à plaire au juge plutôt qu'à mieux répondre (_reward hacking_). Les contraintes dures ne peuvent pas être trompées : une citation est dans le contexte ou ne l'est pas. Le grounding utilise un modèle NLI dédié, distinct du générateur et du juge d'évaluation. Et l'évaluation finale emploie Ragas, que l'entraînement n'a jamais vu : une amélioration qui n'apparaîtrait que sur les métriques de la récompense serait suspecte.

> [!TIP]
> **POURQUOI DES DONNÉES ON-POLICY**
> Les réponses viennent du modèle qu'on veut améliorer, dans les conditions réelles (mêmes passages, même prompt). Le DPO corrige alors ses erreurs effectives, pas celles d'un autre modèle. Des paires construites avec un modèle plus fort comme « chosen » apprendraient un style que le modèle 8B ne sait pas reproduire.

## 11.4 Étape 3 : l'entraînement

`train_dpo.py`, lancé depuis `dpo_colab.ipynb` sur un GPU de 16 Go :

- **modèle de base** : `meta-llama/Llama-3.1-8B-Instruct`, le même que `llama3.1:8b` dans Ollama, pour que l'adaptateur s'applique au modèle déployé (accès Hugging Face à demander) ;

- **QLoRA** : chargement en 4 bits NF4 avec double quantification ; LoRA r = 16, α = 32, dropout 0,05, sur les 7 projections (`q, k, v, o, gate, up, down`) ; calcul en bf16 si le GPU le supporte nativement (Ampere et suivants), sinon fp16 (T4) ;

- **DPO** (TRL) : β = 0,1, taux d'apprentissage 5·10<sup>−6</sup> , planification cosinus avec 10 % de warm-up, 1 époque, batch effectif 8 (1 × 8 d'accumulation), gradient checkpointing, 10 % des paires en validation ;

- **gabarit de chat** : celui qu'Ollama applique à Llama 3.1 (`OLLAMA_LLAMA3_CHAT_TEMPLATE`), et non celui de Hugging Face, qui ajoute des lignes de date au message système : on entraîne sur le format exact de l'inférence ;

- **longueur** : les paires de plus de 3 072 tokens sont écartées, pas tronquées (une troncature couperait la fin de la réponse, c'est-à-dire ce qu'on optimise).

Fonctions : `load_pairs`, `sequence_length`, `split_pairs` (découpage reproductible), `prepare_tokenizer`, `build_trainer` (assemble `DPOConfig` et `DPOTrainer`, utilisé aussi par le test), `summarize_logs`, `parse_args`, `main`. Sortie : `training/outputs/dpo-lora/` (adaptateur, tokenizer, `training_summary.json` avec l'exactitude et la marge des récompenses implicites sur la validation).

## 11.5 Étape 4 : retour dans Ollama et décision

`export_ollama.py` propose deux voies :

- **adaptateur** (par défaut) : copie de l'adaptateur et Modelfile `FROM llama3.1:8b` + `ADAPTER ./adapter`, avec `num_ctx` et les jetons d'arrêt de Llama 3.1 ; puis `ollama create rag-assistant-dpo`. Rapide, mais l'adaptateur entraîné sur une base NF4 est appliqué à une base Q4_K_M ; la documentation d'Ollama prévient que le résultat peut s'écarter ;

- **fusion** (`--merge`) : fusion de l'adaptateur dans les poids fp16 (environ 16 Go, 32 Go de RAM conseillés : Kaggle oui, Colab gratuit non), puis `ollama create --quantize q4_K_M`. À utiliser si la première voie dégrade les résultats.

Fonctions : `adapter_modelfile`, `merged_modelfile`, `export_adapter`, `export_merged`, `main`.

```
FROM llama3.1:8b
ADAPTER ./adapter
PARAMETER num_ctx 8192
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
```

> [!IMPORTANT]
> **RÈGLE DE DÉCISION**
> Le modèle DPO n'est adopté que si `compare_runs.py` montre une amélioration significative d'au moins une métrique visée (`quotes_faithful`, `faithfulness`, `false_refusal`, `citations_valid`) **et** aucune dégradation significative de `answer_relevancy` ni de `correct_refusal`. Avant tout entraînement, si la baseline montre ces métriques déjà au plafond, le DPO n'a rien à corriger : on s'arrête.

> [!WARNING]
> **ATTENTION**
>
> - La récompense ne mesure pas si la réponse _répond_ à la question. Risque : un modèle qui apprend des réponses fidèles mais évasives. Garde-fous : longueur minimale, answer_relevancy dans la règle de décision, lecture de 30 paires avant d'entraîner.
>
> - **Les étapes 1 et 2 ont été exécutées, puis la sélection revue** (20.6). En CI, l'entraînement ne peut pas tourner : depuis la version 1.15, TRL calcule les log‑probabilités avec un noyau Triton qui exige un GPU CUDA. Le test CPU vérifie ce qui précède (données, gabarit, tokenisation, LoRA). Le premier run Colab, avec `--limit 20`, est le vrai test.
>
> - Sur T4 (fp16), TRL convertit les adaptateurs en bf16 pour les modèles quantifiés : c'est le point le plus susceptible de poser problème.

## 11.6 Choix et alternatives

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Ne pas entraîner du tout** : améliorer le prompt (few-shot), ajouter un post-traitement qui supprime les citations non vérifiées, changer de générateur (Qwen 2.5 7B, Mistral). Moins coûteux, à essayer d'abord si la baseline révèle peu de violations.
>
> - **SFT** (fine-tuning supervisé) sur les meilleures réponses seulement (rejection sampling / best-of-n) : plus simple que DPO, mais n'apprend pas explicitement ce qu'il ne faut pas faire.
>
> - **Variantes de DPO** : IPO (plus robuste au surapprentissage sur des paires faciles), KTO (accepte des réponses notées bonnes/mauvaises sans paires), ORPO (combine SFT et préférences sans modèle de référence), SimPO (sans référence, normalisé par la longueur). Toutes disponibles dans TRL.
>
> - **RL en ligne** : PPO ou GRPO avec la même récompense vérifiable, appelée pendant l'entraînement. C'est l'approche des modèles de raisonnement récents ; plus puissante (le modèle explore), mais bien plus coûteuse en calcul et délicate à stabiliser.
>
> - **Bandit contextuel sur le retrieval** : apprendre quand augmenter top-k, activer le reranker ou réécrire la requête. Léger, mais gain faible sur un corpus de 10 documents.
>
> - **Unsloth** pour l'entraînement : plus rapide et plus économe en mémoire sur Colab, avec export GGUF intégré vers Ollama.
