# Alignement DPO du générateur

Procédure complète, de la baseline à la décision d'adopter (ou non) le modèle entraîné.
Les étapes 1 à 3 et 6 à 8 tournent sur ta machine (Ollama) ; l'étape 5 sur un GPU de 16 Go.

## 1. Baseline mesurée

```bash
python ingestion/chunking.py && python retrieval/vector_store.py
python evaluation/run_eval.py --run-name baseline_v2 --repeats 2
```

Regarder d'abord ce que le DPO peut corriger : `quotes_faithful`, `citations_valid`,
`false_refusal`, `faithfulness`. Si ces métriques sont déjà au plafond (intervalles de
confiance collés à 1), l'entraînement n'a rien à gagner : s'arrêter là.

## 2. Questions d'entraînement

```bash
python training/generate_questions.py --per-chunk 3
```

Environ 3 questions par chunk + 60 questions hors périmètre, après déduplication et filtre
anti-fuite (les questions écartées sont affichées). Relire une vingtaine de questions au hasard
dans `training/data/questions.jsonl` : une question mal posée produit une paire inutile.

## 3. Paires de préférences

```bash
python training/build_preferences.py --samples 4
```

Une réponse greedy + 4 échantillonnées (température 0,8) par question, notées par
`reward.py`. Compter environ 30 s par question sur une RTX 4060 (quelques heures pour 600
questions) ; `--limit 20` pour un essai, `--no-nli` pour aller plus vite (sans la composante de
grounding).

**Règle de sélection.** La paire oppose la meilleure réponse valide à la moins bonne qui laisse
un écart d'au moins `--min-margin` (0,2). Si cette réponse rejetée ne viole aucune contrainte
dure, l'écart doit aussi porter sur le fond (ancrage, code, concision) d'au moins
`--min-substance` (0,1), et pas seulement sur la citation du document. Sans cette condition
(`--no-substance`, règle du premier run), 48 paires sur 164 ne tenaient qu'à la citation, et la
relecture de 30 paires en a trouvé 8 où la réponse retenue était moins bonne sur le fond.

Toutes les réponses candidates et leur récompense sont enregistrées dans
`preferences_candidates.jsonl`. Pour essayer une autre règle sans rien regénérer :

```bash
python training/select_pairs.py --min-margin 0.2 --min-substance 0.1
```

Le modèle NLI de la récompense doit tourner sur le GPU : le script affiche `Modèle NLI sur
cuda` au démarrage. Avec un PyTorch installé sans CUDA, il tourne sur le CPU et devient le
goulot (environ 45 s par réponse au lieu de 1 s sur une RTX 4060). Installer alors la roue CUDA
de PyTorch adaptée au pilote (sélecteur sur pytorch.org). Vérification avant de lancer, venv
activé (sinon `python` peut désigner une autre installation, sans CUDA) :

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # doit finir par True
```

Pendant le run, `nvidia-smi` doit lister un processus `python` à côté d'Ollama.

Le script enregistre sa progression (`preferences_progress.jsonl`) : relancé après une
interruption (mise en veille, redémarrage), il reprend là où il s'était arrêté. `--restart`
repart de zéro ; il est exigé si les paramètres ou les questions ont changé.

`training/data/preferences_stats.json` résume la récolte :
- `skipped_source_not_retrieved` : questions écartées car le chunk source n'a pas été récupéré
  (le retrieval rate le passage ; ce n'est pas au générateur de corriger ça) ;
- `rejected_reason:*` : pourquoi les réponses rejetées l'ont été (faux refus, citation
  inventée…) — c'est le profil d'erreurs que le DPO va corriger ;
- `no_pair` : questions sans écart suffisant entre réponses (le modèle est constant dessus).

**Inspection obligatoire avant d'entraîner** : lire 30 paires au hasard. Vérifier que la
réponse « chosen » est réellement meilleure, et qu'elle n'est pas systématiquement plus courte,
plus évasive ou seulement mieux citée que la « rejected » (la récompense ne mesure pas la
pertinence). Viser au moins 200 paires ; en dessous, augmenter `--per-chunk` ou `--samples`.

Premier run (15 octobre 2026, questions à 2 par chunk, sans condition sur le fond) : 164 paires,
conservées dans `training/data/run1_20261015/`. Relecture de 30 paires : 9 où la retenue est
vraiment meilleure, 13 où seule la citation diffère, 8 où la retenue est moins bonne ou douteuse.
D'où la condition sur le fond et la régénération à 3 questions par chunk.

## 4. Pourquoi une récompense vérifiable

Entraîner contre un juge LLM de 7-8B revient à optimiser un signal bruité et biaisé : le
modèle apprend à plaire au juge (reward hacking). Les contraintes dures de `reward.py` sont
vérifiées mécaniquement (refus, sources, citations textuelles), et la composante de grounding
utilise un modèle NLI dédié, distinct du générateur et du juge d'évaluation. L'évaluation
finale utilise en plus Ragas, que l'entraînement n'a jamais vu.

## 5. Entraînement (Colab / Kaggle)

Ouvrir `training/dpo_colab.ipynb` sur Colab (GPU T4) ou Kaggle, téléverser
`training/data/preferences.jsonl`, exécuter. Il faut un token Hugging Face avec l'accès accepté
à `meta-llama/Llama-3.1-8B-Instruct`.

Lecture de `training_summary.json` :
- `eval_rewards/accuracies` : part des paires de validation où le modèle préfère la bonne
  réponse. Proche de 1 dès le départ : paires trop faciles. Proche de 0,5 : rien appris.
- `eval_rewards/margins` : doit croître sans exploser (une marge énorme avec un `beta` faible
  peut signaler une dérive loin du modèle de référence).

Hyperparamètres de départ : 1 époque, `lr 5e-6`, `beta 0.1`, LoRA `r 16`. Ne pas les ajuster
en regardant le jeu d'évaluation (sinon il cesse d'être une mesure indépendante) : utiliser la
validation interne de `train_dpo.py`.

## 6. Export vers Ollama

```bash
unzip dpo-lora.zip -d training/outputs/
python training/export_ollama.py --adapter training/outputs/dpo-lora
ollama create rag-assistant-dpo -f training/outputs/ollama/Modelfile
```

## 7. Évaluation et décision

```bash
python evaluation/run_eval.py --run-name dpo_v1 --model rag-assistant-dpo --repeats 2
python evaluation/compare_runs.py baseline_v2 dpo_v1
```

Adopter `rag-assistant-dpo` (variable `OLLAMA_LLM_MODEL`) seulement si au moins une métrique
visée s'améliore significativement **et** si `answer_relevancy` et `correct_refusal` ne se
dégradent pas significativement. Si les résultats sont erratiques, refaire l'export avec
`--merge` (voir la docstring de `export_ollama.py`).

## 8. Limites à garder en tête

- Le gabarit de chat d'entraînement reproduit celui d'Ollama pour Llama 3.1 sans outils ; une
  mise à jour d'Ollama qui le modifierait créerait un léger décalage.
- Les questions d'entraînement sont synthétiques ; des questions réelles (logs de l'API)
  seraient un meilleur matériau pour une V3.
