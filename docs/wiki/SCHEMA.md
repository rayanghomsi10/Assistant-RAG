# Conventions du wiki

Ce wiki est la documentation de référence du projet. Il est écrit comme un cours : chaque
composant est expliqué, justifié et confronté aux alternatives. Il est tenu à jour à chaque
évolution du code, par la personne ou l'agent qui fait la modification. Le PDF
`docs/Assistant-RAG_documentation_cours.pdf` n'est qu'une sortie : on ne le modifie jamais à la
main, on le régénère.

## Organisation

| Élément | Rôle |
|---|---|
| `index.md` | Catalogue : une ligne par page, avec son résumé. Point d'entrée de toute lecture. |
| `log.md` | Journal des modifications, une entrée par évolution, la plus récente en bas. |
| `NN-titre.md` | Un chapitre par page, numérotés dans l'ordre de lecture du PDF. |
| `A-…`, `B-…` | Annexes (glossaire, questions de révision). |
| `assets/` | Figures (PNG), référencées en chemin relatif. |
| `SCHEMA.md` | Ce fichier. |

Les sources « brutes » du wiki sont le code du dépôt, `training/README.md`, les résultats
versionnés (`evaluation/results/`) et les fichiers de données (`training/data/*stats.json`). Le
wiki ne recopie pas ces sources : il les explique, et renvoie vers elles par leur chemin.

## Format d'une page

- Titre de niveau 1 : `# N. Titre` (ou `# A. Titre` pour une annexe). Sections : `## N.M Titre`,
  sous-sections : `### Titre`. Les numéros de section servent aux renvois (« section 19.4 »).
- Identifiants de code, chemins, commandes et noms de variables entre accents graves.
- Blocs de code et sorties de terminal en blocs délimités par trois accents graves.
- Tableaux Markdown simples (pas de cellules fusionnées, pas de retour à la ligne dans une
  cellule).
- Encadrés au format des alertes GitHub, avec l'étiquette en gras sur la première ligne :

  | Alerte | Étiquettes |
  |---|---|
  | `> [!TIP]` | POURQUOI CE CHOIX, POURQUOI … |
  | `> [!NOTE]` | CE QU'ON AURAIT PU FAIRE |
  | `> [!WARNING]` | ATTENTION |
  | `> [!IMPORTANT]` | LA LEÇON, RÈGLE DE DÉCISION, ÉTAT RÉEL DU PROJET |

- Figures : `![Figure N. Légende](assets/figure-N.png)` suivie de la légende en italique.

## Règles de contenu

- **Un chiffre non mesuré n'apparaît jamais comme un résultat.** Tout résultat cite son run
  (`evaluation/results/<run>/`), sa date et, si possible, son intervalle de confiance.
- Une conclusion s'accompagne de ce qu'elle ne prouve pas (taille de l'échantillon, bruit du
  juge, absence de comparaison).
- Les dates sont absolues (« 18 octobre 2026 »), jamais relatives.
- Le texte est en français ; on garde les termes techniques anglais usuels (chunk, retrieval,
  reranker) et on les définit dans le glossaire.

## Opérations

### Mise à jour après une évolution du code

1. Repérer les pages concernées avec `index.md` et une recherche du nom du module ou de la
   variable modifiés dans `docs/wiki/`.
2. Modifier le passage existant plutôt qu'ajouter une rectification ailleurs ; si l'historique
   compte (une décision revue, un défaut corrigé), le dire dans le chapitre qui en parle.
3. Mettre à jour les passages transverses : tableau de configuration (3.5), arborescence (3.3),
   nombre de tests (12), mode d'emploi (18), état du DPO (19.6), pistes (17), limites (16).
4. Ajouter une entrée à `log.md` et, si une page a été créée ou son contenu a changé de nature,
   corriger sa ligne dans `index.md`.
5. Régénérer le PDF : `python docs/build_pdf.py`.

### Nouveau résultat mesuré

Ajouter la mesure au chapitre qui traite du composant, avec la commande, le run, la date et
l'intervalle de confiance ; mettre à jour 19.6 ou 19.7 si l'état d'avancement change.

### Vérification (lint)

Avant de régénérer le PDF, vérifier : renvois de section qui pointent vers une section existante,
chiffres cohérents d'une page à l'autre (nombre de tests, de paires, de questions), affirmations
périmées (« à faire » devenu fait), liens d'images valides. `python docs/build_pdf.py --check`
signale les renvois et les images cassés.
