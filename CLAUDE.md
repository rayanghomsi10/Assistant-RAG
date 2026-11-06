# Consignes pour les agents

## Documentation

La documentation de référence est le wiki `docs/wiki/` (point d'entrée : `docs/wiki/index.md`,
conventions : `docs/wiki/SCHEMA.md`). À chaque évolution du code :

1. mettre à jour les pages concernées du wiki, `README.md` et `training/README.md` si besoin ;
2. ajouter une entrée à `docs/wiki/log.md` ;
3. lancer `python docs/build_pdf.py --check`, puis régénérer le PDF avec `python docs/build_pdf.py`.

Ne jamais modifier `docs/Assistant-RAG_documentation_cours.pdf` à la main : c'est une sortie.

## Code

- Vérifications : `ruff check .` et `pytest -q` (sans modèle ni GPU).
- Ne jamais afficher ni committer de secrets (`.env`, `langfuse/.env`).
- Messages de commit en français.
