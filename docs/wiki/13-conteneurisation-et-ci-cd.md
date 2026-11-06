# 13. Conteneurisation et CI/CD

L'API est livrée sous forme d'image Docker ; GitHub Actions vérifie chaque modification et publie l'image.

## 13.1 Docker

Le `Dockerfile` part de `python:3.12-slim`, installe PyTorch en version CPU (les embeddings et le reranker tournent sur CPU dans le conteneur ; le LLM tourne dans Ollama sur l'hôte, avec le GPU), puis les dépendances, copie le code et **construit les chunks au build**. Au démarrage, l'API indexe si nécessaire. Un `HEALTHCHECK` interroge `/health`.

`docker-compose.yml` publie l'API sur le port 8600 (le 8000 peut être réservé par Windows avec Hyper-V), monte `./chroma_db` en volume, met en cache les modèles Hugging Face dans un volume nommé, et contacte Ollama sur l'hôte via `host.docker.internal`. Le `.dockerignore` exclut les tests, l'entraînement, les résultats et les secrets.

## 13.2 CI/CD

![Figure 5. Workflow `.github/workflows/ci-cd.yml`.](assets/figure-5.png)

*Figure 5. Workflow `.github/workflows/ci-cd.yml`.*

Le job « Corpus » mérite une explication : il relance `build_corpus.py` sur le runner et exige que `corpus/` soit identique au contenu versionné. Il garantit que le corpus n'a pas été modifié à la main et qu'il est reproductible depuis les sources figées. Le smoke test du conteneur garantit que l'image publiée démarre, télécharge ses modèles, construit son index et répond.

Tags d'image produits : SHA court, nom de branche, version sémantique pour les tags `v*`, et `latest` sur `main`. Le paquet GHCR est créé en privé ; on peut le rendre public dans les réglages du paquet.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Évaluation en CI** : un run `--retrieval-only` sur chaque PR, avec commentaire automatique des métriques et comparaison avec `main`. Demande de télécharger les modèles d'embeddings (cache possible).
>
> - **Déploiement continu** vers un hébergeur (Fly.io, Render, Cloud Run). Bloqué ici par le LLM : il faudrait un GPU ou une API.
>
> - **pre-commit** en local (ruff, fin de ligne) pour attraper les erreurs avant le push.
>
> - **Dependabot** pour les mises à jour des actions et des dépendances (et la fin du support de Node 20 dans les actions `checkout` / `setup-python` v4/v5).
