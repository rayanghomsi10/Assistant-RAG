"""
Observabilité avec Langfuse (optionnelle).

Actif seulement si LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY sont définies (voir .env.example).
Sinon, toutes les fonctions de ce module sont sans effet et le paquet langfuse n'est même pas
importé : les tests et la CI ne dépendent d'aucun serveur.

Ce qui est tracé : une trace par question, qu'elle vienne de l'API, de l'évaluation ou de la
construction des paires DPO. Elle contient la recherche (passages et scores) et l'appel au LLM
(prompt exact, réponse, latence, tokens) comme observations imbriquées ; l'évaluation y attache
ses métriques comme scores, ce qui permet de filtrer dans Langfuse les réponses en échec.
"""
import math
from contextlib import contextmanager
from functools import lru_cache

from config import LANGFUSE_ENABLED


def observe(name: str, as_type: str = "span"):
    """Décorateur : trace la fonction comme une observation (entrées et sortie capturées)."""
    if not LANGFUSE_ENABLED:
        return lambda func: func
    from langfuse import observe as langfuse_observe
    return langfuse_observe(name=name, as_type=as_type)


@lru_cache(maxsize=1)
def _langchain_handler():
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def langchain_config(run_name: str) -> dict:
    """Configuration d'invocation LangChain qui rattache l'appel au LLM à la trace en cours."""
    if not LANGFUSE_ENABLED:
        return {"run_name": run_name}
    return {"run_name": run_name, "callbacks": [_langchain_handler()]}


@contextmanager
def trace(name: str, *, session_id: str | None = None, tags: list[str] | None = None,
          metadata: dict | None = None, input=None):
    """Ouvre une observation ; ce qui est tracé à l'intérieur s'y rattache.

    Hors de toute trace, elle devient la racine d'une nouvelle trace, regroupée par session
    (un run d'évaluation, un run DPO) et étiquetée par `tags`.
    """
    if not LANGFUSE_ENABLED:
        yield
        return
    from langfuse import get_client, propagate_attributes
    with get_client().start_as_current_observation(name=name, as_type="span", input=input, metadata=metadata), \
            propagate_attributes(session_id=session_id, tags=tags, trace_name=name):
        yield


@contextmanager
def attributes(*, session_id: str | None = None, tags: list[str] | None = None):
    """Étiquette les traces créées à l'intérieur (par exemple celles de l'API)."""
    if not LANGFUSE_ENABLED:
        yield
        return
    from langfuse import propagate_attributes
    with propagate_attributes(session_id=session_id, tags=tags):
        yield


def set_output(output=None, metadata: dict | None = None) -> None:
    """Renseigne la sortie et les métadonnées de l'observation en cours."""
    if LANGFUSE_ENABLED:
        from langfuse import get_client
        get_client().update_current_span(output=output, metadata=metadata)


def _valid(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def score(name: str, value, comment: str | None = None) -> None:
    """Attache un score numérique à la trace en cours (valeurs manquantes ignorées)."""
    if LANGFUSE_ENABLED and _valid(value):
        from langfuse import get_client
        get_client().score_current_trace(name=name, value=float(value), comment=comment)


def score_trace(trace_id: str | None, name: str, value) -> None:
    """Attache un score à une trace déjà terminée (scores Ragas, calculés après coup)."""
    if LANGFUSE_ENABLED and trace_id and _valid(value):
        from langfuse import get_client
        get_client().create_score(trace_id=trace_id, name=name, value=float(value))


def current_trace_id() -> str | None:
    if not LANGFUSE_ENABLED:
        return None
    from langfuse import get_client
    return get_client().get_current_trace_id()


def trace_url(trace_id: str | None) -> str | None:
    """Lien vers la trace dans l'interface Langfuse."""
    if not (LANGFUSE_ENABLED and trace_id):
        return None
    from langfuse import get_client
    return get_client().get_trace_url(trace_id=trace_id)


def flush() -> None:
    """Envoie les données en attente : à appeler à la fin d'un script de courte durée."""
    if LANGFUSE_ENABLED:
        from langfuse import get_client
        get_client().flush()


if __name__ == "__main__":
    # Vérification de la configuration : python -m rag_chain.tracing
    import os
    if not LANGFUSE_ENABLED:
        raise SystemExit("Traçage inactif : renseigner LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans .env "
                         "(et LANGFUSE_TRACING_ENABLED différent de false).")
    from langfuse import get_client
    client = get_client()
    print(f"Serveur : {os.getenv('LANGFUSE_BASE_URL') or os.getenv('LANGFUSE_HOST') or 'https://cloud.langfuse.com'}")
    if not client.auth_check():
        raise SystemExit("Clés refusées : vérifier les clés et la région (LANGFUSE_BASE_URL).")
    with trace("verification-configuration", tags=["test"], input="ping"):
        set_output("pong")
        url = client.get_trace_url()
    flush()
    print(f"Connexion OK, trace de test envoyée : {url}")
