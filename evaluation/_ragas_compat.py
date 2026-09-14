"""
Correctif de compatibilité : ragas 0.4.3 importe inconditionnellement
`langchain_community.chat_models.vertexai.ChatVertexAI`, un sous-module que
`langchain-community` (en cours de dépréciation, cf. son propre warning) a retiré
dans ses versions récentes. Comme ce projet est 100% local (Ollama) et n'utilise
jamais VertexAI, on enregistre un module factice avant l'import de ragas pour éviter
un ModuleNotFoundError qui n'a aucun rapport avec notre pipeline.

Doit être importé avant tout `import ragas` (voir run_ragas_eval.py).
"""
import sys
import types

_MODULE_PATH = "langchain_community.chat_models.vertexai"

if _MODULE_PATH not in sys.modules:
    try:
        __import__(_MODULE_PATH)
    except ModuleNotFoundError:
        stub = types.ModuleType(_MODULE_PATH)

        class ChatVertexAI:  # noqa: N801 - nom imposé par l'import attendu
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "ChatVertexAI indisponible : ce projet n'utilise que des modèles "
                    "locaux via Ollama, VertexAI n'est pas installé."
                )

        stub.ChatVertexAI = ChatVertexAI
        sys.modules[_MODULE_PATH] = stub
