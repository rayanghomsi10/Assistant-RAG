"""
Étape 4 du pipeline : chaîne RAG complète.

Pattern retriever -> prompt template -> LLM -> parsing de sortie, assemblé avec
LangChain Expression Language (LCEL). Le prompt impose au modèle de répondre
uniquement à partir du contexte fourni, de citer ses sources par nom de document et de ne
mettre entre guillemets que des passages recopiés mot pour mot.

`build_messages` est la **seule** définition du prompt : la chaîne l'utilise à l'inférence, et
le pipeline DPO (training/) l'utilise pour construire les exemples d'entraînement. Un modèle
entraîné sur un prompt différent de celui qu'il reçoit en production n'apprend pas la bonne
chose.

Mémoire de conversation : les derniers échanges sont insérés entre le message système et la
question, et une relance elliptique (« et pour les vues ? ») est d'abord reformulée en question
autonome, qui sert à la recherche. Sans historique, le prompt est strictement celui d'avant :
l'évaluation et le DPO, qui travaillent question par question, ne sont pas affectés.
"""
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HISTORY_MAX_TURNS,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_NUM_CTX,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from rag_chain import tracing
from rag_chain.checks import REFUSAL_MESSAGE, cited_documents, is_refusal
from retrieval.vector_store import search

SYSTEM_PROMPT = f"""Tu es un assistant technique qui répond à des questions sur la documentation \
dbt et Apache Airflow.

Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. N'utilise aucune autre connaissance.
- Si le contexte ne contient pas la réponse, réponds exactement : "{REFUSAL_MESSAGE}" \
et rien d'autre. N'invente pas.
- Cite les documents qui justifient ta réponse par leur nom entre crochets, tel qu'il apparaît \
dans le contexte, par exemple [airflow_04_xcoms.md]. Ne cite jamais un document absent du contexte.
- N'utilise des guillemets que pour recopier un passage du contexte mot pour mot, dans sa langue \
d'origine (anglais). Sinon, reformule sans guillemets.
- Écris les noms de fonctions, de paramètres et de commandes exactement comme dans le contexte, \
entre accents graves (par exemple `xcom_push`).
- Réponds en français, de façon concise et précise.

Contexte :
{{context}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Tu reformules la dernière question d'une conversation sur dbt et Apache Airflow en une "
               "question autonome, compréhensible sans l'historique, en français. Remplace les pronoms et "
               "les ellipses par ce qu'ils désignent (outil, fonctionnalité, paramètre). Si la question est "
               "déjà autonome, recopie-la telle quelle. Ne réponds pas à la question : renvoie uniquement "
               "la question reformulée, sur une ligne."),
    ("human", "Historique :\n{history}\n\nDernière question : {question}"),
])

# Une réponse longue de l'historique est tronquée : elle sert de contexte de conversation, pas de
# source (les sources restent les passages récupérés pour la question courante).
HISTORY_ANSWER_MAX_CHARS = 1500


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        header = f"[Document : {h['source']}"
        if h.get("section"):
            header += f" | Section : {h['section']}"
        blocks.append(f"{header}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def recent_history(history: list[dict] | None, max_turns: int = HISTORY_MAX_TURNS) -> list[dict]:
    """Derniers échanges {question, answer} conservés pour le prompt."""
    return list(history or [])[-max_turns:] if max_turns > 0 else []


def history_messages(history: list[dict] | None) -> list:
    msgs = []
    for turn in recent_history(history):
        msgs += [HumanMessage(turn["question"]), AIMessage(turn["answer"][:HISTORY_ANSWER_MAX_CHARS])]
    return msgs


def build_messages(question: str, hits: list[dict], history: list[dict] | None = None) -> list[dict]:
    """Messages de chat (format OpenAI/HF) exactement tels que le LLM les reçoit."""
    msgs = prompt.format_messages(context=format_context(hits), question=question,
                                  history=history_messages(history))
    role = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": role[m.type], "content": m.content} for m in msgs]


def get_llm(model: str = OLLAMA_LLM_MODEL, temperature: float = 0.0, seed: int | None = None) -> ChatOllama:
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=model, temperature=temperature,
                      num_ctx=OLLAMA_NUM_CTX, seed=seed)


@lru_cache(maxsize=8)
def build_chain(model: str = OLLAMA_LLM_MODEL, temperature: float = 0.0, seed: int | None = None):
    """Construit (une fois par configuration) la chaîne LCEL : prompt -> LLM -> texte."""
    return prompt | get_llm(model, temperature, seed) | StrOutputParser()


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    return search(question, k=k)


def generate(question: str, hits: list[dict], model: str = OLLAMA_LLM_MODEL,
             temperature: float = 0.0, seed: int | None = None, history: list[dict] | None = None) -> str:
    chain = build_chain(model, temperature, seed)
    return chain.invoke({"context": format_context(hits), "question": question,
                         "history": history_messages(history)},
                        config=tracing.langchain_config("génération"))


def condense_question(question: str, history: list[dict] | None, model: str = OLLAMA_LLM_MODEL) -> str:
    """Question autonome pour la recherche ; la question elle-même s'il n'y a pas d'historique."""
    turns = recent_history(history)
    if not turns:
        return question
    text = "\n".join(f"Utilisateur : {t['question']}\nAssistant : {t['answer'][:HISTORY_ANSWER_MAX_CHARS]}"
                     for t in turns)
    chain = CONDENSE_PROMPT | get_llm(model) | StrOutputParser()
    out = chain.invoke({"history": text, "question": question}, config=tracing.langchain_config("reformulation"))
    lines = [line.strip().strip('"«» ') for line in out.strip().splitlines() if line.strip()]
    # Garde-fou : une sortie vide ou bavarde (le modèle a répondu au lieu de reformuler) est ignorée.
    if not lines or len(lines[0]) > 4 * len(question) + 200:
        return question
    return lines[0]


def below_threshold(hits: list[dict], threshold: float | None = RELEVANCE_THRESHOLD) -> bool:
    return threshold is not None and (not hits or hits[0]["score"] < threshold)


@tracing.observe("question-rag", as_type="chain")
def answer_question(question: str, k: int = TOP_K, model: str = OLLAMA_LLM_MODEL,
                    temperature: float = 0.0, history: list[dict] | None = None,
                    complements: bool = False) -> dict:
    """Point d'entrée principal : question (+ historique) -> réponse générée + sources.

    `history` : échanges précédents [{question, answer}] ; `complements` : chercher en plus des
    mises en garde, alternatives ou informations utiles (rag_chain/complements.py).
    """
    if not question or not question.strip():
        raise ValueError("La question est vide.")

    standalone = condense_question(question, history, model=model)
    hits = retrieve(standalone, k=k)
    retrieved_sources = sorted({h["source"] for h in hits})

    if below_threshold(hits):
        answer, gated = REFUSAL_MESSAGE, True
    else:
        answer, gated = generate(question, hits, model=model, temperature=temperature, history=history), False

    extra = []
    if complements and not is_refusal(answer):
        from rag_chain.complements import suggest_complements
        extra = suggest_complements(standalone, answer, hits, model=model)

    return {
        "question": question,
        "standalone_question": standalone,
        "answer": answer,
        "complements": extra,
        "refused": is_refusal(answer),
        "refused_by_threshold": gated,
        "retrieved_sources": retrieved_sources,
        # Documents réellement cités dans la réponse (et parmi les documents récupérés).
        "cited_sources": sorted(cited_documents(answer) & set(retrieved_sources)),
        "contexts": [h["text"] for h in hits],
        "hits": [{k_: h[k_] for k_ in ("id", "source", "section", "similarity", "rerank_score", "score")}
                 for h in hits],
        "model": model,
        "trace_id": tracing.current_trace_id(),
    }


if __name__ == "__main__":
    demo_questions = [
        "Quelle est la différence entre une table et une vue en dbt ?",
        "Comment Airflow décide-t-il quand exécuter un Dag ?",
    ]
    for q in demo_questions:
        print(f"\nQ: {q}")
        result = answer_question(q)
        print(f"R: {result['answer']}")
        print(f"Sources citées: {result['cited_sources']}  (récupérées: {result['retrieved_sources']})")
