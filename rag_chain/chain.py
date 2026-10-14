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
"""
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_NUM_CTX, RELEVANCE_THRESHOLD, TOP_K
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
    ("human", "{question}"),
])


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        header = f"[Document : {h['source']}"
        if h.get("section"):
            header += f" | Section : {h['section']}"
        blocks.append(f"{header}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    """Messages de chat (format OpenAI/HF) exactement tels que le LLM les reçoit."""
    msgs = prompt.format_messages(context=format_context(hits), question=question)
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
             temperature: float = 0.0, seed: int | None = None) -> str:
    chain = build_chain(model, temperature, seed)
    return chain.invoke({"context": format_context(hits), "question": question},
                        config=tracing.langchain_config("génération"))


def below_threshold(hits: list[dict], threshold: float | None = RELEVANCE_THRESHOLD) -> bool:
    return threshold is not None and (not hits or hits[0]["score"] < threshold)


@tracing.observe("question-rag", as_type="chain")
def answer_question(question: str, k: int = TOP_K, model: str = OLLAMA_LLM_MODEL,
                    temperature: float = 0.0) -> dict:
    """Point d'entrée principal : question -> réponse générée + sources."""
    if not question or not question.strip():
        raise ValueError("La question est vide.")

    hits = retrieve(question, k=k)
    retrieved_sources = sorted({h["source"] for h in hits})

    if below_threshold(hits):
        answer, gated = REFUSAL_MESSAGE, True
    else:
        answer, gated = generate(question, hits, model=model, temperature=temperature), False

    return {
        "question": question,
        "answer": answer,
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
