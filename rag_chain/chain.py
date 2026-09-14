"""
Étape 5 de la feuille de route : chaîne RAG complète.

Pattern retriever -> prompt template -> LLM -> parsing de sortie, assemblé avec
LangChain Expression Language (LCEL). Le prompt impose au modèle de répondre
uniquement à partir du contexte fourni, pour réduire les hallucinations.
"""
import sys
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, TOP_K
from retrieval.vector_store import similarity_search

SYSTEM_PROMPT = """Tu es un assistant technique qui répond à des questions sur la documentation \
dbt et Apache Airflow.

Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. N'utilise aucune autre connaissance.
- Si le contexte ne contient pas la réponse, dis explicitement "Je ne trouve pas cette \
information dans les documents fournis." Ne l'invente pas.
- Cite les sources pertinentes (nom de document) qui justifient ta réponse.
- Réponds en français, de façon concise et précise.

Contexte :
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL, temperature=temperature)


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[Source: {h['source']}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    return similarity_search(question, k=k)


def build_chain(temperature: float = 0.0):
    """Construit la chaîne LCEL : question -> prompt enrichi -> LLM -> texte."""
    llm = get_llm(temperature=temperature)
    return (
        {"context": RunnableLambda(lambda x: format_context(x["hits"])), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_question(question: str, k: int = TOP_K, temperature: float = 0.0) -> dict:
    """Point d'entrée principal : question -> réponse générée + sources citées."""
    if not question or not question.strip():
        raise ValueError("La question est vide.")

    hits = retrieve(question, k=k)
    chain = build_chain(temperature=temperature)
    answer = chain.invoke({"question": question, "hits": hits})

    sources = sorted({h["source"] for h in hits})
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "contexts": [h["text"] for h in hits],  # utile pour l'évaluation Ragas (context_precision)
    }


if __name__ == "__main__":
    demo_questions = [
        "Quelle est la différence entre une table et une vue en dbt ?",
        "Comment Airflow décide-t-il quand exécuter un DAG ?",
    ]
    for q in demo_questions:
        print(f"\nQ: {q}")
        result = answer_question(q)
        print(f"R: {result['answer']}")
        print(f"Sources: {result['sources']}")
