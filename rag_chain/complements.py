"""
Compléments à une réponse : mises en garde, autres façons de faire, informations à connaître.

Le but est de signaler ce qu'un ingénieur risque d'ignorer, sans sacrifier l'ancrage :

1. Recherche dédiée : les passages qui contiennent une mise en garde (encadrés « Note: »,
   « Warning: », « Tip: »... conservés par build_corpus.py) ou qui viennent du guide de bonnes
   pratiques d'Airflow, liés à la question selon le reranker, et absents du contexte de la réponse.
   Les 4 passages de la réponse sont choisis pour répondre, pas pour trouver ce qui l'entoure.
2. Un appel au LLM, séparé de la réponse principale (dont le prompt reste celui de l'évaluation et
   du DPO), qui renvoie au plus quelques compléments en JSON, chacun avec son document source.
3. Filtrage par programme : un complément qui cite un document absent de ces passages, ou dont les
   citations ou identifiants de code n'y figurent pas, est écarté (rag_chain/checks.py), de même
   qu'une question posée au lieu d'une information.
4. Filtrage sémantique : le modèle NLI (rag_chain/nli.py, celui de la récompense DPO) doit juger
   que chaque phrase du complément découle d'un des passages. Les premiers essais ont montré que
   le modèle de 8B produit sinon des conseils génériques présentés comme tirés de la documentation.
   Si le NLI ne peut pas être chargé, aucun complément n'est montré : mieux vaut rien qu'un
   complément non vérifié.

Ne rien proposer est le résultat attendu le plus souvent.
"""
import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import (
    COMPLEMENTS_MAX,
    COMPLEMENTS_MIN_ENTAILMENT,
    COMPLEMENTS_MIN_SCORE,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_NUM_CTX,
)
from rag_chain import tracing
from rag_chain.checks import check_answer
from rag_chain.nli import grounding_score
from retrieval.vector_store import search

log = logging.getLogger(__name__)

CAVEAT_MARKER = re.compile(r"^(Note|Warning|Caution|Important|Tip|Info|Danger|Attention)\s*:", re.M | re.I)
BEST_PRACTICE_DOCS = {"airflow_05_best_practices.md"}
TYPES = {"mise_en_garde": "Mise en garde", "alternative": "Autre façon de faire", "a_savoir": "À savoir"}

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu aides un ingénieur data en lui signalant ce qu'il risque d'ignorer. On te donne sa \
question, la réponse qu'il a déjà reçue et des extraits de documentation (mises en garde, notes, bonnes \
pratiques).

Propose au plus {max_items} compléments utiles qui ne figurent PAS déjà dans la réponse :
- "mise_en_garde" : un piège, une limite ou un risque lié à ce qu'il cherche à faire ;
- "alternative" : une autre manière de faire, quand l'extrait en décrit une ;
- "a_savoir" : une information qu'il a besoin de connaître pour appliquer la réponse.

Règles strictes :
- Chaque complément s'appuie sur un extrait et cite son document, tel qu'il apparaît dans les extraits.
- N'utilise aucune autre connaissance. Écris en français, en une ou deux phrases.
- Écris les noms de fonctions, de paramètres et de commandes exactement comme dans les extraits.
- Si aucun extrait n'apporte rien d'utile pour cette question, renvoie une liste vide : c'est la \
réponse attendue dans la plupart des cas.

Réponds uniquement en JSON : {{"complements": [{{"type": "mise_en_garde", "texte": "...", \
"source": "nom_du_document.md"}}]}}"""),
    ("human", "Question : {question}\n\nRéponse déjà donnée :\n{answer}\n\nExtraits :\n{context}"),
])


def is_caveat(hit: dict) -> bool:
    return hit["source"] in BEST_PRACTICE_DOCS or bool(CAVEAT_MARKER.search(hit["text"]))


def find_caveats(question: str, exclude_ids: set[str], k: int = 4, candidates: int = 40,
                 min_score: float = COMPLEMENTS_MIN_SCORE) -> list[dict]:
    """Passages de mise en garde liés à la question, hors contexte de la réponse."""
    hits = search(question, k=candidates, candidates=candidates)
    keep = [h for h in hits if h["id"] not in exclude_ids and is_caveat(h)
            and (h["rerank_score"] is None or h["rerank_score"] >= min_score)]
    return keep[:k]


def parse_complements(raw: str, caveats: list[dict], max_items: int = COMPLEMENTS_MAX) -> list[dict]:
    """Compléments valides : type connu, document parmi les extraits, citations et code vérifiés."""
    try:
        items = json.loads(raw).get("complements", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    sections = {}
    for h in caveats:
        sections.setdefault(h["source"], h.get("section", ""))
    contexts = [h["text"] for h in caveats]
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        kind, text, source = it.get("type"), str(it.get("texte", "")).strip(), str(it.get("source", "")).strip()
        source = source.strip("[]")
        if kind not in TYPES or not text or text.endswith("?") or source not in sections or text in seen:
            continue
        check = check_answer(text, contexts, list(sections))
        if check.unsupported_quotes or check.unsupported_code:
            continue
        seen.add(text)
        out.append({"type": kind, "label": TYPES[kind], "text": text, "source": source,
                    "section": sections[source]})
    return out[:max_items]


def keep_grounded(items: list[dict], caveats: list[dict], nli,
                  min_entailment: float = COMPLEMENTS_MIN_ENTAILMENT) -> list[dict]:
    """Compléments dont les phrases découlent des passages selon le NLI (score moyen par phrase)."""
    contexts = [h["text"] for h in caveats]
    kept = []
    for it in items:
        score = grounding_score(it["text"], contexts, nli)
        if score >= min_entailment:
            kept.append({**it, "grounding": round(score, 3)})
    return kept


@tracing.observe("compléments", as_type="chain")
def suggest_complements(question: str, answer: str, hits: list[dict], model: str = OLLAMA_LLM_MODEL,
                        max_items: int = COMPLEMENTS_MAX) -> list[dict]:
    """Compléments pour une réponse ; liste vide si rien d'utile ou en cas d'erreur."""
    from rag_chain.chain import format_context
    try:
        caveats = find_caveats(question, exclude_ids={h["id"] for h in hits})
        if not caveats:
            return []
        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=model, temperature=0.0, num_ctx=OLLAMA_NUM_CTX,
                         format="json")
        raw = (PROMPT | llm).invoke({"question": question, "answer": answer, "context": format_context(caveats),
                                     "max_items": max_items}, config=tracing.langchain_config("génération")).content
        items = parse_complements(raw, caveats, max_items)
        if not items:
            return []
        from rag_chain.nli import get_nli_scorer
        return keep_grounded(items, caveats, get_nli_scorer())
    except Exception as exc:  # un complément ne doit jamais faire échouer la réponse principale
        log.warning("Compléments ignorés : %s", exc)
        return []
