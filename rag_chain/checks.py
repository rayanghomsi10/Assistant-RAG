"""
Vérifications déterministes d'une réponse RAG, sans LLM juge.

Utilisées à deux endroits :
  - l'évaluation (evaluation/run_eval.py) : métriques reproductibles au bit près, qui ne
    souffrent ni du bruit ni du biais d'un juge LLM ;
  - la construction des préférences DPO (training/build_preferences.py) : la récompense repose
    sur ces vérifications, pour qu'un modèle entraîné ne puisse pas « plaire au juge » sans
    réellement respecter les règles (reward hacking).

Ce qui est vérifié :
  - refus : la réponse est-elle le refus attendu ?
  - sources citées : chaque document cité fait-il partie des documents récupérés ?
  - citations entre guillemets : chaque passage cité est-il présent mot pour mot dans le contexte ?
  - identifiants de code (`xcom_push`, `materialized`…) : apparaissent-ils dans le contexte ?
"""
import re
import unicodedata
from dataclasses import asdict, dataclass, field

REFUSAL_MESSAGE = "Je ne trouve pas cette information dans les documents fournis."

_REFUSAL_PATTERNS = [
    r"je ne trouve pas cette information",
    r"je ne trouve pas (?:d'|d’)?informations?",
    r"(?:le|les) (?:contexte|documents?)(?: fournis?)? ne (?:contient|contiennent|mentionnent?|précisent?) pas",
    r"(?:aucune|pas d') ?information (?:à ce sujet|sur ce sujet|pertinente)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS))

# Nom de document du corpus : dbt_01_sql_models(.md), airflow_04_xcoms(.md)…
_DOC_NAME_RE = re.compile(r"\b((?:dbt|airflow)_\d{2}_[a-z0-9_]+?)(?:\.md|\.pdf)?\b")
_QUOTE_RE = re.compile(r"«\s*(.+?)\s*»|“(.+?)”|\"(.+?)\"", re.S)
_CODE_RE = re.compile(r"`([^`\n]+)`")

MIN_QUOTE_WORDS = 4   # en dessous, des guillemets encadrent souvent un simple terme ("views")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _strip_markdown(text: str) -> str:
    return re.sub(r"[*_`]", "", text)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(normalize(answer)))


def is_clean_refusal(answer: str, max_words: int = 40) -> bool:
    """Refus net : la réponse refuse, sans dérouler ensuite une réponse inventée."""
    return is_refusal(answer) and len(answer.split()) <= max_words


def cited_documents(answer: str) -> set[str]:
    """Documents du corpus cités dans la réponse (normalisés avec l'extension .md)."""
    return {m.group(1) + ".md" for m in _DOC_NAME_RE.finditer(answer.lower())}


def _doc_stem(source: str) -> str:
    return re.sub(r"\.(md|pdf)$", "", source.lower())


def extract_quotes(answer: str, min_words: int = MIN_QUOTE_WORDS) -> list[str]:
    quotes = []
    for m in _QUOTE_RE.finditer(answer):
        q = next(g for g in m.groups() if g is not None).strip()
        if len(q.split()) >= min_words:
            quotes.append(q)
    return quotes


def extract_code_spans(answer: str) -> list[str]:
    spans = []
    for s in _CODE_RE.findall(answer):
        s = s.strip()
        if s and len(s.split()) <= 8 and not _DOC_NAME_RE.fullmatch(s.lower()):
            spans.append(s)
    return spans


def _in_contexts(fragment: str, contexts_norm: list[str]) -> bool:
    frag = normalize(_strip_markdown(fragment))
    return any(frag in c for c in contexts_norm)


def _code_in_contexts(span: str, contexts_norm: list[str]) -> bool:
    # Même normalisation que les contextes (marqueurs Markdown retirés des deux côtés).
    bare = re.sub(r"\(\s*\)$", "", span).strip()          # xcom_push() -> xcom_push
    candidates = {normalize(_strip_markdown(s)) for s in (span, bare, bare.lstrip("@"))}  # @task -> task
    return any(c and any(c in ctx for ctx in contexts_norm) for c in candidates)


@dataclass
class AnswerCheck:
    refusal: bool
    clean_refusal: bool
    cited: list[str] = field(default_factory=list)
    cited_not_retrieved: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    unsupported_quotes: list[str] = field(default_factory=list)
    code_spans: list[str] = field(default_factory=list)
    unsupported_code: list[str] = field(default_factory=list)
    n_words: int = 0

    @property
    def has_citation(self) -> bool:
        return bool(self.cited)

    @property
    def citations_valid(self) -> bool:
        return not self.cited_not_retrieved

    @property
    def quotes_faithful(self) -> bool:
        return not self.unsupported_quotes

    @property
    def code_support(self) -> float:
        """Part des identifiants de code présents dans le contexte (1.0 s'il n'y en a pas)."""
        if not self.code_spans:
            return 1.0
        return 1.0 - len(self.unsupported_code) / len(self.code_spans)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(has_citation=self.has_citation, citations_valid=self.citations_valid,
                 quotes_faithful=self.quotes_faithful, code_support=self.code_support)
        return d


def check_answer(answer: str, contexts: list[str], retrieved_sources: list[str]) -> AnswerCheck:
    contexts_norm = [normalize(_strip_markdown(c)) for c in contexts]
    retrieved = {_doc_stem(s) for s in retrieved_sources}

    cited = sorted(cited_documents(answer))
    quotes = extract_quotes(answer)
    code = extract_code_spans(answer)
    return AnswerCheck(
        refusal=is_refusal(answer),
        clean_refusal=is_clean_refusal(answer),
        cited=cited,
        cited_not_retrieved=[c for c in cited if _doc_stem(c) not in retrieved],
        quotes=quotes,
        unsupported_quotes=[q for q in quotes if not _in_contexts(q, contexts_norm)],
        code_spans=code,
        unsupported_code=[s for s in code if not _code_in_contexts(s, contexts_norm)],
        n_words=len(answer.split()),
    )
