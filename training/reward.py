"""
Récompense utilisée pour construire les paires de préférences DPO.

Principe : la récompense repose sur des critères **vérifiables**, pas sur un LLM juge. Un juge
LLM de 7-8B est bruité et biaisé (la V1 l'a montré sur answer_relevancy) ; optimiser un modèle
contre lui revient à lui apprendre à plaire au juge (reward hacking). Ici :

Contraintes dures (récompense 0 si l'une est violée) :
  - question du périmètre dont le passage source a été récupéré : refuser est une erreur ;
  - question hors périmètre : répondre au lieu de refuser est une erreur ;
  - citer un document qui n'est pas dans le contexte ;
  - mettre entre guillemets un passage absent du contexte ;
  - réponse vide ou quasi vide (moins de MIN_WORDS mots), qui serait sinon trivialement « fidèle ».

Composantes graduées (questions du périmètre), pondérées et normalisées dans [0, 1] :
  - grounding : part des phrases de la réponse impliquées (NLI multilingue) par un passage du
    contexte — l'équivalent vérifiable de la faithfulness ; optionnel (modèle NLI requis) ;
  - cites_source : la réponse cite le document dont la question est tirée ;
  - code_support : part des identifiants de code présents dans le contexte ;
  - has_citation : au moins une source citée ;
  - concision : pénalité au-delà de LENGTH_SOFT_LIMIT mots.

Risque connu : aucune composante ne mesure directement si la réponse *répond* à la question.
D'où la contrainte de longueur minimale, le contrôle de answer_relevancy dans l'évaluation
(Ragas) et l'inspection manuelle d'un échantillon de paires (voir training/README.md).
"""
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from rag_chain.checks import check_answer

MIN_WORDS = 12
LENGTH_SOFT_LIMIT = 180
LENGTH_HARD_LIMIT = 400

WEIGHTS = {"grounding": 0.40, "cites_source": 0.20, "code_support": 0.15,
           "has_citation": 0.10, "concision": 0.15}

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


@dataclass
class Reward:
    total: float
    hard_fail: str | None = None
    components: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.hard_fail is None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# NLI (optionnel)
# ---------------------------------------------------------------------------------------------

class NliScorer:
    """Probabilité d'implication (entailment) d'une hypothèse par une prémisse, multilingue :
    les réponses sont en français, le contexte en anglais."""

    def __init__(self, model_name: str = NLI_MODEL_NAME, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail_idx = labels["entailment"]

    def entailment(self, premises: list[str], hypothesis: str) -> list[float]:
        with self.torch.no_grad():
            enc = self.tokenizer(premises, [hypothesis] * len(premises), truncation="only_first",
                                 max_length=512, padding=True, return_tensors="pt").to(self.device)
            probs = self.model(**enc).logits.softmax(-1)[:, self.entail_idx]
        return probs.tolist()


@lru_cache(maxsize=1)
def get_nli_scorer() -> NliScorer:
    return NliScorer()


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def answer_sentences(answer: str) -> list[str]:
    text = re.sub(r"\[[^\]]+\]", "", answer)                      # références [doc.md]
    sentences = [s.strip(" -*•\t") for s in _SENT_SPLIT.split(text)]
    return [s for s in sentences if len(s.split()) >= 4]


def grounding_score(answer: str, contexts: list[str], nli: NliScorer) -> float:
    sentences = answer_sentences(answer)
    if not sentences:
        return 0.0
    scores = [max(nli.entailment(contexts, s)) for s in sentences]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------------------------

def concision(n_words: int) -> float:
    if n_words <= LENGTH_SOFT_LIMIT:
        return 1.0
    if n_words >= LENGTH_HARD_LIMIT:
        return 0.0
    return 1.0 - (n_words - LENGTH_SOFT_LIMIT) / (LENGTH_HARD_LIMIT - LENGTH_SOFT_LIMIT)


def compute_reward(answer: str, question_type: str, hits: list[dict], source_document: str | None = None,
                   nli: NliScorer | None = None) -> Reward:
    """Récompense d'une réponse. `question_type` : 'in_scope' (passage source récupéré) ou
    'out_of_scope'. `source_document` : document dont la question a été tirée."""
    contexts = [h["text"] for h in hits]
    chk = check_answer(answer, contexts, sorted({h["source"] for h in hits}))

    if question_type == "out_of_scope":
        if not chk.refusal:
            return Reward(0.0, "answered_out_of_scope")
        return Reward(1.0 if chk.clean_refusal else 0.5, None, {"clean_refusal": float(chk.clean_refusal)})

    if chk.refusal:
        return Reward(0.0, "false_refusal")
    if chk.n_words < MIN_WORDS:
        return Reward(0.0, "too_short")
    if chk.cited_not_retrieved:
        return Reward(0.0, "cites_unretrieved_document")
    if chk.unsupported_quotes:
        return Reward(0.0, "fabricated_quote")

    comps = {
        "cites_source": float(source_document in chk.cited) if source_document else float(chk.has_citation),
        "code_support": chk.code_support,
        "has_citation": float(chk.has_citation),
        "concision": concision(chk.n_words),
    }
    if nli is not None:
        comps["grounding"] = grounding_score(answer, contexts, nli)
    weights = {k: WEIGHTS[k] for k in comps}
    total = sum(weights[k] * comps[k] for k in comps) / sum(weights.values())
    return Reward(round(total, 4), None, comps)
