"""
Ancrage par NLI (inférence en langage naturel) : une phrase découle-t-elle d'un passage ?

Partagé par la récompense DPO (training/reward.py) et le filtrage des compléments
(rag_chain/complements.py). Modèle multilingue : les phrases sont en français, les passages en
anglais. Chargé à la première utilisation ; sur le GPU s'il est disponible.
"""
import re
from functools import lru_cache

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


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
