"""
Métriques déterministes par question et outils statistiques (intervalles de confiance).

Pourquoi des intervalles de confiance : en V1, deux configurations ont été comparées sur 15
questions avec une seule exécution chacune, et des écarts de 0,07 ont été interprétés alors que
des questions passaient de 1,0 à 0,0 d'une exécution à l'autre. Avec un bootstrap sur les
questions, on sait si un écart dépasse le bruit d'échantillonnage ; avec un bootstrap apparié
(compare_runs.py), on compare deux runs question par question.
"""
import numpy as np

from evaluation.dataset import norm
from rag_chain.checks import check_answer

# Métriques calculées sur les questions du périmètre / hors périmètre.
RETRIEVAL_METRICS = ["evidence_recall", "evidence_hit", "source_hit", "mrr"]
IN_SCOPE_GEN_METRICS = ["answered", "false_refusal", "has_citation", "citations_valid",
                        "cites_expected_source", "quotes_faithful", "code_support"]
OUT_OF_SCOPE_METRICS = ["correct_refusal", "clean_refusal"]
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def retrieval_metrics(item: dict, hits: list[dict]) -> dict:
    """Le retrieval a-t-il ramené les passages qui contiennent la réponse ?

    - evidence_recall : part des preuves présentes dans au moins un chunk récupéré ;
    - evidence_hit : au moins une preuve présente ;
    - source_hit : au moins un document attendu parmi les documents récupérés ;
    - mrr : 1 / rang du premier chunk contenant une preuve (0 si aucun).
    """
    texts = [norm(h["text"]) for h in hits]
    evidence = [norm(e) for e in item["evidence"]]
    found = [any(e in t for t in texts) for e in evidence]
    first_rank = next((r for r, t in enumerate(texts, 1) if any(e in t for e in evidence)), None)
    return {
        "evidence_recall": sum(found) / len(found) if found else 0.0,
        "evidence_hit": float(any(found)),
        "source_hit": float(bool(set(item["expected_sources"]) & {h["source"] for h in hits})),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def generation_metrics(item: dict, answer: str, hits: list[dict], evidence_hit: float | None = None) -> dict:
    """Vérifications déterministes de la réponse (voir rag_chain/checks.py)."""
    retrieved_sources = sorted({h["source"] for h in hits})
    chk = check_answer(answer, [h["text"] for h in hits], retrieved_sources)
    if item["type"] == "out_of_scope":
        return {"correct_refusal": float(chk.refusal), "clean_refusal": float(chk.clean_refusal)}

    answered = not chk.refusal
    row = {
        "answered": float(answered),
        # Refus alors que la preuve était dans le contexte : le modèle avait de quoi répondre.
        "false_refusal": float(chk.refusal and bool(evidence_hit)),
        "has_citation": float(chk.has_citation) if answered else np.nan,
        "citations_valid": float(chk.citations_valid) if answered else np.nan,
        "cites_expected_source": float(bool(set(chk.cited) & set(item["expected_sources"]))) if answered else np.nan,
        "quotes_faithful": float(chk.quotes_faithful) if answered else np.nan,
        "code_support": chk.code_support if answered else np.nan,
    }
    row["_unsupported_quotes"] = " | ".join(chk.unsupported_quotes)
    row["_unsupported_code"] = " | ".join(chk.unsupported_code)
    row["_cited"] = " ".join(chk.cited)
    return row


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float, int]:
    """Moyenne et intervalle de confiance bootstrap (percentile) ; ignore les NaN.
    Retourne (moyenne, borne basse, borne haute, n)."""
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(x.mean()), float(lo), float(hi), len(x))


def paired_bootstrap(a, b, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Différence appariée b - a (mêmes questions dans le même ordre), avec IC bootstrap et
    une p-valeur bilatérale approchée. Les paires contenant un NaN sont ignorées."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = b[mask] - a[mask]
    if len(d) == 0:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # p-valeur : proportion des moyennes bootstrap recentrées au moins aussi extrêmes.
    centered = boots - d.mean()
    p = float(np.mean(np.abs(centered) >= abs(d.mean())))
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": p, "n": int(len(d))}


def suggest_threshold(in_scope_scores: list[float], out_scope_scores: list[float]) -> dict:
    """Seuil sur le score du meilleur chunk qui maximise l'exactitude équilibrée
    (questions du périmètre conservées / hors périmètre rejetées)."""
    ins, outs = np.asarray(in_scope_scores), np.asarray(out_scope_scores)
    candidates = np.unique(np.concatenate([ins, outs]))
    best = {"threshold": None, "balanced_accuracy": 0.0}
    for t in candidates:
        bal = 0.5 * ((ins >= t).mean() + (outs < t).mean())
        if bal > best["balanced_accuracy"]:
            best = {"threshold": float(t), "balanced_accuracy": float(bal),
                    "in_scope_kept": float((ins >= t).mean()), "out_scope_rejected": float((outs < t).mean())}
    return best
