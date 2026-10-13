"""
Boucle d'évaluation : métriques déterministes + Ragas + intervalles de confiance + MLflow.

Trois familles de métriques, sur le jeu evaluation/eval_dataset.json (tenu à l'écart de
l'entraînement DPO) :

1. Retrieval, sans LLM (questions du périmètre) : evidence_recall, evidence_hit, source_hit, mrr.
   Calculées à partir des preuves recopiées du corpus : exactes, reproductibles, instantanées.
2. Génération, sans LLM juge : refus corrects hors périmètre, faux refus dans le périmètre,
   validité des sources citées, fidélité des citations entre guillemets, identifiants de code
   présents dans le contexte (voir rag_chain/checks.py).
3. Ragas (questions du périmètre uniquement) : faithfulness, answer_relevancy, context_precision,
   avec un juge **distinct** du générateur (JUDGE_LLM_MODEL) et des embeddings multilingues.
   Les questions hors périmètre sont exclues de Ragas : un refus correct y valait 0 par
   construction et tirait les moyennes vers le bas en V1 ; il est maintenant mesuré à part.

Chaque moyenne est accompagnée d'un intervalle de confiance bootstrap à 95 % sur les questions.
Pour comparer deux runs, utiliser compare_runs.py (bootstrap apparié).

Exemples :
    python evaluation/run_eval.py --retrieval-only              # rapide, sans LLM
    python evaluation/run_eval.py --run-name baseline_v2
    python evaluation/run_eval.py --run-name dpo_v1 --model rag-assistant-dpo
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JUDGE_LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_NUM_CTX,
    RELEVANCE_THRESHOLD,
    RERANKER_MODEL_NAME,
    RETRIEVAL_CANDIDATES,
    TOP_K,
)
from evaluation.dataset import (
    EVAL_DATASET_PATH,
    dataset_fingerprint,
    load_corpus_texts,
    load_eval_dataset,
    validate_dataset,
)
from evaluation.metrics import (
    IN_SCOPE_GEN_METRICS,
    OUT_OF_SCOPE_METRICS,
    RAGAS_METRICS,
    RETRIEVAL_METRICS,
    bootstrap_ci,
    generation_metrics,
    retrieval_metrics,
    suggest_threshold,
)
from rag_chain.chain import below_threshold, generate
from rag_chain.checks import REFUSAL_MESSAGE
from retrieval.vector_store import get_index_metadata, search

RESULTS_DIR = Path(__file__).parent / "results"


def safe_name(run_name: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_name) if run_name else time.strftime("run_%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------------------------

def run_pipeline(items: list[dict], model: str, repeats: int, retrieval_only: bool) -> pd.DataFrame:
    hits_cache = {}  # le retrieval est déterministe : inutile de le refaire à chaque répétition
    rows = []
    for rep in range(repeats if not retrieval_only else 1):
        for n, it in enumerate(items, 1):
            if it["id"] not in hits_cache:
                hits_cache[it["id"]] = search(it["question"])
            hits = hits_cache[it["id"]]
            row = {"id": it["id"], "type": it["type"], "question": it["question"], "repeat": rep,
                   "top_score": hits[0]["score"] if hits else np.nan,
                   "retrieved_ids": " ".join(h["id"] for h in hits)}
            if it["type"] == "in_scope":
                row.update(retrieval_metrics(it, hits))
            if not retrieval_only:
                print(f"  [rep {rep + 1}/{repeats}] [{n}/{len(items)}] {it['question']}")
                start = time.time()
                gated = below_threshold(hits)
                answer = REFUSAL_MESSAGE if gated else generate(it["question"], hits, model=model)
                row.update({"answer": answer, "refused_by_threshold": gated,
                            "latency_s": round(time.time() - start, 2),
                            "contexts": json.dumps([h["text"] for h in hits], ensure_ascii=False)})
                row.update(generation_metrics(it, answer, hits, row.get("evidence_hit")))
            rows.append(row)
    return pd.DataFrame(rows)


def run_ragas(df: pd.DataFrame, items_by_id: dict) -> pd.DataFrame:
    """Ajoute les scores Ragas aux lignes du périmètre (une évaluation par répétition)."""
    # Doit précéder `import ragas` ; « isort: split » empêche le tri de le redescendre.
    from evaluation import _ragas_compat  # noqa: F401
    # isort: split
    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from langchain_ollama import ChatOllama
    from ragas import RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    from ingestion.embeddings import embed_queries

    class QueryEmbeddings(Embeddings):
        """answer_relevancy compare la question d'origine à des questions régénérées à partir de
        la réponse : des requêtes des deux côtés, donc le préfixe « query: » des deux côtés."""
        def embed_documents(self, texts):
            return embed_queries(texts)

        def embed_query(self, text):
            return embed_queries([text])[0]

    judge = LangchainLLMWrapper(ChatOllama(base_url=OLLAMA_BASE_URL, model=JUDGE_LLM_MODEL,
                                           temperature=0.0, num_ctx=OLLAMA_NUM_CTX))
    embeddings = LangchainEmbeddingsWrapper(QueryEmbeddings())
    # Ollama local sert les requêtes en série (OLLAMA_NUM_PARALLEL=1 par défaut) : la concurrence
    # par défaut de Ragas (16) provoque des TimeoutError (voir README, limites connues).
    run_config = RunConfig(timeout=300, max_workers=2, max_retries=3, max_wait=30)

    df = df.copy()
    for m in RAGAS_METRICS:
        df[m] = np.nan
    mask_all = df["type"] == "in_scope"
    for rep in sorted(df["repeat"].unique()):
        mask = mask_all & (df["repeat"] == rep)
        sub = df[mask]
        dataset = Dataset.from_list([{
            "user_input": r.question,
            "response": r.answer,
            "retrieved_contexts": json.loads(r.contexts),
            "reference": items_by_id[r.id]["ground_truth"],
        } for r in sub.itertuples()])
        print(f"\nRagas (juge = {JUDGE_LLM_MODEL}), répétition {rep + 1} : {len(sub)} questions ...")
        res = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision],
                       llm=judge, embeddings=embeddings, run_config=run_config).to_pandas()
        for m in RAGAS_METRICS:
            df.loc[mask, m] = res[m].to_numpy()
    return df


# ---------------------------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> dict:
    """Moyenne par question sur les répétitions, puis moyenne + IC bootstrap sur les questions."""
    summary = {}
    groups = [("in_scope", RETRIEVAL_METRICS + IN_SCOPE_GEN_METRICS + RAGAS_METRICS),
              ("out_of_scope", OUT_OF_SCOPE_METRICS)]
    for typ, metrics in groups:
        sub = df[df["type"] == typ]
        for m in metrics:
            if m not in sub:
                continue
            per_q = sub.groupby("id")[m].mean()
            mean, lo, hi, n = bootstrap_ci(per_q.to_numpy())
            entry = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n,
                     "n_missing": int(per_q.isna().sum())}
            if sub["repeat"].nunique() > 1:
                entry["std_between_repeats"] = float(sub.groupby("repeat")[m].mean().std())
            summary[m] = entry
    if "latency_s" in df:
        summary["latency_s_median"] = {"mean": float(df["latency_s"].median())}
    return summary


def print_summary(summary: dict) -> None:
    print("\n--- Scores (moyenne [IC 95 %] — n questions) ---")
    for m, s in summary.items():
        if "ci_low" in s:
            miss = f", {s['n_missing']} manquant(s)" if s.get("n_missing") else ""
            rep = f", σ entre répétitions {s['std_between_repeats']:.3f}" if "std_between_repeats" in s else ""
            print(f"  {m:24s} {s['mean']:.3f} [{s['ci_low']:.3f} ; {s['ci_high']:.3f}] — n={s['n']}{miss}{rep}")
        else:
            print(f"  {m:24s} {s['mean']:.2f}")


def retrieval_report(df: pd.DataFrame) -> dict:
    ins = df[df["type"] == "in_scope"].drop_duplicates("id")["top_score"].tolist()
    outs = df[df["type"] == "out_of_scope"].drop_duplicates("id")["top_score"].tolist()
    if not ins or not outs:  # --limit peut ne garder qu'un seul des deux types de questions
        return {"threshold": None, "balanced_accuracy": 0.0}
    best = suggest_threshold(ins, outs)
    print("\n--- Score du meilleur chunk (calibrage de RELEVANCE_THRESHOLD) ---")
    print(f"  dans le périmètre : min {np.min(ins):.3f}  p10 {np.quantile(ins, .1):.3f}  médiane {np.median(ins):.3f}")
    print(f"  hors périmètre    : médiane {np.median(outs):.3f}  p90 {np.quantile(outs, .9):.3f}  "
          f"max {np.max(outs):.3f}")
    if best["threshold"] is not None:
        print(f"  seuil suggéré : {best['threshold']:.3f} (exactitude équilibrée {best['balanced_accuracy']:.2f} ; "
              f"{best['in_scope_kept']:.0%} du périmètre conservé, "
              f"{best['out_scope_rejected']:.0%} du hors-périmètre rejeté)")
        print("  -> Ne l'activer que si le périmètre est conservé à ~100 % : un faux rejet coûte une réponse.")
    return best


# ---------------------------------------------------------------------------------------------

def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Évaluation du pipeline RAG")
    p.add_argument("--run-name", default=None, help="Nom du run (ex. 'baseline_v2')")
    p.add_argument("--model", default=OLLAMA_LLM_MODEL, help="Modèle Ollama générateur à évaluer")
    p.add_argument("--repeats", type=int, default=1, help="Répétitions complètes (génération + juge)")
    p.add_argument("--retrieval-only", action="store_true", help="Uniquement le retrieval (sans LLM)")
    p.add_argument("--skip-ragas", action="store_true", help="Sans Ragas (métriques déterministes seules)")
    p.add_argument("--limit", type=int, default=None, help="N premières questions (test rapide)")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)

    items = load_eval_dataset()
    problems = validate_dataset(items, load_corpus_texts())
    if problems:
        raise SystemExit("Jeu d'évaluation invalide :\n" + "\n".join(problems))
    if args.limit:
        items = items[: args.limit]
    items_by_id = {it["id"]: it for it in items}
    n_in = sum(it["type"] == "in_scope" for it in items)
    print(f"{len(items)} questions ({n_in} dans le périmètre, {len(items) - n_in} hors périmètre), "
          f"générateur = {args.model}\n")

    df = run_pipeline(items, args.model, args.repeats, args.retrieval_only)
    if not (args.retrieval_only or args.skip_ragas):
        df = run_ragas(df, items_by_id)

    summary = summarize(df)
    print_summary(summary)
    threshold_info = retrieval_report(df)

    run = safe_name(args.run_name)
    out_dir = RESULTS_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, summary_path = out_dir / "per_question.csv", out_dir / "summary.json"
    df.to_csv(csv_path, index=False)

    index_meta = get_index_metadata()
    params = {
        **{f"index_{k}": v for k, v in index_meta.items()},
        "generator_model": args.model if not args.retrieval_only else "",
        "judge_model": JUDGE_LLM_MODEL if not (args.retrieval_only or args.skip_ragas) else "",
        "top_k": TOP_K, "retrieval_candidates": RETRIEVAL_CANDIDATES,
        "reranker": RERANKER_MODEL_NAME or "none",
        "relevance_threshold": RELEVANCE_THRESHOLD if RELEVANCE_THRESHOLD is not None else "none",
        "num_ctx": OLLAMA_NUM_CTX, "repeats": args.repeats, "n_questions": len(items),
        "eval_dataset_sha": dataset_fingerprint(EVAL_DATASET_PATH),
        "mode": "retrieval_only" if args.retrieval_only else ("no_ragas" if args.skip_ragas else "full"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"params": params, "metrics": summary, "threshold_suggestion": threshold_info},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\nRésultats : {csv_path} et {summary_path}")

    if not args.no_mlflow:
        from evaluation.mlflow_tracking import log_eval_run
        flat = {}
        for m, s in summary.items():
            for k in ("mean", "ci_low", "ci_high"):
                if k in s:
                    flat[m if k == "mean" else f"{m}_{k}"] = s[k]
        log_eval_run(params, flat, [csv_path, summary_path, EVAL_DATASET_PATH], run_name=run)
    return summary


if __name__ == "__main__":
    main()
