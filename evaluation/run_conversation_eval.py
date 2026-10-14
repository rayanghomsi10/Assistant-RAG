"""
Évaluation de la mémoire de conversation : la reformulation des relances retrouve-t-elle le bon passage ?

evaluation/conversations.json contient des relances elliptiques (« Laquelle est utilisée par
défaut ? ») qui suivent une question du jeu d'évaluation. Chacune est rattachée à une question
« cible » du même jeu, dont les preuves sont connues : on mesure le retrieval sans LLM juge.

Trois recherches par relance :
- brute : la relance telle quelle (ce que faisait le système sans mémoire) ;
- reformulée : la question autonome produite par condense_question (le système actuel) ;
- oracle : la question cible écrite en entier (plafond atteignable).

L'historique contient la question précédente et sa réponse de référence (ground_truth), pour
ne mesurer que la reformulation, pas la qualité d'une réponse générée.

    python evaluation/run_conversation_eval.py --run-name memoire_v1
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_LLM_MODEL
from evaluation.dataset import load_eval_dataset
from evaluation.metrics import bootstrap_ci, paired_bootstrap, retrieval_metrics
from rag_chain import tracing
from rag_chain.chain import condense_question
from retrieval.vector_store import search

CONVERSATIONS_PATH = Path(__file__).parent / "conversations.json"
RESULTS_DIR = Path(__file__).parent / "results"
METRICS = ["evidence_hit", "evidence_recall", "mrr", "source_hit"]


def load_conversations(path: Path = CONVERSATIONS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_conversations(conversations: list[dict], items_by_id: dict) -> list[str]:
    problems, seen = [], set()
    for c in conversations:
        if c["id"] in seen:
            problems.append(f"{c['id']} : identifiant en double")
        seen.add(c["id"])
        for ref in c["history"] + [c["target"]]:
            if ref not in items_by_id or items_by_id[ref]["type"] != "in_scope":
                problems.append(f"{c['id']} : {ref} n'est pas une question du périmètre du jeu d'évaluation")
    return problems


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description="Évaluation de la reformulation des relances")
    p.add_argument("--run-name", default=time.strftime("conversations_%Y%m%d_%H%M%S"))
    p.add_argument("--model", default=OLLAMA_LLM_MODEL, help="Modèle qui reformule les relances")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    items_by_id = {it["id"]: it for it in load_eval_dataset()}
    conversations = load_conversations()[: args.limit] if args.limit else load_conversations()
    problems = validate_conversations(conversations, items_by_id)
    if problems:
        raise SystemExit("Jeu de conversations invalide :\n" + "\n".join(problems))

    rows = []
    for n, c in enumerate(conversations, 1):
        target = items_by_id[c["target"]]
        history = [{"question": items_by_id[h]["question"], "answer": items_by_id[h]["ground_truth"]}
                   for h in c["history"]]
        with tracing.trace(f"conversation:{c['id']}", session_id=args.run_name, tags=["evaluation", "conversation"],
                           input=c["question"]):
            standalone = condense_question(c["question"], history, model=args.model)
            tracing.set_output(standalone)
        row = {"id": c["id"], "question": c["question"], "standalone": standalone, "target": c["target"]}
        for mode, query in (("brute", c["question"]), ("reformulee", standalone), ("oracle", target["question"])):
            for m, v in retrieval_metrics(target, search(query)).items():
                row[f"{m}_{mode}"] = v
        rows.append(row)
        print(f"[{n}/{len(conversations)}] {c['question']}\n    -> {standalone}  "
              f"(preuve : brute {row['evidence_hit_brute']:.0f}, reformulée {row['evidence_hit_reformulee']:.0f})")
    tracing.flush()

    df = pd.DataFrame(rows)
    summary = {}
    print("\n--- Retrieval sur la relance (moyenne [IC 95 %]) ---")
    print(f"  {'métrique':16s} {'brute':>22s} {'reformulée':>22s} {'oracle':>22s}   écart reformulée - brute")
    for m in METRICS:
        cells = {}
        for mode in ("brute", "reformulee", "oracle"):
            mean, lo, hi, _ = bootstrap_ci(df[f"{m}_{mode}"].to_numpy())
            cells[mode] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        diff = paired_bootstrap(df[f"{m}_brute"].to_numpy(), df[f"{m}_reformulee"].to_numpy())
        summary[m] = {**cells, "delta_reformulee_vs_brute": diff}
        fmt = lambda c: f"{c['mean']:.3f} [{c['ci_low']:.2f} ; {c['ci_high']:.2f}]"  # noqa: E731
        verdict = "significatif" if diff["ci_low"] > 0 or diff["ci_high"] < 0 else "non significatif"
        print(f"  {m:16s} {fmt(cells['brute']):>22s} {fmt(cells['reformulee']):>22s} {fmt(cells['oracle']):>22s}"
              f"   {diff['delta']:+.3f} [{diff['ci_low']:+.2f} ; {diff['ci_high']:+.2f}] {verdict}")

    out_dir = RESULTS_DIR / re.sub(r"[^A-Za-z0-9_.-]", "_", args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_conversation.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({"model": args.model, "n": len(df), "metrics": summary},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultats : {out_dir}")
    return df


if __name__ == "__main__":
    main()
