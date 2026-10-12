"""
Compare deux runs d'évaluation question par question (bootstrap apparié).

Une différence de moyennes ne suffit pas : sur quelques dizaines de questions, quelques
basculements suffisent à la produire. Le bootstrap apparié rééchantillonne les *différences par
question* : si l'intervalle de confiance de la différence exclut 0, l'écart dépasse le bruit
d'échantillonnage des questions.

Usage :
    python evaluation/compare_runs.py baseline_v2 dpo_v1
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import (IN_SCOPE_GEN_METRICS, OUT_OF_SCOPE_METRICS, RAGAS_METRICS,
                                RETRIEVAL_METRICS, paired_bootstrap)

RESULTS_DIR = Path(__file__).parent / "results"
# Métriques où une baisse est une amélioration.
LOWER_IS_BETTER = {"false_refusal"}


def load_run(name: str) -> pd.DataFrame:
    path = Path(name)
    if not path.exists():
        path = RESULTS_DIR / name / "per_question.csv"
    return pd.read_csv(path)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("in_scope", RETRIEVAL_METRICS + IN_SCOPE_GEN_METRICS + RAGAS_METRICS),
              ("out_of_scope", OUT_OF_SCOPE_METRICS)]
    for typ, metrics in groups:
        for m in metrics:
            if m not in a or m not in b:
                continue
            pa = a[a["type"] == typ].groupby("id")[m].mean()
            pb = b[b["type"] == typ].groupby("id")[m].mean()
            common = pa.index.intersection(pb.index)
            if len(common) == 0:
                continue
            res = paired_bootstrap(pa.loc[common].to_numpy(), pb.loc[common].to_numpy())
            significant = res["n"] > 0 and (res["ci_low"] > 0 or res["ci_high"] < 0)
            better = None
            if significant:
                better = (res["delta"] < 0) if m in LOWER_IS_BETTER else (res["delta"] > 0)
            rows.append({"metric": m, "A": pa.loc[common].mean(), "B": pb.loc[common].mean(), **res,
                         "verdict": ("B meilleur" if better else "B moins bon") if significant else "non significatif"})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("run_a", help="run de référence (nom de dossier dans evaluation/results ou chemin CSV)")
    p.add_argument("run_b", help="run comparé")
    args = p.parse_args(argv)
    table = compare(load_run(args.run_a), load_run(args.run_b))
    pd.set_option("display.width", 160)
    print(f"A = {args.run_a}   B = {args.run_b}   (delta = B - A, IC 95 % bootstrap apparié)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table


if __name__ == "__main__":
    main()
