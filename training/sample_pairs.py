"""
Tire un échantillon de paires de préférences à relire avant l'entraînement (étape obligatoire).

Écrit training/data/pairs_sample.md : pour chaque paire, la question, la réponse retenue et la
réponse rejetée, avec le détail de leur récompense. À noter ✓ ou ✗ : la réponse retenue doit être
réellement meilleure, et pas seulement mieux citée, plus courte ou plus évasive.

    python training/sample_pairs.py --n 30
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_DATA_DIR
from training.build_preferences import PREFERENCES_PATH

SAMPLE_PATH = TRAINING_DATA_DIR / "pairs_sample.md"


def components(reward: dict) -> str:
    if reward["hard_fail"]:
        return f"contrainte dure : {reward['hard_fail']}"
    return ", ".join(f"{k} {v:.2f}" for k, v in reward["components"].items())


def render(pairs: list[dict], seed: int) -> str:
    out = [f"# {len(pairs)} paires de préférences tirées au hasard (graine {seed})", "",
           "Pour chaque paire : la réponse retenue est-elle réellement meilleure que la rejetée, et pas "
           "seulement mieux citée, plus courte ou plus évasive ? Noter ✓ ou ✗ à côté du numéro.", ""]
    for i, p in enumerate(pairs, 1):
        rc, rr = p["chosen_reward"], p["rejected_reward"]
        out += [f"## {i}. {p['question']}", "",
                f"*{p['id']}* — retenue {rc['total']:.2f} contre rejetée {rr['total']:.2f} "
                f"({rr['hard_fail'] or 'score plus bas'})", "",
                f"**Retenue** ({components(rc)})", "", p["chosen"][0]["content"].strip(), "",
                f"**Rejetée** ({components(rr)})", "", p["rejected"][0]["content"].strip(), "", "---", ""]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    pairs = [json.loads(line) for line in PREFERENCES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = random.Random(args.seed).sample(pairs, min(args.n, len(pairs)))
    SAMPLE_PATH.write_text(render(sample, args.seed), encoding="utf-8")
    print(f"{len(sample)} paires sur {len(pairs)} -> {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
