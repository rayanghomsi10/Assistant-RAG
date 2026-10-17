"""
Re-sélectionne les paires de préférences à partir des réponses candidates déjà notées, sans rien
regénérer : utile pour essayer une autre règle de sélection (écart minimal, écart sur le fond).

build_preferences.py enregistre chaque réponse candidate et sa récompense dans
training/data/preferences_candidates.jsonl ; ce script refait la sélection et réécrit
preferences.jsonl et preferences_stats.json (après une copie de sauvegarde des fichiers existants).

    python training/select_pairs.py --min-margin 0.2 --min-substance 0.1
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAINING_DATA_DIR
from training.build_preferences import CANDIDATES_PATH, PREFERENCES_PATH, pair_record, pair_stats, select_pair
from training.reward import Reward


def reselect(records: list[dict], min_margin: float, min_substance: float | None) -> tuple[list[dict], Counter]:
    pairs, stats = [], Counter()
    for rec in records:
        scored = [(c["answer"], Reward(**c["reward"])) for c in rec["candidates"]]
        pair = select_pair(scored, min_margin, min_substance)
        stats.update(pair_stats(rec["type"], scored, pair))
        if pair is not None:
            pairs.append(pair_record(rec, rec["prompt"], pair, len(scored)))
    return pairs, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--min-margin", type=float, default=0.2)
    p.add_argument("--min-substance", type=float, default=0.1)
    p.add_argument("--no-substance", action="store_true", help="Aucune condition sur le fond (règle du premier run)")
    args = p.parse_args(argv)

    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"{CANDIDATES_PATH} introuvable : lancer d'abord training/build_preferences.py.")
    records = [json.loads(line) for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    min_substance = None if args.no_substance else args.min_substance
    pairs, stats = reselect(records, args.min_margin, min_substance)

    stats_path = TRAINING_DATA_DIR / "preferences_stats.json"
    for path in (PREFERENCES_PATH, stats_path):
        if path.exists():
            shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
    PREFERENCES_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs), encoding="utf-8")
    stats_path.write_text(json.dumps(dict(stats), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(records)} questions avec candidats -> {len(pairs)} paires "
          f"(écart minimal {args.min_margin}, écart sur le fond {min_substance})")
    print("\n".join(f"  {k}: {v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
