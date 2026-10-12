"""
Étape DPO 2/4 : paires de préférences (chosen / rejected), sur la machine locale avec Ollama.

Pour chaque question d'entraînement :
  1. retrieval avec le pipeline réel (mêmes chunks, même reranker qu'en production) ;
  2. si c'est une question du périmètre dont le chunk source n'a PAS été récupéré, on l'écarte :
     on ne peut pas savoir si un refus serait correct, la paire serait ambiguë ;
  3. génération de plusieurs réponses par le modèle déployé (une en greedy + K échantillonnées) :
     les données sont « on-policy », elles reflètent les erreurs réelles du modèle ;
  4. score de chaque réponse par la récompense vérifiable (training/reward.py) ;
  5. paire = meilleure réponse (qui doit respecter toutes les contraintes dures) contre la moins
     bonne, si l'écart de récompense dépasse --min-margin.

Le prompt enregistré est exactement celui de l'inférence (rag_chain.chain.build_messages).

Usage :
    python training/build_preferences.py --samples 4
Sortie : training/data/preferences.jsonl (+ preferences_stats.json)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_LLM_MODEL, TRAINING_DATA_DIR
from rag_chain.chain import build_messages, generate
from retrieval.vector_store import search
from training.generate_questions import QUESTIONS_PATH
from training.reward import compute_reward

PREFERENCES_PATH = TRAINING_DATA_DIR / "preferences.jsonl"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_answers(question: str, hits: list[dict], model: str, n_samples: int, temperature: float,
                   seed: int) -> list[str]:
    answers = [generate(question, hits, model=model, temperature=0.0)]
    answers += [generate(question, hits, model=model, temperature=temperature, seed=seed + i)
                for i in range(n_samples)]
    # Doublons exacts retirés (fréquents à température basse).
    seen, unique = set(), []
    for a in answers:
        if a.strip() not in seen:
            seen.add(a.strip())
            unique.append(a)
    return unique


def select_pair(scored: list[tuple[str, object]], min_margin: float):
    """(chosen, rejected) ou None. Le chosen doit respecter toutes les contraintes dures."""
    valid = [(a, r) for a, r in scored if r.passed]
    if not valid:
        return None
    chosen = max(valid, key=lambda x: x[1].total)
    rejected = min(scored, key=lambda x: x[1].total)
    if chosen[1].total - rejected[1].total < min_margin:
        return None
    return chosen, rejected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", default=OLLAMA_LLM_MODEL, help="Modèle à améliorer (génère les candidats)")
    p.add_argument("--samples", type=int, default=4, help="Réponses échantillonnées par question (+1 greedy)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--min-margin", type=float, default=0.2)
    p.add_argument("--no-nli", action="store_true", help="Sans le score de grounding NLI (plus rapide)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    nli = None
    if not args.no_nli:
        from training.reward import get_nli_scorer
        nli = get_nli_scorer()

    questions = load_questions()[: args.limit] if args.limit else load_questions()
    stats = Counter()
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREFERENCES_PATH, "w", encoding="utf-8") as out:
        for n, q in enumerate(questions, 1):
            hits = search(q["question"])
            if q["type"] == "in_scope" and q["source_chunk_id"] not in {h["id"] for h in hits}:
                stats["skipped_source_not_retrieved"] += 1
                continue
            answers = sample_answers(q["question"], hits, args.model, args.samples, args.temperature,
                                     args.seed + 1000 * n)
            source_doc = q.get("source")
            scored = [(a, compute_reward(a, q["type"], hits, source_doc, nli)) for a in answers]
            for _, r in scored:
                stats[f"hard_fail:{r.hard_fail}" if r.hard_fail else "passed"] += 1
            pair = select_pair(scored, args.min_margin)
            if pair is None:
                stats["no_pair"] += 1
                continue
            (chosen, rc), (rejected, rr) = pair
            stats[f"pairs_{q['type']}"] += 1
            stats[f"rejected_reason:{rr.hard_fail or 'lower_score'}"] += 1
            out.write(json.dumps({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "prompt": build_messages(q["question"], hits),
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "chosen_reward": rc.to_dict(), "rejected_reward": rr.to_dict(),
                "n_candidates": len(answers),
            }, ensure_ascii=False) + "\n")
            print(f"[{n}/{len(questions)}] paire {rc.total:.2f} vs {rr.total:.2f} "
                  f"({rr.hard_fail or 'score'}) — {q['question'][:70]}")

    stats_path = TRAINING_DATA_DIR / "preferences_stats.json"
    stats_path.write_text(json.dumps(dict(stats), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items())))
    print(f"-> {PREFERENCES_PATH}")


if __name__ == "__main__":
    main()
