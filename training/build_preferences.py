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

Le run dure plusieurs heures : chaque question traitée est enregistrée au fur et à mesure dans
preferences_progress.jsonl, et une relance reprend là où le run s'était arrêté (mise en veille,
redémarrage de Windows...). --restart repart de zéro.

Usage :
    python training/build_preferences.py --samples 4
Sortie : training/data/preferences.jsonl (+ preferences_stats.json)
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_LLM_MODEL, TRAINING_DATA_DIR
from rag_chain import tracing
from rag_chain.chain import build_messages, generate
from retrieval.vector_store import search
from training.generate_questions import QUESTIONS_PATH
from training.reward import compute_reward

PREFERENCES_PATH = TRAINING_DATA_DIR / "preferences.jsonl"
PROGRESS_PATH = TRAINING_DATA_DIR / "preferences_progress.jsonl"
# Paramètres qui doivent être identiques pour qu'une reprise produise le même jeu de paires.
RUN_PARAMS = ("model", "samples", "temperature", "min_margin", "no_nli", "seed")


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


def load_progress(params: dict, question_ids: set[str]) -> tuple[set[str], Counter]:
    """Questions déjà traitées et statistiques cumulées d'un run interrompu.

    Refuse de reprendre si les paramètres ou les questions ont changé : le fichier de paires
    mélangerait alors deux runs différents.
    """
    done, stats = set(), Counter()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines or lines[0].get("params") != params:
        raise SystemExit(f"{PROGRESS_PATH.name} vient d'un run aux paramètres différents "
                         f"({lines[0].get('params') if lines else 'vide'}) : relancer avec --restart.")
    for entry in lines[1:]:
        if entry["id"] not in question_ids:
            raise SystemExit(f"La question {entry['id']} du run interrompu n'existe plus dans "
                             "questions.jsonl (questions régénérées ?) : relancer avec --restart.")
        done.add(entry["id"])
        stats.update(entry["stats"])
    return done, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", default=OLLAMA_LLM_MODEL, help="Modèle à améliorer (génère les candidats)")
    p.add_argument("--samples", type=int, default=4, help="Réponses échantillonnées par question (+1 greedy)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--min-margin", type=float, default=0.2)
    p.add_argument("--no-nli", action="store_true", help="Sans le score de grounding NLI (plus rapide)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true", help="Ignorer un run interrompu et repartir de zéro")
    args = p.parse_args(argv)

    questions = load_questions()[: args.limit] if args.limit else load_questions()
    params = {k: getattr(args, k) for k in RUN_PARAMS}
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS_PATH.exists() and not args.restart:
        done, stats = load_progress(params, {q["id"] for q in questions})
        # Une paire écrite juste avant l'interruption, sans sa ligne de progression, sera refaite :
        # on la retire pour ne pas l'avoir en double.
        if PREFERENCES_PATH.exists():
            kept = [line for line in PREFERENCES_PATH.read_text(encoding="utf-8").splitlines()
                    if line.strip() and json.loads(line)["id"] in done]
            PREFERENCES_PATH.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        print(f"Reprise d'un run interrompu : {len(done)} question(s) déjà traitée(s).")
    elif PREFERENCES_PATH.exists() and PREFERENCES_PATH.stat().st_size and not args.restart:
        raise SystemExit(f"{PREFERENCES_PATH} existe sans suivi de progression (run d'une version "
                         "précédente) : --restart pour l'écraser.")
    else:
        done, stats = set(), Counter()
        PREFERENCES_PATH.write_text("", encoding="utf-8")
        PROGRESS_PATH.write_text(json.dumps({"params": params}) + "\n", encoding="utf-8")

    nli = None
    if not args.no_nli:
        from training.reward import get_nli_scorer
        nli = get_nli_scorer()
        print(f"Modèle NLI sur {nli.device}" + (" (lent : PyTorch sans CUDA)" if nli.device == "cpu" else ""))

    start = time.time()
    session = time.strftime("dpo-preferences-%Y%m%d-%H%M")
    todo = [(n, q) for n, q in enumerate(questions, 1) if q["id"] not in done]
    with open(PREFERENCES_PATH, "a", encoding="utf-8") as out, \
            open(PROGRESS_PATH, "a", encoding="utf-8") as progress:
        for i, (n, q) in enumerate(todo, 1):
            with tracing.trace(f"dpo:{q['id']}", session_id=session, tags=["dpo", q["type"]],
                               metadata={"id": q["id"], "source_chunk_id": q.get("source_chunk_id")},
                               input=q["question"]):
                q_stats, outcome = process_question(n, q, args, nli, out)
                tracing.set_output(outcome, metadata=dict(q_stats))
            # La progression n'est écrite qu'une fois la paire enregistrée : une question
            # interrompue en cours de route sera simplement refaite à la reprise.
            progress.write(json.dumps({"id": q["id"], "stats": q_stats}, ensure_ascii=False) + "\n")
            progress.flush()
            stats.update(q_stats)
            remaining = (time.time() - start) / i * (len(todo) - i)
            print(f"[{n}/{len(questions)}] {outcome} — reste ~{remaining / 60:.0f} min — {q['question'][:60]}")
    tracing.flush()

    stats_path = TRAINING_DATA_DIR / "preferences_stats.json"
    stats_path.write_text(json.dumps(dict(stats), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items())))
    print(f"-> {PREFERENCES_PATH}")


def process_question(n: int, q: dict, args, nli, out) -> tuple[Counter, str]:
    """Traite une question : (statistiques de la question, résumé du résultat)."""
    q_stats = Counter()
    hits = search(q["question"])
    if q["type"] == "in_scope" and q["source_chunk_id"] not in {h["id"] for h in hits}:
        q_stats["skipped_source_not_retrieved"] += 1
        outcome = "écartée (passage source non récupéré)"
    else:
        answers = sample_answers(q["question"], hits, args.model, args.samples, args.temperature,
                                 args.seed + 1000 * n)
        source_doc = q.get("source")
        scored = [(a, compute_reward(a, q["type"], hits, source_doc, nli)) for a in answers]
        for _, r in scored:
            q_stats[f"hard_fail:{r.hard_fail}" if r.hard_fail else "passed"] += 1
        pair = select_pair(scored, args.min_margin)
        if pair is None:
            q_stats["no_pair"] += 1
            outcome = "sans paire"
        else:
            (chosen, rc), (rejected, rr) = pair
            q_stats[f"pairs_{q['type']}"] += 1
            q_stats[f"rejected_reason:{rr.hard_fail or 'lower_score'}"] += 1
            out.write(json.dumps({
                "id": q["id"], "type": q["type"], "question": q["question"],
                "prompt": build_messages(q["question"], hits),
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "chosen_reward": rc.to_dict(), "rejected_reward": rr.to_dict(),
                "n_candidates": len(answers),
            }, ensure_ascii=False) + "\n")
            out.flush()
            outcome = f"paire {rc.total:.2f} vs {rr.total:.2f} ({rr.hard_fail or 'score'})"
            tracing.score("reward_chosen", rc.total)
            tracing.score("reward_rejected", rr.total)
    return q_stats, outcome


if __name__ == "__main__":
    main()
