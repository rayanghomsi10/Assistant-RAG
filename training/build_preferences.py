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
# Toutes les réponses candidates et leur récompense : permet de re-sélectionner les paires avec
# d'autres règles (training/select_pairs.py) sans regénérer des heures de réponses.
CANDIDATES_PATH = TRAINING_DATA_DIR / "preferences_candidates.jsonl"
# Paramètres qui doivent être identiques pour qu'une reprise produise le même jeu de paires.
RUN_PARAMS = ("model", "samples", "temperature", "min_margin", "min_substance", "no_nli", "seed")
# Composantes de la récompense qui portent sur le fond (les deux autres portent sur la citation).
SUBSTANCE_COMPONENTS = ("grounding", "code_support", "concision")


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


def substance_margin(rc, rr) -> float:
    """Écart de récompense sur le fond seulement (ancrage, code, concision), pondéré comme dans la
    récompense. Les composantes absentes (NLI désactivé) sont ignorées."""
    from training.reward import WEIGHTS
    keys = [k for k in SUBSTANCE_COMPONENTS if k in rc.components and k in rr.components]
    if not keys:
        return 0.0
    return sum(WEIGHTS[k] * (rc.components[k] - rr.components[k]) for k in keys) / sum(WEIGHTS[k] for k in keys)


def select_pair(scored: list[tuple[str, object]], min_margin: float, min_substance: float | None = None):
    """(chosen, rejected) ou None. Le chosen est la meilleure réponse qui respecte toutes les
    contraintes dures. Le rejected est la moins bonne réponse qui laisse un écart d'au moins
    `min_margin` ; si elle respecte elle-même les contraintes dures, l'écart doit aussi porter sur
    le fond (`min_substance`), et pas seulement sur la présence d'une citation : sinon la paire
    apprendrait à citer au détriment du contenu (relecture du premier run, 48 paires sur 164).
    `min_substance=None` : aucune condition sur le fond (règle du premier run)."""
    valid = [(a, r) for a, r in scored if r.passed]
    if not valid:
        return None
    chosen = max(valid, key=lambda x: x[1].total)
    for a, r in sorted(scored, key=lambda x: x[1].total):
        if chosen[1].total - r.total < min_margin:
            break                       # les suivantes sont encore plus proches du chosen
        if r.hard_fail or min_substance is None or substance_margin(chosen[1], r) >= min_substance:
            return chosen, (a, r)
    return None


def pair_record(q: dict, prompt: list[dict], pair, n_candidates: int) -> dict:
    (chosen, rc), (rejected, rr) = pair
    return {"id": q["id"], "type": q["type"], "question": q["question"], "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "chosen_reward": rc.to_dict(), "rejected_reward": rr.to_dict(), "n_candidates": n_candidates}


def pair_stats(q_type: str, scored, pair) -> Counter:
    stats = Counter()
    for _, r in scored:
        stats[f"hard_fail:{r.hard_fail}" if r.hard_fail else "passed"] += 1
    if pair is None:
        stats["no_pair"] += 1
    else:
        stats[f"pairs_{q_type}"] += 1
        stats[f"rejected_reason:{pair[1][1].hard_fail or 'lower_score'}"] += 1
    return stats


def _keep_done(path: Path, done: set[str]) -> None:
    """Retire d'un fichier JSONL les lignes des questions non terminées (reprise sans doublon)."""
    if path.exists():
        kept = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["id"] in done]
        path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


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
    p.add_argument("--min-substance", type=float, default=0.1,
                   help="Écart minimal sur le fond quand la réponse rejetée respecte les contraintes dures")
    p.add_argument("--no-substance", action="store_true", help="Aucune condition sur le fond (règle du premier run)")
    p.add_argument("--no-nli", action="store_true", help="Sans le score de grounding NLI (plus rapide)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restart", action="store_true", help="Ignorer un run interrompu et repartir de zéro")
    args = p.parse_args(argv)
    if args.no_substance:
        args.min_substance = None

    questions = load_questions()[: args.limit] if args.limit else load_questions()
    params = {k: getattr(args, k) for k in RUN_PARAMS}
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROGRESS_PATH.exists() and not args.restart:
        done, stats = load_progress(params, {q["id"] for q in questions})
        # Une paire écrite juste avant l'interruption, sans sa ligne de progression, sera refaite :
        # on la retire pour ne pas l'avoir en double.
        _keep_done(PREFERENCES_PATH, done)
        _keep_done(CANDIDATES_PATH, done)
        print(f"Reprise d'un run interrompu : {len(done)} question(s) déjà traitée(s).")
    elif PREFERENCES_PATH.exists() and PREFERENCES_PATH.stat().st_size and not args.restart:
        raise SystemExit(f"{PREFERENCES_PATH} existe sans suivi de progression (run d'une version "
                         "précédente) : --restart pour l'écraser.")
    else:
        done, stats = set(), Counter()
        PREFERENCES_PATH.write_text("", encoding="utf-8")
        CANDIDATES_PATH.write_text("", encoding="utf-8")
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
            open(CANDIDATES_PATH, "a", encoding="utf-8") as cand, \
            open(PROGRESS_PATH, "a", encoding="utf-8") as progress:
        for i, (n, q) in enumerate(todo, 1):
            with tracing.trace(f"dpo:{q['id']}", session_id=session, tags=["dpo", q["type"]],
                               metadata={"id": q["id"], "source_chunk_id": q.get("source_chunk_id")},
                               input=q["question"]):
                q_stats, outcome = process_question(n, q, args, nli, out, cand)
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


def process_question(n: int, q: dict, args, nli, out, cand=None) -> tuple[Counter, str]:
    """Traite une question : (statistiques de la question, résumé du résultat)."""
    hits = search(q["question"])
    if q["type"] == "in_scope" and q["source_chunk_id"] not in {h["id"] for h in hits}:
        return Counter(skipped_source_not_retrieved=1), "écartée (passage source non récupéré)"
    answers = sample_answers(q["question"], hits, args.model, args.samples, args.temperature,
                             args.seed + 1000 * n)
    scored = [(a, compute_reward(a, q["type"], hits, q.get("source"), nli)) for a in answers]
    prompt = build_messages(q["question"], hits)
    if cand is not None:
        cand.write(json.dumps({"id": q["id"], "type": q["type"], "question": q["question"], "prompt": prompt,
                               "candidates": [{"answer": a, "reward": r.to_dict()} for a, r in scored]},
                              ensure_ascii=False) + "\n")
        cand.flush()
    pair = select_pair(scored, args.min_margin, args.min_substance)
    q_stats = pair_stats(q["type"], scored, pair)
    if pair is None:
        return q_stats, "sans paire"
    out.write(json.dumps(pair_record(q, prompt, pair, len(answers)), ensure_ascii=False) + "\n")
    out.flush()
    (_, rc), (_, rr) = pair
    tracing.score("reward_chosen", rc.total)
    tracing.score("reward_rejected", rr.total)
    return q_stats, f"paire {rc.total:.2f} vs {rr.total:.2f} ({rr.hard_fail or 'score'})"


if __name__ == "__main__":
    main()
