"""
Jeu d'évaluation : chargement et validation contre le corpus.

Leçon de la V1 : une réponse de référence rédigée de mémoire plutôt que vérifiée contre le
corpus a faussé une ligne de l'évaluation. Ici, chaque question du périmètre porte des preuves
(`evidence`) recopiées mot pour mot du corpus, et `validate_dataset` vérifie mécaniquement que
chaque preuve figure dans un des documents attendus. Le test tests/test_dataset.py échoue si une
preuve ne se retrouve plus dans le corpus (par exemple après une reconstruction du corpus).

Les preuves servent aussi à mesurer le retrieval sans LLM : un chunk récupéré « contient la
preuve » s'il en contient le texte (voir metrics.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_DIR
from rag_chain.checks import _strip_markdown, normalize

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def norm(text: str) -> str:
    return normalize(_strip_markdown(text))


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"] if isinstance(data, dict) else data


def load_corpus_texts(corpus_dir: Path = CORPUS_DIR) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def evidence_in(evidence: str, text: str) -> bool:
    return norm(evidence) in norm(text)


def validate_dataset(items: list[dict], corpus: dict[str, str]) -> list[str]:
    """Retourne la liste des problèmes (vide si le jeu est valide)."""
    errors = []
    seen_ids, seen_questions = set(), set()
    normalized_corpus = {name: norm(text) for name, text in corpus.items()}
    for it in items:
        iid = it.get("id", "?")
        if iid in seen_ids:
            errors.append(f"{iid}: identifiant en double")
        seen_ids.add(iid)
        q = norm(it["question"])
        if q in seen_questions:
            errors.append(f"{iid}: question en double")
        seen_questions.add(q)

        if it["type"] == "out_of_scope":
            if it.get("evidence") or it.get("expected_sources"):
                errors.append(f"{iid}: une question hors périmètre ne doit avoir ni preuve ni source")
            continue
        if it["type"] != "in_scope":
            errors.append(f"{iid}: type inconnu {it['type']!r}")
            continue
        if not it.get("evidence"):
            errors.append(f"{iid}: aucune preuve")
        for src in it["expected_sources"]:
            if src not in corpus:
                errors.append(f"{iid}: source attendue absente du corpus : {src}")
        for ev in it.get("evidence", []):
            if not any(norm(ev) in normalized_corpus.get(src, "") for src in it["expected_sources"]):
                errors.append(f"{iid}: preuve introuvable dans {it['expected_sources']} : {ev!r}")
    return errors


def dataset_fingerprint(path: Path = EVAL_DATASET_PATH) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    items = load_eval_dataset()
    problems = validate_dataset(items, load_corpus_texts())
    n_in = sum(it["type"] == "in_scope" for it in items)
    print(f"{len(items)} questions ({n_in} dans le périmètre, {len(items) - n_in} hors périmètre)")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("Toutes les preuves sont présentes dans le corpus.")
