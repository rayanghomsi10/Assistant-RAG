"""
Étape DPO 1/4 : questions d'entraînement, disjointes du jeu d'évaluation.

- Questions du périmètre : générées par un LLM (Ollama, par défaut le modèle juge, d'une autre
  famille que le générateur) à partir de chaque chunk de chunks.json. On garde l'identifiant du
  chunk source : il permet ensuite de savoir si un refus est une erreur (le passage qui contient
  la réponse était dans le contexte) ou non.
- Questions hors périmètre : training/out_of_scope_train.json (rédigées à la main, distinctes de
  celles de l'évaluation).
- Anti-fuite : toute question trop proche d'une question d'évaluation (recouvrement lexical, et
  similarité d'embeddings si disponible) est écartée. Sans ce filtre, l'évaluation après DPO
  mesurerait en partie la mémorisation des questions d'entraînement.

Usage (machine locale avec Ollama) :
    python training/generate_questions.py --per-chunk 2
Sortie : training/data/questions.jsonl
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JUDGE_LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_NUM_CTX, TRAINING_DATA_DIR, TRAINING_DIR
from evaluation.dataset import load_eval_dataset
from ingestion.chunking import load_chunks
from rag_chain.checks import normalize

OUT_OF_SCOPE_PATH = TRAINING_DIR / "out_of_scope_train.json"
QUESTIONS_PATH = TRAINING_DATA_DIR / "questions.jsonl"

PROMPT = """Voici un extrait de la documentation officielle de {product}.

<extrait>
{text}
</extrait>

Rédige {n} questions en français, variées, auxquelles cet extrait répond explicitement.
Contraintes :
- chaque question doit être compréhensible seule et mentionner {product} ;
- la réponse doit se trouver dans l'extrait, pas dans des connaissances générales ;
- ne recopie pas les phrases de l'extrait et ne fais pas référence à « l'extrait » ;
- pas de questions oui/non triviales.
Réponds uniquement en JSON : {{"questions": ["...", "..."]}}"""

_STOP = set("le la les un une des de du d l à au aux et ou en dans sur pour par avec est sont qu que qui quoi "
            "quel quelle quels quelles comment pourquoi quand ce cette ces il elle on se sa son ses peut faut "
            "t a y ne pas plus".split())


def content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", normalize(text)) if len(w) > 2 and w not in _STOP}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def product_of(source: str) -> str:
    return "Apache Airflow" if source.startswith("airflow") else "dbt"


def generate_for_chunk(llm, chunk: dict, n: int) -> list[str]:
    msg = PROMPT.format(product=product_of(chunk["source"]), text=chunk["text"], n=n)
    from ollama import ResponseError

    try:
        raw = llm.invoke(msg).content
        questions = json.loads(raw).get("questions", [])
    except ResponseError as exc:
        # En mode JSON, le modèle boucle parfois sur un même token : Ollama interrompt la
        # génération. On passe ce chunk plutôt que de perdre toutes les questions déjà générées.
        print(f"  !! {chunk['id']} ignoré : {exc}")
        return []
    except (json.JSONDecodeError, AttributeError):
        return []
    return [q.strip() for q in questions if isinstance(q, str) and q.strip().endswith("?")][:n]


class LeakageFilter:
    """Écarte les questions trop proches des questions d'évaluation."""

    def __init__(self, eval_questions: list[str], jaccard_max: float = 0.5,
                 cosine_max: float | None = None, use_embeddings: bool = True):
        self.eval_tokens = [content_tokens(q) for q in eval_questions]
        self.jaccard_max, self.cosine_max = jaccard_max, cosine_max
        self.eval_vecs = None
        if use_embeddings:
            import numpy as np

            from ingestion.embeddings import embed_queries
            self._embed, self._np = embed_queries, np
            self.eval_vecs = np.asarray(embed_queries(eval_questions))
            if self.cosine_max is None:
                # L'échelle des similarités dépend du modèle d'embeddings (avec multilingual-e5,
                # deux questions sans rapport dépassent déjà 0,8) : un seuil fixe écarte presque
                # tout. On le calibre sur le jeu d'évaluation lui-même, dont les questions sont
                # distinctes : n'est une fuite que ce qui est plus proche d'une question
                # d'évaluation que ne le sont entre elles les deux plus proches.
                sims = self.eval_vecs @ self.eval_vecs.T
                np.fill_diagonal(sims, -1.0)
                self.cosine_max = float(sims.max())

    def keep(self, questions: list[str]) -> list[bool]:
        keep = [max(jaccard(content_tokens(q), t) for t in self.eval_tokens) < self.jaccard_max for q in questions]
        if self.eval_vecs is not None and questions:
            sims = self._np.asarray(self._embed(questions)) @ self.eval_vecs.T
            keep = [k and s.max() < self.cosine_max for k, s in zip(keep, sims, strict=True)]
        return keep


def dedupe(rows: list[dict], threshold: float = 0.8) -> list[dict]:
    kept, seen = [], []
    for r in rows:
        toks = content_tokens(r["question"])
        if any(jaccard(toks, s) >= threshold for s in seen):
            continue
        kept.append(r)
        seen.append(toks)
    return kept


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--per-chunk", type=int, default=2)
    p.add_argument("--model", default=JUDGE_LLM_MODEL, help="Modèle Ollama qui rédige les questions")
    p.add_argument("--min-tokens", type=int, default=60, help="Ignorer les chunks plus courts")
    p.add_argument("--max-chunks", type=int, default=None, help="Limiter (test rapide)")
    p.add_argument("--no-embed-filter", action="store_true", help="Filtre anti-fuite lexical seulement")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from langchain_ollama import ChatOllama
    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=args.model, temperature=0.7,
                     num_ctx=OLLAMA_NUM_CTX, format="json", seed=args.seed)

    _, chunks = load_chunks()
    chunks = [c for c in chunks if c.get("n_tokens", 999) >= args.min_tokens]
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    rows = []
    for i, c in enumerate(chunks, 1):
        qs = generate_for_chunk(llm, c, args.per_chunk)
        print(f"[{i}/{len(chunks)}] {c['id']} -> {len(qs)} question(s)")
        rows += [{"question": q, "type": "in_scope", "source_chunk_id": c["id"], "source": c["source"]} for q in qs]

    oos = json.loads(OUT_OF_SCOPE_PATH.read_text(encoding="utf-8"))
    rows += [{"question": q, "type": "out_of_scope", "source_chunk_id": None, "source": None} for q in oos]

    n_generated = len(rows)
    rows = dedupe(rows)
    eval_questions = [it["question"] for it in load_eval_dataset()]
    flt = LeakageFilter(eval_questions, use_embeddings=not args.no_embed_filter)
    keep = flt.keep([r["question"] for r in rows])
    leaked = [r["question"] for r, k in zip(rows, keep, strict=True) if not k]
    rows = [r for r, k in zip(rows, keep, strict=True) if k]

    random.Random(args.seed).shuffle(rows)
    for n, r in enumerate(rows):
        r["id"] = f"train_{n:04d}"
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_in = sum(r["type"] == "in_scope" for r in rows)
    if flt.eval_vecs is not None:
        print(f"\nSeuil anti-fuite (cosinus, calibré sur le jeu d'évaluation) : {flt.cosine_max:.3f}")
    print(f"\n{n_generated} questions générées, {len(rows)} conservées "
          f"({n_in} dans le périmètre, {len(rows) - n_in} hors périmètre) ; "
          f"{len(leaked)} écartées car trop proches de l'évaluation.")
    for q in leaked[:10]:
        print(f"  écartée : {q}")
    print(f"-> {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
