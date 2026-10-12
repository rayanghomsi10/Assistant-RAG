"""
Étape 2 du pipeline : ingestion des documents -> chunks.

Chunking **par sections** : le corpus Markdown (voir build_corpus.py) conserve les titres de la
documentation. On découpe d'abord aux frontières de sections, on regroupe les petites sections
voisines d'une même section de niveau 2, et on ne recoupe en taille fixe (avec chevauchement)
que les sections trop longues. Chaque chunk est préfixé par son fil d'Ariane
(« Document > Section > Sous-section ») : le chunk reste compréhensible hors contexte, et le
titre aide le retrieval.

Pourquoi : en V1, la coupe à taille fixe tombait au milieu des explications, et l'itération
300/30 tokens a dégradé la faithfulness — symptôme de passages tronqués.

La taille est mesurée avec le **tokenizer du modèle d'embeddings** : c'est sa longueur maximale
(512 tokens pour multilingual-e5) qui fixe ce qu'il peut réellement encoder. Au-delà, le texte
est tronqué silencieusement à l'indexation.
"""
import hashlib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, CHUNKS_PATH, CORPUS_DIR, EMBEDDING_MODEL_NAME

TokenCounter = Callable[[str], int]

# Version de l'algorithme de découpage, enregistrée avec l'index et dans MLflow : deux runs
# avec des versions différentes ne sont pas comparables à paramètres égaux.
CHUNKER_VERSION = "sections-v2"


@dataclass
class Chunk:
    id: str
    source: str
    chunk_index: int
    section: str
    url: str
    text: str
    n_tokens: int


@dataclass
class Section:
    path: list[str]
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


@lru_cache(maxsize=1)
def get_token_counter() -> TokenCounter:
    """Compteur de tokens du modèle d'embeddings (tokenizer seul, sans charger le modèle)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------------------------

def load_pdf_text(pdf_path: Path) -> str:
    """Extrait le texte d'un PDF (couche de texte requise, pas de scan). Conservé pour pouvoir
    ajouter des documents PDF au corpus ; ils n'ont pas de structure de sections exploitable."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_corpus_documents(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Charge les documents du corpus (.md en priorité, .pdf accepté). Retourne
    [{source, text}, ...] trié par nom de fichier."""
    files = sorted(list(corpus_dir.glob("*.md")) + list(corpus_dir.glob("*.pdf")))
    if not files:
        raise FileNotFoundError(
            f"Aucun document trouvé dans {corpus_dir}. Lance d'abord `python ingestion/build_corpus.py`."
        )
    documents = []
    for path in files:
        text = path.read_text(encoding="utf-8") if path.suffix == ".md" else load_pdf_text(path)
        if len(text.strip()) < 50:
            print(f"  !! {path.name}: texte extrait trop court (PDF scanné ?) — ignoré")
            continue
        documents.append({"source": path.name, "text": text})
    return documents


# ---------------------------------------------------------------------------------------------
# Découpage
# ---------------------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SOURCE_LINE = re.compile(r"^Source\s*:\s*(\S+)")


def parse_sections(markdown: str) -> tuple[str, list[Section]]:
    """Découpe un document Markdown en sections (en ignorant les # des blocs de code).
    Retourne (url_source, sections). Le chemin d'une section contient le titre du document."""
    url = ""
    sections: list[Section] = []
    path: list[str] = []
    current = Section(path=[])
    in_code = False
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            if current.body:
                sections.append(current)
            level = len(m.group(1))
            path = path[: level - 1] + [""] * max(0, level - 1 - len(path)) + [m.group(2).strip()]
            current = Section(path=[p for p in path if p])
            continue
        if not in_code and not url and (sm := _SOURCE_LINE.match(stripped)):
            url = sm.group(1)
            continue
        current.lines.append(line)
    if current.body:
        sections.append(current)
    return url, sections


def _breadcrumb(path: list[str], fallback: str) -> str:
    return " > ".join(path) if path else fallback


def chunk_document(
    source: str,
    text: str,
    count_tokens: TokenCounter,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    url, sections = parse_sections(text)
    title = Path(source).stem

    # 1) Regroupement des petites sections consécutives d'une même section de niveau 2.
    groups: list[tuple[list[str], str]] = []   # (chemin de la 1re section, texte)
    for sec in sections:
        if groups:
            g_path, g_text = groups[-1]
            same_parent = g_path[:2] == sec.path[:2]
            # Le titre d'une sous-section regroupée reste dans le texte, en clair.
            extra = (f"{sec.path[-1]}\n" if sec.path and sec.path != g_path else "") + sec.body
            candidate = f"{g_text}\n\n{extra}"
            if same_parent and count_tokens(f"{_breadcrumb(g_path, title)}\n\n{candidate}") <= chunk_size:
                groups[-1] = (g_path, candidate)
                continue
        groups.append((sec.path, sec.body))

    # 2) Découpe à taille fixe des groupes trop longs, en préfixant chaque morceau du fil d'Ariane.
    chunks: list[Chunk] = []
    for g_path, g_text in groups:
        crumb = _breadcrumb(g_path, title)
        header = f"{crumb}\n\n"
        budget = max(64, chunk_size - count_tokens(header))
        if count_tokens(g_text) <= budget:
            pieces = [g_text]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=budget,
                chunk_overlap=min(chunk_overlap, budget // 4),
                length_function=count_tokens,
                separators=["\n\n", "\n```", "\n", ". ", " ", ""],
            )
            pieces = splitter.split_text(g_text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunk_text = header + piece
            i = len(chunks)
            chunks.append(Chunk(
                id=f"{source}::chunk_{i:03d}",
                source=source,
                chunk_index=i,
                section=crumb,
                url=url,
                text=chunk_text,
                n_tokens=count_tokens(chunk_text),
            ))
    return chunks


def chunk_documents(
    documents: list[dict],
    count_tokens: TokenCounter | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    count_tokens = count_tokens or get_token_counter()
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc["source"], doc["text"], count_tokens, chunk_size, chunk_overlap))
    return chunks


def corpus_fingerprint(documents: list[dict]) -> str:
    """Empreinte du corpus (12 premiers caractères du SHA-256 des documents)."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d["source"]):
        h.update(doc["source"].encode() + b"\0" + doc["text"].encode() + b"\0")
    return h.hexdigest()[:12]


def chunking_params(documents: list[dict] | None = None) -> dict:
    """Paramètres qui déterminent le contenu de chunks.json (enregistrés avec l'index)."""
    return {
        "corpus_sha": corpus_fingerprint(documents) if documents else "",
        "chunker_version": CHUNKER_VERSION,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "chunk_tokenizer": EMBEDDING_MODEL_NAME,
    }


def save_chunks(chunks: list[Chunk], documents: list[dict], out_path: Path = CHUNKS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": chunking_params(documents), "chunks": [asdict(c) for c in chunks]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_chunks(path: Path = CHUNKS_PATH) -> tuple[dict, list[dict]]:
    """Retourne (paramètres de chunking, chunks). Accepte l'ancien format (liste nue)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {}, data
    return data.get("params", {}), data["chunks"]


def main():
    print(f"Chargement du corpus depuis {CORPUS_DIR} ...")
    documents = load_corpus_documents()
    print(f"  {len(documents)} document(s) chargé(s)")

    print(f"Chunking par sections (taille max={CHUNK_SIZE_TOKENS}, overlap={CHUNK_OVERLAP_TOKENS} "
          f"tokens du tokenizer {EMBEDDING_MODEL_NAME}) ...")
    chunks = chunk_documents(documents)
    sizes = sorted(c.n_tokens for c in chunks)
    print(f"  {len(chunks)} chunks — tokens : médiane {sizes[len(sizes) // 2]}, max {sizes[-1]}")

    save_chunks(chunks, documents)
    print(f"Chunks sauvegardés dans {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
