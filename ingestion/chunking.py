"""
Étape 3 du pipeline : ingestion des documents.

Un document PDF en entrée -> chunks de texte en sortie.

Chunking simple par taille fixe (~500 tokens) avec chevauchement (~50 tokens) : un découpage
simple qui fonctionne vaut mieux qu'un découpage complexe (sémantique ou par structure)
introduit prématurément. C'est le paramètre à faire varier ensuite, mesures à l'appui
(voir README, section "Itération : effet de la taille de chunk").
"""
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_DIR, CHUNKS_PATH, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


@dataclass
class Chunk:
    id: str
    source: str
    chunk_index: int
    text: str


def load_pdf_text(pdf_path: Path) -> str:
    """Extrait le texte d'un PDF. Suppose une couche de texte extractible (pas un scan)."""
    reader = PdfReader(str(pdf_path))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


def load_corpus_documents(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Charge tous les PDF du corpus. Retourne [{source, text}, ...]."""
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans {corpus_dir}. "
            "Lance d'abord la préparation du corpus (voir README, étape 1)."
        )
    documents = []
    for pdf_path in pdf_files:
        text = load_pdf_text(pdf_path)
        if len(text.strip()) < 50:
            print(f"  !! {pdf_path.name}: texte extrait trop court, PDF probablement scanné (image) — ignoré")
            continue
        documents.append({"source": pdf_path.name, "text": text})
    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Découpe chaque document en chunks de taille fixe (en tokens) avec chevauchement."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
    )
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(
                id=f"{doc['source']}::chunk_{i}",
                source=doc["source"],
                chunk_index=i,
                text=piece.strip(),
            ))
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path = CHUNKS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def main():
    print(f"Chargement du corpus depuis {CORPUS_DIR} ...")
    documents = load_corpus_documents()
    print(f"  {len(documents)} document(s) chargé(s)")

    print(f"Chunking (taille={CHUNK_SIZE_TOKENS} tokens, overlap={CHUNK_OVERLAP_TOKENS} tokens) ...")
    chunks = chunk_documents(documents)
    print(f"  {len(chunks)} chunks générés")

    save_chunks(chunks)
    print(f"Chunks sauvegardés dans {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
