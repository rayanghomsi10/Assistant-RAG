"""
Étape 1 du pipeline : préparation du corpus.

Télécharge les pages de documentation officielle dbt + Airflow, en extrait le texte principal
(sans navigation, pieds de page ni artefacts de mise en page), puis génère un PDF par document
dans corpus/.

Les PDF générés sont versionnés dans le dépôt : ce script sert à reconstruire ou mettre à jour
le corpus, pas à le télécharger à chaque exécution du pipeline.

Usage :
    python ingestion/build_corpus.py
"""
import re
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_DIR

RAW_TEXT_DIR = CORPUS_DIR / "raw_text"

# Un seul domaine cohérent (outillage data engineering), comme recommandé pour une V1.
PAGES = {
    "dbt_01_sql_models": "https://docs.getdbt.com/docs/build/sql-models",
    "dbt_02_jinja_macros": "https://docs.getdbt.com/docs/build/jinja-macros",
    "dbt_03_tests": "https://docs.getdbt.com/docs/build/data-tests",
    "dbt_04_incremental_models": "https://docs.getdbt.com/docs/build/incremental-models",
    "dbt_05_materializations": "https://docs.getdbt.com/docs/build/materializations",
    "airflow_01_dags": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html",
    "airflow_02_operators": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html",
    "airflow_03_scheduler": "https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/scheduler.html",
    "airflow_04_xcoms": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html",
    "airflow_05_best_practices": "https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) rag-genai-eval-corpus-builder/1.0"

# En-têtes/pieds de page récurrents des sites de doc, sans valeur pour le RAG.
JUNK_LINE_PATTERNS = [
    r"^View as Markdown$", r"^View this page as plain Markdown$",
    r"^Open in( (ChatGPT|Claude|Perplexity))?$", r"^(ChatGPT|Claude|Perplexity)$",
    r"^Ask questions about this page$", r"^Was this page helpful\??$",
    r"^Edit this page", r"^Copy$", r"^Copy page$", r"^Yes$", r"^No$",
]
_junk_re = re.compile("|".join(JUNK_LINE_PATTERNS), re.I)

_styles = getSampleStyleSheet()
_title_style = ParagraphStyle("DocTitle", parent=_styles["Heading1"], spaceAfter=14)
_body_style = ParagraphStyle(
    "Body", parent=_styles["Normal"], fontSize=10, leading=14, alignment=TA_LEFT, spaceAfter=6
)
_source_style = ParagraphStyle(
    "Source", parent=_styles["Normal"], fontSize=8, textColor="#666666", spaceAfter=16
)


def fetch(url: str) -> str:
    result = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "20", "-A", USER_AGENT, url],
        capture_output=True, text=True, encoding="utf-8", errors="ignore",
    )
    return result.stdout


def clean_html(html: str) -> str:
    """Isole le contenu principal et le convertit en texte ligne à ligne."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "button"]):
        tag.decompose()
    for tag in soup.find_all(
        attrs={"class": re.compile(r"(sidebar|navbar|toc|breadcrumb|pagination|feedback|cookie)", re.I)}
    ):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or soup

    lines = [ln.strip() for ln in main.get_text(separator="\n").splitlines()]
    lines = [ln for ln in lines if ln and not _junk_re.match(ln)]

    # Déduplique les lignes consécutives identiques (artefacts de mise en page).
    deduped = []
    for ln in lines:
        if not deduped or deduped[-1] != ln:
            deduped.append(ln)
    return "\n".join(deduped)


def text_to_pdf(name: str, source_url: str, text: str, pdf_path: Path) -> None:
    title = name.replace("_", " ").title()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=title,
    )
    story = [
        Paragraph(escape(title), _title_style),
        Paragraph(f"Source originale : {escape(source_url)}", _source_style),
    ]
    story.extend(Paragraph(escape(ln), _body_style) for ln in text.splitlines() if ln.strip())
    doc.build(story)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in PAGES.items():
        print(f"{name} <- {url}")
        html = fetch(url)
        if len(html) < 500:
            print(f"  !! réponse trop courte ({len(html)} caractères) — page ignorée, vérifier l'URL")
            continue

        text = clean_html(html)
        (RAW_TEXT_DIR / f"{name}.txt").write_text(f"Source: {url}\n\n{text}", encoding="utf-8")

        pdf_path = CORPUS_DIR / f"{name}.pdf"
        text_to_pdf(name, url, text, pdf_path)
        print(f"  -> {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB, {len(text)} caractères)")

    print(f"\nCorpus reconstruit dans {CORPUS_DIR}. Relancer ensuite :")
    print("  python ingestion/chunking.py && python retrieval/vector_store.py")


if __name__ == "__main__":
    main()
