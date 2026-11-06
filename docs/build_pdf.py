"""Génère le PDF de la documentation à partir du wiki Markdown (docs/wiki/).

Le wiki est la source ; le PDF n'est qu'une sortie : on ne le modifie jamais à la main.

    pip install -r requirements-docs.txt
    python docs/build_pdf.py              # écrit docs/Assistant-RAG_documentation_cours.pdf
    python docs/build_pdf.py --check      # vérifie renvois de section et images, sans générer
    python docs/build_pdf.py --html out.html   # garde aussi le HTML intermédiaire

Rendu : Markdown -> HTML (bibliothèque markdown) -> PDF par Chrome ou Edge en mode headless.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import markdown

DOCS_DIR = Path(__file__).resolve().parent
WIKI_DIR = DOCS_DIR / "wiki"
PDF_PATH = DOCS_DIR / "Assistant-RAG_documentation_cours.pdf"
TITLE = "Assistant RAG & Évaluation d'IA générative"
SUBTITLE = "Documentation et cours du projet"

PARTS = [
    ("Partie I", "Notions", ["01", "02"]),
    ("Partie II", "Architecture", ["03"]),
    ("Partie III", "Les modules", ["04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]),
    ("Partie IV", "Recul et résultats", ["14", "15", "16", "17", "18", "19", "20"]),
    ("Annexes", "", ["A", "B"]),
]
ALERTS = {"TIP": "tip", "NOTE": "note", "WARNING": "warning", "IMPORTANT": "important"}
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
body { font-family: Cambria, Georgia, serif; font-size: 10.5pt; line-height: 1.45; color: #1f2933; }
h1 { font-family: "Segoe UI", Arial, sans-serif; color: #1d4e89; font-size: 20pt;
     border-bottom: 2px solid #1d4e89; padding-bottom: 4pt; margin-top: 0; }
h2 { font-family: "Segoe UI", Arial, sans-serif; color: #1d4e89; font-size: 14pt; margin-top: 18pt; }
h3 { font-family: "Segoe UI", Arial, sans-serif; font-size: 11.5pt; margin-top: 14pt; }
h2, h3 { break-after: avoid; }
.chapter { break-before: page; }
code { font-family: "DejaVu Sans Mono", Consolas, monospace; font-size: 8.6pt;
       background: #f3f4f6; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f5f6f8; border: 1px solid #e1e4e8; padding: 7pt 9pt; font-size: 8pt;
      line-height: 1.35; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.8pt; }
th { background: #1d4e89; color: white; text-align: left; }
th, td { border: 1px solid #d0d7de; padding: 3pt 5pt; vertical-align: top; }
tr { break-inside: avoid; }
tr:nth-child(even) td { background: #f7f9fb; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; }
p > em:only-child { display: block; text-align: center; font-size: 9pt; color: #52606d; }
.alert { border-left: 4px solid; padding: 6pt 10pt; margin: 10pt 0; break-inside: avoid; font-size: 9.8pt; }
.alert p:first-child { margin-top: 0; } .alert p:last-child { margin-bottom: 0; }
.alert.tip { background: #e8f3ec; border-color: #2f855a; }
.alert.note { background: #f2ecf8; border-color: #6b46c1; }
.alert.warning { background: #fbf3e2; border-color: #b7791f; }
.alert.important { background: #e8f0f8; border-color: #1d4e89; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.cover h1 { border: none; font-size: 28pt; }
.cover .sub { font-size: 15pt; color: #52606d; }
.cover .meta { margin-top: 30pt; font-size: 10pt; color: #52606d; }
.toc { break-before: page; } .toc ul { list-style: none; padding-left: 0; } .toc li { margin: 2pt 0; }
.toc .part { font-weight: bold; margin-top: 10pt; color: #1d4e89; }
.toc a { color: inherit; text-decoration: none; }
.part-page { break-before: page; height: 240mm; display: flex; flex-direction: column;
             justify-content: center; text-align: center; color: #1d4e89; }
.part-page .num { font-size: 14pt; letter-spacing: 2pt; } .part-page .name { font-size: 26pt; }
"""


def pages() -> list[Path]:
    return sorted(p for p in WIKI_DIR.glob("*.md") if re.match(r"(\d\d|[A-Z])-", p.name))


def page_key(path: Path) -> str:
    return path.name.split("-", 1)[0]


def convert_alerts(md: str) -> str:
    """Alertes GitHub (> [!TIP] …) -> <div class="alert tip"> avec le Markdown interne rendu."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        m = re.match(r"^> \[!(\w+)\]\s*$", lines[i])
        if not m or m.group(1) not in ALERTS:
            out.append(lines[i])
            i += 1
            continue
        body = []
        i += 1
        while i < len(lines) and lines[i].startswith(">"):
            body.append(lines[i][2:] if lines[i].startswith("> ") else "")
            i += 1
        inner = markdown.markdown("\n".join(body), extensions=["tables", "fenced_code", "sane_lists"])
        out += ["", f'<div class="alert {ALERTS[m.group(1)]}">{inner}</div>', ""]
    return "\n".join(out)


def render_page(path: Path) -> tuple[str, str]:
    md = path.read_text(encoding="utf-8")
    title = md.splitlines()[0].lstrip("# ").strip()
    md = convert_alerts(md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])

    def absolute(m):
        return f'src="{(WIKI_DIR / m.group(1)).resolve().as_uri()}"'

    body = re.sub(r'src="(assets/[^"]+)"', absolute, body)
    body = re.sub(r'<a href="([^":]+\.md)">', lambda m: f'<a href="#{Path(m.group(1)).stem}">', body)
    return title, f'<section class="chapter" id="{path.stem}">{body}</section>'


def build_html() -> str:
    by_key = {page_key(p): p for p in pages()}
    toc, chapters = [], []
    for num, name, keys in PARTS:
        label = f"{num} : {name}" if name else num
        toc.append(f'<li class="part">{html.escape(label)}</li>')
        heading = (f'<div class="num">{html.escape(num.upper())}</div><div class="name">{html.escape(name)}</div>'
                   if name else f'<div class="name">{html.escape(num)}</div>')
        chapters.append(f'<div class="part-page">{heading}</div>')
        for key in keys:
            if key not in by_key:
                continue
            title, section = render_page(by_key[key])
            toc.append(f'<li><a href="#{by_key[key].stem}">{html.escape(title)}</a></li>')
            chapters.append(section)
    cover = (f'<div class="cover"><h1>{html.escape(TITLE)}</h1><div class="sub">{SUBTITLE}</div>'
             f'<div class="meta">Généré depuis docs/wiki/ le {date.today():%d/%m/%Y}</div></div>')
    toc_html = '<div class="toc"><h1>Sommaire</h1><ul>' + "".join(toc) + "</ul></div>"
    return (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{html.escape(TITLE)}</title>'
            f"<style>{CSS}</style></head><body>{cover}{toc_html}{''.join(chapters)}</body></html>")


def check() -> list[str]:
    """Renvois « section N.M » / « chapitre N » vers des titres existants, images présentes."""
    problems, sections, chapters = [], set(), set()
    texts = {p: p.read_text(encoding="utf-8") for p in pages()}
    for text in texts.values():
        sections.update(re.findall(r"^## (\d+\.\d+) ", text, re.M))
        chapters.update(re.findall(r"^# (\d+)\. ", text, re.M))
    for path, text in texts.items():
        for ref in re.findall(r"sections? (\d+\.\d+)", text):
            if ref not in sections:
                problems.append(f"{path.name} : section {ref} introuvable")
        for ref in re.findall(r"chapitres? (\d+)\b", text):
            if ref not in chapters:
                problems.append(f"{path.name} : chapitre {ref} introuvable")
        for img in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if not (WIKI_DIR / img).exists():
                problems.append(f"{path.name} : image {img} absente")
    return problems


def find_browser() -> str:
    for candidate in [os.getenv("CHROME_PATH", ""), *BROWSERS]:
        if candidate and Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if found := shutil.which(name):
            return found
    sys.exit("Chrome ou Edge introuvable : définir CHROME_PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="vérifier sans générer")
    parser.add_argument("--html", type=Path, help="enregistrer aussi le HTML intermédiaire")
    parser.add_argument("--out", type=Path, default=PDF_PATH)
    args = parser.parse_args()

    problems = check()
    for p in problems:
        print("⚠", p)
    if args.check:
        sys.exit(1 if problems else 0)

    page = build_html()
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "doc.html"
        html_path.write_text(page, encoding="utf-8")
        if args.html:
            shutil.copy(html_path, args.html)
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--print-to-pdf={args.out.resolve()}", html_path.as_uri()],
                       check=True, capture_output=True, timeout=180)
    print(f"PDF écrit : {args.out} ({args.out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
