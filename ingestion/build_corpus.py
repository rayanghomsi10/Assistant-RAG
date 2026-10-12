"""
Étape 1 du pipeline : préparation du corpus.

Reconstruit le corpus à partir des **sources** de la documentation officielle (Markdown/MDX du
dépôt dbt-labs/docs.getdbt.com, reStructuredText du dépôt apache/airflow), à des versions figées.

Pourquoi les sources plutôt que le HTML des sites : la V1 récupérait le HTML rendu puis le
transformait en PDF. Chaque élément inline (lien, `code`) devenait une ligne à part, la barre de
recherche du site (« ↑↓ Navigate ■ Select Esc Close ») se retrouvait dans le texte, et la structure
en sections était perdue. Les sources donnent un texte propre, versionné et reproductible au
caractère près, avec ses titres de sections — ce qui permet un chunking par sections
(voir ingestion/chunking.py).

Le corpus produit est un fichier Markdown par document dans corpus/ (versionnés dans le dépôt).

Usage :
    python ingestion/build_corpus.py
"""
import html
import re
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CORPUS_DIR

# --- Versions figées des sources (reproductibilité) ---
DBT_DOCS_REF = "3aea32926da44964da97b6b0e42ef29564068e45"   # branche `current` au 2026-10-12
AIRFLOW_REF = "3.3.1"                                        # tag de release Airflow

DBT_RAW = f"https://raw.githubusercontent.com/dbt-labs/docs.getdbt.com/{DBT_DOCS_REF}/website"
AIRFLOW_RAW = f"https://raw.githubusercontent.com/apache/airflow/{AIRFLOW_REF}"

# nom du document -> (URL publique citée dans le corpus, chemin de la source dans le dépôt)
PAGES: dict[str, tuple[str, str]] = {
    "dbt_01_sql_models": ("https://docs.getdbt.com/docs/build/sql-models",
                          f"{DBT_RAW}/docs/docs/build/sql-models.md"),
    "dbt_02_jinja_macros": ("https://docs.getdbt.com/docs/build/jinja-macros",
                            f"{DBT_RAW}/docs/docs/build/jinja-macros.md"),
    "dbt_03_tests": ("https://docs.getdbt.com/docs/build/data-tests",
                     f"{DBT_RAW}/docs/docs/build/data-tests.md"),
    "dbt_04_incremental_models": ("https://docs.getdbt.com/docs/build/incremental-models",
                                  f"{DBT_RAW}/docs/docs/build/incremental-models.md"),
    "dbt_05_materializations": ("https://docs.getdbt.com/docs/build/materializations",
                                f"{DBT_RAW}/docs/docs/build/materializations.md"),
    "airflow_01_dags": ("https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html",
                        f"{AIRFLOW_RAW}/airflow-core/docs/core-concepts/dags.rst"),
    "airflow_02_operators": ("https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html",
                             f"{AIRFLOW_RAW}/airflow-core/docs/core-concepts/operators.rst"),
    "airflow_03_scheduler": ("https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/scheduler.html",
                             f"{AIRFLOW_RAW}/airflow-core/docs/administration-and-deployment/scheduler.rst"),
    "airflow_04_xcoms": ("https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html",
                         f"{AIRFLOW_RAW}/airflow-core/docs/core-concepts/xcoms.rst"),
    "airflow_05_best_practices": ("https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html",
                                  f"{AIRFLOW_RAW}/airflow-core/docs/best-practices.rst"),
}

# Sections de pure navigation (listes de liens), sans contenu utile pour le RAG.
DROPPED_SECTIONS = {"related reference docs", "related docs", "related documentation"}

Fetcher = Callable[[str], str]


def http_fetch(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------------------------
# Utilitaires communs
# ---------------------------------------------------------------------------------------------

def _split_fences(text: str) -> list[tuple[bool, str]]:
    """Découpe un texte Markdown en segments (est_du_code, contenu) selon les blocs ``` / ~~~."""
    segments: list[tuple[bool, str]] = []
    buf: list[str] = []
    in_code = False
    fence = ""
    for line in text.split("\n"):
        stripped = line.strip()
        if not in_code and (stripped.startswith("```") or stripped.startswith("~~~")):
            if buf:
                segments.append((False, "\n".join(buf)))
            buf, in_code, fence = [line], True, stripped[:3]
        elif in_code and stripped.startswith(fence) and stripped.strip("`~") == "":
            buf.append(line)
            segments.append((True, "\n".join(buf)))
            buf, in_code = [], False
        else:
            buf.append(line)
    if buf:
        segments.append((in_code, "\n".join(buf)))
    return segments


def _drop_sections(markdown: str, titles: set[str]) -> str:
    """Supprime les sections de navigation dont le titre (normalisé) est dans `titles` : le
    titre et la liste de liens qui suit. Le premier paragraphe qui n'est pas un élément de liste
    met fin à la suppression (sur docs.getdbt.com, un encadré important suit souvent la liste
    sans nouveau titre)."""
    out, dropping = [], False
    for line in markdown.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m and m.group(2).strip().lower() in titles:
            dropping = True
            continue
        if dropping:
            if not line.strip() or re.match(r"^\s*([-*+]|\d+\.)\s", line):
                continue
            dropping = False
        out.append(line)
    return "\n".join(out)


def _tidy(text: str) -> str:
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# ---------------------------------------------------------------------------------------------
# MDX (dbt)
# ---------------------------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    meta: dict[str, str] = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().split("\n"):
                if ":" in line:
                    key, _, value = line.partition(":")
                    meta[key.strip()] = value.strip().strip("'\"")
            text = text[end + 4:]
    return meta, text


def _scan_jsx_tags(text: str):
    """Itère sur les balises JSX/HTML (y compris sur plusieurs lignes, avec props {…}).
    Renvoie (début, fin, nom, est_fermante, est_autofermante, props_brutes)."""
    i = 0
    while True:
        m = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)").search(text, i)
        if not m:
            return
        j, depth, quote = m.end(), 0, ""
        while j < len(text):
            c = text[j]
            if quote:
                if c == quote:
                    quote = ""
            elif c in "'\"":
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == ">" and depth <= 0:
                break
            j += 1
        if j >= len(text):
            return
        raw = text[m.end():j]
        yield m.start(), j + 1, m.group(2), m.group(1) == "/", raw.rstrip().endswith("/"), raw
        i = j + 1


def _prop(raw: str, name: str) -> str | None:
    m = re.search(rf"""\b{name}\s*=\s*(["'])(.*?)\1""", raw, re.S)
    return m.group(2) if m else None


def _term_label(term_id: str) -> str:
    label = term_id.replace("-", " ")
    return label.upper() if len(label) <= 3 else label


class MdxConverter:
    """Convertit une page MDX de docs.getdbt.com en Markdown simple."""

    def __init__(self, fetch: Fetcher, raw_base: str = DBT_RAW):
        self.fetch = fetch
        self.raw_base = raw_base
        self._constants: dict[str, str] | None = None

    @property
    def constants(self) -> dict[str, str]:
        if self._constants is None:
            src = self.fetch(f"{self.raw_base}/constants.js")
            self._constants = dict(re.findall(r"^\s*['\"]?(\w+)['\"]?\s*:\s*['\"]([^'\"]+)['\"]", src, re.M))
        return self._constants

    def convert(self, text: str, heading_shift: int = 0) -> tuple[str, str]:
        meta, body = parse_frontmatter(text)
        title = meta.get("title", "")

        # Imports de fragments réutilisables (/snippets/…) : remplacés par leur contenu.
        snippets: dict[str, str] = {}
        segments = _split_fences(body)
        rebuilt = []
        for is_code, seg in segments:
            if not is_code:
                def _imp(m):
                    snippets[m.group(1)] = m.group(2)
                    return ""
                seg = re.sub(r"^import\s+(\w+)\s+from\s+['\"](/snippets/[^'\"]+)['\"];?\s*$", _imp, seg, flags=re.M)
            rebuilt.append(seg)
        body = "\n".join(rebuilt)

        parts = []
        for is_code, seg in _split_fences(body):
            parts.append(self._clean_code_fence(seg) if is_code else self._convert_prose(seg, snippets))
        out = "\n".join(parts)
        if heading_shift:
            out = re.sub(r"^(#{1,6})(\s)", lambda m: "#" * min(6, len(m.group(1)) + heading_shift) + m.group(2),
                         out, flags=re.M)
        return title, out

    @staticmethod
    def _clean_code_fence(seg: str) -> str:
        lines = seg.split("\n")
        m = re.match(r"^(\s*)(```|~~~)(\w*)", lines[0])
        if m:
            lines[0] = f"```{m.group(3)}"
        lines[-1] = "```" if lines[-1].strip().startswith(("```", "~~~")) else lines[-1]
        return textwrap.dedent("\n".join(lines))

    def _convert_prose(self, seg: str, snippets: dict[str, str]) -> str:
        out, last = [], 0
        tab_labels: dict[str, str] = {}
        for start, end, name, closing, selfclosing, raw in list(_scan_jsx_tags(seg)):
            out.append(seg[last:start])
            last = end
            if closing:
                continue
            if name == "Constant":
                out.append(self.constants.get(_prop(raw, "name") or "", _prop(raw, "name") or ""))
            elif name == "Term":
                if selfclosing:
                    out.append(_term_label(_prop(raw, "id") or ""))
            elif name == "File":
                out.append(f"File `{_prop(raw, 'name')}`:")
            elif name == "Tabs":
                tab_labels = dict((v, l) for l, v in re.findall(r"label:\s*['\"](.+?)['\"],\s*value:\s*['\"](.+?)['\"]", raw))
            elif name == "TabItem":
                value = _prop(raw, "value") or ""
                out.append(f"**{_prop(raw, 'label') or tab_labels.get(value, value)}**")
            elif name == "FAQ":
                out.append(self._faq(_prop(raw, "path") or ""))
            elif name in snippets:
                snippet = self.fetch(f"{self.raw_base}{snippets[name]}")
                props = dict((k, v) for k, _, v in re.findall(r"""(\w+)\s*=\s*(["'])(.*?)\2""", raw))
                snippet = re.sub(r"\{props\.(\w+)\}", lambda pm: props.get(pm.group(1), pm.group(1)), snippet)
                _, snippet_md = self.convert(snippet)
                out.append("\n" + snippet_md + "\n")
            # Autres balises (Lightbox, CourseCallout, VersionBlock, details…) : on garde le
            # contenu éventuel entre balises, on supprime la balise elle-même.
        out.append(seg[last:])
        text = "".join(out)

        text = re.sub(r"^[ \t]*:::[ \t]*(\w+)[ \t]*(.*)$",
                      lambda m: f"{m.group(1).capitalize()}: {m.group(2).strip()}".rstrip(),
                      text, flags=re.M)
        text = re.sub(r"^[ \t]*:::[ \t]*$", "", text, flags=re.M)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)                       # images
        text = re.sub(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)", r"\1", text)    # liens -> texte
        text = html.unescape(text)
        # Indentation résiduelle des balises JSX supprimées (sinon interprétée comme du code).
        text = "\n".join(ln.strip() if not re.match(r"^\s*([-*+]|\d+\.)\s", ln) else ln.rstrip()
                         for ln in text.split("\n"))
        return text

    def _faq(self, path: str) -> str:
        meta, body = parse_frontmatter(self.fetch(f"{self.raw_base}/docs/faqs/{path}.md"))
        _, md = self.convert("---\n---\n" + body, heading_shift=4)
        return f"\n#### FAQ: {meta.get('title', path)}\n\n{md.strip()}\n"


# ---------------------------------------------------------------------------------------------
# reStructuredText (Airflow)
# ---------------------------------------------------------------------------------------------

_ADMONITIONS = {"note": "Note", "tip": "Tip", "warning": "Warning", "important": "Important",
                "seealso": "See also", "caution": "Caution", "attention": "Attention", "hint": "Hint"}
_VERSION_DIRECTIVES = {"versionadded": "New in version", "versionchanged": "Changed in version",
                       "deprecated": "Deprecated since version"}
_DROPPED_DIRECTIVES = {"image", "figure", "toctree", "contents", "spelling:word-list", "raw", "only",
                       "autoclass", "automodule", "redirect-from", "jinja", "airflow-providers-list"}
_UNDERLINE = re.compile(r"""^([=\-~^'"*+#`:.])\1{2,}\s*$""")


def rst_inline(text: str) -> str:
    text = re.sub(r":[\w:.-]+:`([^`<]*?)\s*<[^`>]*>`", r"\1", text)            # :role:`texte <cible>`
    text = re.sub(r":[\w:.-]+:`~?(?:[\w.]+\.)?([^`.]+?)(\(\))?`",
                  lambda m: f"`{m.group(1)}{m.group(2) or ''}`", text)          # :class:`~a.b.C` -> `C`
    text = re.sub(r":[\w:.-]+:`([^`]+)`", r"`\1`", text)
    text = re.sub(r"`([^`<]+?)\s*<[^`>]+>`__?", r"\1", text)                    # liens externes
    text = re.sub(r"``([^`]+)``", r"`\1`", text)                                # littéral
    text = re.sub(r"(?<![`\w])`([^`]+)`_+", r"\1", text)                        # références `nom`_
    text = re.sub(r"\|(\w+)\|", r"\1", text)
    return text


class RstConverter:
    """Convertit une page reStructuredText de la doc Airflow en Markdown simple."""

    def __init__(self, fetch: Fetcher, raw_base: str = AIRFLOW_RAW, docs_root: str = "airflow-core/docs"):
        self.fetch = fetch
        self.raw_base = raw_base
        self.docs_root = docs_root

    def convert(self, text: str) -> tuple[str, str]:
        md = self._convert_lines(text.replace("\t", "    ").split("\n"), heading_levels=[])
        m = re.search(r"^# (.+)$", md, re.M)
        return (m.group(1).strip() if m else ""), md

    # --- résolution des inclusions -----------------------------------------------------------
    def _resolve(self, path: str) -> str:
        # Chemins Sphinx : "/x" est relatif à la racine de la doc (airflow-core/docs).
        parts = []
        for p in (self.docs_root + "/" + path.lstrip("/")).split("/"):
            if p == "..":
                parts.pop()
            elif p and p != ".":
                parts.append(p)
        return self.fetch(f"{self.raw_base}/{'/'.join(parts)}")

    @staticmethod
    def _slice(content: str, opts: dict[str, str]) -> str:
        if "start-after" in opts:
            idx = content.find(opts["start-after"])
            if idx != -1:
                content = content[idx + len(opts["start-after"]):]
                content = content.split("\n", 1)[1] if "\n" in content else ""
        if "end-before" in opts:
            idx = content.find(opts["end-before"])
            if idx != -1:
                content = content[:idx]
                content = content.rsplit("\n", 1)[0]
        return content

    # --- parcours ligne à ligne -------------------------------------------------------------
    def _convert_lines(self, lines: list[str], heading_levels: list[str]) -> str:
        out: list[str] = []
        i, n = 0, len(lines)

        def block_after(start: int, base_indent: int) -> tuple[list[str], int]:
            """Lignes indentées (ou vides) qui suivent, au-delà de base_indent."""
            j, block = start, []
            while j < n and (not lines[j].strip() or _indent(lines[j]) > base_indent):
                block.append(lines[j])
                j += 1
            while block and not block[-1].strip():
                block.pop()
                j -= 1
            return block, j

        while i < n:
            line = lines[i]
            stripped = line.strip()
            indent = _indent(line)

            # Titres : ligne de texte soulignée (avec surlignement éventuel).
            if (stripped and i + 1 < n and _UNDERLINE.match(lines[i + 1].strip())
                    and len(lines[i + 1].strip()) >= len(stripped) * 0.8 and not _UNDERLINE.match(stripped)
                    and indent == 0):
                style = lines[i + 1].strip()[0] + ("o" if out and i > 0 and _UNDERLINE.match(lines[i - 1].strip()) else "")
                if style not in heading_levels:
                    heading_levels.append(style)
                if out and _UNDERLINE.match(out[-1].strip() if out else ""):
                    out.pop()  # surlignement déjà émis
                level = min(heading_levels.index(style) + 1, 6)
                out.append("")
                out.append("#" * level + " " + rst_inline(stripped))
                out.append("")
                i += 2
                continue

            m = re.match(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$", line)
            if m:
                name, arg = m.group(2).lower(), m.group(3).strip()
                j = i + 1
                opts: dict[str, str] = {}
                while j < n and re.match(r"^\s+:[\w-]+:", lines[j]) and _indent(lines[j]) > indent:
                    om = re.match(r"^\s+:([\w-]+):\s*(.*)$", lines[j])
                    opts[om.group(1)] = om.group(2).strip()
                    j += 1
                block, j = block_after(j, indent)
                content = textwrap.dedent("\n".join(block)).strip("\n")
                out.extend(self._directive(name, arg, opts, content, heading_levels).split("\n"))
                i = j
                continue

            # Cibles (.. _label:) et commentaires (.. texte) : supprimés avec leur bloc.
            if re.match(r"^\s*\.\.(\s|$)", line):
                _, j = block_after(i + 1, indent)
                i = j
                continue

            # Bloc littéral introduit par "::" en fin de paragraphe.
            if stripped.endswith("::") and not stripped.startswith(".."):
                text = rst_inline(line.rstrip()[:-2].rstrip())
                if text.strip():
                    out.append(text + ":")
                block, j = block_after(i + 1, indent)
                if block:
                    out += ["", "```", textwrap.dedent("\n".join(block)).strip("\n"), "```", ""]
                i = j
                continue

            out.append(rst_inline(line))
            i += 1
        return "\n".join(out)

    def _directive(self, name: str, arg: str, opts: dict[str, str], content: str, levels: list[str]) -> str:
        if name in ("code-block", "code", "sourcecode"):
            return f"\n```{arg}\n{content}\n```\n"
        if name == "exampleinclude":
            try:
                code = textwrap.dedent(self._slice(self._resolve(arg), opts)).strip("\n")
            except requests.RequestException:
                return ""
            return f"\n```{opts.get('language', '')}\n{code}\n```\n"
        if name == "include":
            try:
                included = self._slice(self._resolve(arg), opts)
            except requests.RequestException:
                return ""
            return self._convert_lines(included.split("\n"), levels)
        if name in _ADMONITIONS:
            body = self._convert_lines(((arg + "\n") if arg else "").split("\n") + content.split("\n"), levels).strip()
            return f"\n{_ADMONITIONS[name]}: {body}\n"
        if name in _VERSION_DIRECTIVES:
            body = self._convert_lines(content.split("\n"), levels).strip()
            return f"\n{_VERSION_DIRECTIVES[name]} {arg}.{(' ' + body) if body else ''}\n"
        if name in _DROPPED_DIRECTIVES:
            return ""
        return "\n" + self._convert_lines(content.split("\n"), levels) + "\n"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


# ---------------------------------------------------------------------------------------------

def build_document(name: str, public_url: str, source_url: str, fetch: Fetcher = http_fetch) -> str:
    raw = fetch(source_url)
    if source_url.endswith(".rst"):
        title, body = RstConverter(fetch).convert(raw)
        body = re.sub(r"^# .+\n", "", body.lstrip(), count=1)   # le titre est remis en tête
        ref = f"apache/airflow@{AIRFLOW_REF}"
    else:
        title, body = MdxConverter(fetch).convert(raw)
        ref = f"dbt-labs/docs.getdbt.com@{DBT_DOCS_REF[:12]}"
    body = _drop_sections(body, DROPPED_SECTIONS)
    product = "Apache Airflow" if name.startswith("airflow") else "dbt"
    header = f"# {product} — {title or name}\n\nSource : {public_url} (version des sources : {ref})\n\n"
    return _tidy(header + body)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (public_url, source_url) in PAGES.items():
        print(f"{name} <- {source_url}")
        md = build_document(name, public_url, source_url)
        path = CORPUS_DIR / f"{name}.md"
        path.write_text(md, encoding="utf-8")
        print(f"  -> {path.name} ({len(md.split())} mots)")
    print(f"\nCorpus reconstruit dans {CORPUS_DIR}. Relancer ensuite :")
    print("  python ingestion/chunking.py && python retrieval/vector_store.py")


if __name__ == "__main__":
    main()
