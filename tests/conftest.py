import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Les tests ne dépendent d'aucun serveur Langfuse, même si des clés figurent dans le .env local :
# défini avant l'import de config (load_dotenv ne remplace pas une variable déjà présente).
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"


@pytest.fixture
def word_counter():
    """Compteur de tokens approximatif (sans télécharger de tokenizer)."""
    return lambda text: len(re.findall(r"\w+|[^\w\s]", text))


@pytest.fixture
def make_hit():
    def _make(text, source="airflow_04_xcoms.md", score=1.0, idx=0):
        return {"id": f"{source}::chunk_{idx:03d}", "text": text, "source": source, "section": "",
                "url": "", "similarity": score, "rerank_score": None, "score": score}
    return _make
