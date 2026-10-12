"""Tests de bout en bout de l'évaluation et de l'API, avec retrieval et LLM simulés
(aucun modèle, aucune base vectorielle, aucun serveur Ollama)."""
import json

import pytest

from evaluation import run_eval
from evaluation.dataset import load_eval_dataset
from rag_chain.checks import REFUSAL_MESSAGE

ITEMS = {it["question"]: it for it in load_eval_dataset()}


def fake_search(question, k=4, **_):
    it = ITEMS[question]
    if it["type"] == "out_of_scope":
        return [{"id": "dbt_01_sql_models.md::chunk_000", "text": "Unrelated text.", "source": "dbt_01_sql_models.md",
                 "section": "", "url": "", "similarity": 0.2, "rerank_score": -5.0, "score": -5.0}]
    return [{"id": f"{it['expected_sources'][0]}::chunk_000", "text": " ".join(it["evidence"]),
             "source": it["expected_sources"][0], "section": "", "url": "", "similarity": 0.8,
             "rerank_score": 5.0, "score": 5.0}]


def fake_generate(question, hits, model=None, **_):
    it = ITEMS[question]
    if it["type"] == "out_of_scope":
        return REFUSAL_MESSAGE
    return f"{it['ground_truth']} [{it['expected_sources'][0]}] « {it['evidence'][0]} »"


@pytest.fixture
def patched_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(run_eval, "search", fake_search)
    monkeypatch.setattr(run_eval, "generate", fake_generate)
    monkeypatch.setattr(run_eval, "get_index_metadata", lambda: {"embedding_model": "fake", "corpus_sha": "abc"})
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_run_eval_deterministic_metrics(patched_eval):
    summary = run_eval.main(["--run-name", "t", "--skip-ragas", "--no-mlflow"])
    for m in ("evidence_recall", "evidence_hit", "source_hit", "mrr", "answered", "has_citation",
              "citations_valid", "cites_expected_source", "quotes_faithful", "correct_refusal"):
        assert summary[m]["mean"] == 1.0, m
    assert summary["false_refusal"]["mean"] == 0.0
    saved = json.loads((patched_eval / "t" / "summary.json").read_text())
    assert saved["params"]["index_embedding_model"] == "fake"
    assert saved["params"]["mode"] == "no_ragas"
    assert saved["threshold_suggestion"]["balanced_accuracy"] == 1.0


def test_run_eval_retrieval_only(patched_eval):
    summary = run_eval.main(["--run-name", "r", "--retrieval-only", "--no-mlflow"])
    assert summary["evidence_recall"]["n"] == 73
    assert "answered" not in summary


def test_compare_runs_detects_regression(patched_eval, monkeypatch):
    from evaluation import compare_runs
    run_eval.main(["--run-name", "a", "--skip-ragas", "--no-mlflow"])
    monkeypatch.setattr(run_eval, "generate", lambda q, h, **_: REFUSAL_MESSAGE)  # tout refuser
    run_eval.main(["--run-name", "b", "--skip-ragas", "--no-mlflow"])
    monkeypatch.setattr(compare_runs, "RESULTS_DIR", patched_eval)
    table = compare_runs.main(["a", "b"]).set_index("metric")
    assert table.loc["answered", "verdict"] == "B moins bon"
    assert table.loc["false_refusal", "verdict"] == "B moins bon"
    assert table.loc["correct_refusal", "verdict"] == "non significatif"


def test_api_query(monkeypatch):
    from fastapi.testclient import TestClient

    from api import main as api_main
    monkeypatch.setattr(api_main, "index_is_stale", lambda: False)
    monkeypatch.setattr(api_main, "ensure_index_ready", lambda: None)
    monkeypatch.setattr(api_main, "answer_question", lambda q, k: {
        "question": q, "answer": "Avec `xcom_push` [airflow_04_xcoms.md].", "refused": False,
        "cited_sources": ["airflow_04_xcoms.md"], "retrieved_sources": ["airflow_04_xcoms.md"], "model": "m"})
    with TestClient(api_main.app) as client:
        assert client.get("/health").json()["index_ready"] is True
        r = client.post("/query", json={"question": "Comment pousser un XCom ?"})
        assert r.status_code == 200
        assert r.json()["cited_sources"] == ["airflow_04_xcoms.md"]
        assert client.post("/query", json={"question": ""}).status_code == 422
