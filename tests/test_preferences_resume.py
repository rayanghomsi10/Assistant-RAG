"""Reprise de build_preferences.py après une interruption, et traçage inactif sans clés."""
import json
from collections import Counter

import pytest

from rag_chain import tracing
from training import build_preferences as bp

QUESTIONS = [{"id": f"q{i}", "type": "in_scope", "question": f"Question {i} ?", "source_chunk_id": "c",
              "source": "d.md"} for i in range(5)]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "TRAINING_DATA_DIR", tmp_path)
    monkeypatch.setattr(bp, "PREFERENCES_PATH", tmp_path / "preferences.jsonl")
    monkeypatch.setattr(bp, "PROGRESS_PATH", tmp_path / "preferences_progress.jsonl")
    monkeypatch.setattr(bp, "load_questions", lambda: QUESTIONS)
    calls = []

    def fake_process(n, q, args, nli, out, crash_on=None):
        calls.append(q["id"])
        out.write(json.dumps({"id": q["id"]}) + "\n")   # une paire par question
        out.flush()
        if q["id"] == crash_on:
            raise KeyboardInterrupt  # interruption après l'écriture de la paire
        return Counter({"pairs_in_scope": 1}), "paire"

    return tmp_path, calls, fake_process, monkeypatch


def pair_ids(tmp_path):
    return [json.loads(line)["id"] for line in (tmp_path / "preferences.jsonl").read_text().splitlines()]


def test_resume_after_interruption(run):
    tmp_path, calls, fake_process, monkeypatch = run
    monkeypatch.setattr(bp, "process_question", lambda *a: fake_process(*a, crash_on="q2"))
    with pytest.raises(KeyboardInterrupt):
        bp.main(["--no-nli"])
    assert calls == ["q0", "q1", "q2"]

    monkeypatch.setattr(bp, "process_question", fake_process)
    bp.main(["--no-nli"])
    # q2 a été interrompue avant sa ligne de progression : refaite, sans paire en double.
    assert calls == ["q0", "q1", "q2", "q2", "q3", "q4"]
    assert pair_ids(tmp_path) == ["q0", "q1", "q2", "q3", "q4"]
    stats = json.loads((tmp_path / "preferences_stats.json").read_text(encoding="utf-8"))
    assert stats == {"pairs_in_scope": 5}


def test_resume_refuses_changed_parameters(run):
    tmp_path, calls, fake_process, monkeypatch = run
    monkeypatch.setattr(bp, "process_question", lambda *a: fake_process(*a, crash_on="q1"))
    with pytest.raises(KeyboardInterrupt):
        bp.main(["--no-nli", "--samples", "4"])
    monkeypatch.setattr(bp, "process_question", fake_process)
    with pytest.raises(SystemExit, match="--restart"):
        bp.main(["--no-nli", "--samples", "2"])
    bp.main(["--no-nli", "--samples", "2", "--restart"])
    assert pair_ids(tmp_path) == ["q0", "q1", "q2", "q3", "q4"]


def test_old_preferences_file_is_not_overwritten(run):
    tmp_path, _, fake_process, monkeypatch = run
    (tmp_path / "preferences.jsonl").write_text('{"id": "ancien"}\n', encoding="utf-8")
    monkeypatch.setattr(bp, "process_question", fake_process)
    with pytest.raises(SystemExit, match="--restart"):
        bp.main(["--no-nli"])
    assert pair_ids(tmp_path) == ["ancien"]


def test_tracing_is_a_noop_without_keys(monkeypatch):
    monkeypatch.setattr(tracing, "is_active", lambda: False)

    def f(x):
        return x * 2

    assert tracing.observe("x")(f) is f
    assert tracing.langchain_config("génération") == {"run_name": "génération"}
    with tracing.trace("t", session_id="s", tags=["a"], input="q"), tracing.attributes(tags=["api"]):
        tracing.set_output("r")
        tracing.score("m", 1.0)
    assert tracing.current_trace_id() is None
    tracing.flush()
