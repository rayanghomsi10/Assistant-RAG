"""Le jeu d'évaluation doit rester ancré dans le corpus versionné."""
import copy

from evaluation.dataset import load_corpus_texts, load_eval_dataset, validate_dataset


def test_eval_dataset_evidence_is_in_corpus():
    items = load_eval_dataset()
    assert validate_dataset(items, load_corpus_texts()) == []


def test_eval_dataset_composition():
    items = load_eval_dataset()
    in_scope = [it for it in items if it["type"] == "in_scope"]
    out_scope = [it for it in items if it["type"] == "out_of_scope"]
    assert len(in_scope) >= 60 and len(out_scope) >= 10
    # Les deux outils et les dix documents sont couverts.
    sources = {s for it in in_scope for s in it["expected_sources"]}
    assert len(sources) == 10


def test_validator_rejects_fabricated_evidence():
    items = copy.deepcopy(load_eval_dataset())
    items[0]["evidence"] = ["dbt models are written in COBOL"]
    errors = validate_dataset(items, load_corpus_texts())
    assert len(errors) == 1 and "preuve introuvable" in errors[0]


def test_validator_rejects_duplicates():
    items = load_eval_dataset()
    errors = validate_dataset(items + [copy.deepcopy(items[0])], load_corpus_texts())
    assert any("double" in e for e in errors)
