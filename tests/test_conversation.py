"""Mémoire de conversation et compléments, sans modèle."""
import json

from rag_chain import chain
from rag_chain.complements import is_caveat, keep_grounded, parse_complements

HITS = [{"source": "airflow_04_xcoms.md", "section": "XComs", "text": "XComs are for small amounts of data."}]


def test_prompt_without_history_is_unchanged():
    msgs = chain.build_messages("Qu'est-ce qu'un XCom ?", HITS)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "Qu'est-ce qu'un XCom ?"
    assert chain.build_messages("Q", HITS, history=[]) == chain.build_messages("Q", HITS)


def test_history_is_inserted_before_the_question_and_truncated():
    history = [{"question": f"Q{i}", "answer": f"R{i}"} for i in range(5)]
    msgs = chain.build_messages("Et ensuite ?", HITS, history=history)
    kept = chain.HISTORY_MAX_TURNS
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * kept + ["user"]
    assert msgs[1]["content"] == f"Q{5 - kept}" and msgs[-1]["content"] == "Et ensuite ?"
    long = chain.build_messages("Q", HITS, history=[{"question": "Q1", "answer": "x" * 5000}])
    assert len(long[2]["content"]) == chain.HISTORY_ANSWER_MAX_CHARS


def test_condense_without_history_does_not_call_the_llm(monkeypatch):
    monkeypatch.setattr(chain, "get_llm", lambda *a, **k: (_ for _ in ()).throw(AssertionError("appel LLM")))
    assert chain.condense_question("Qu'est-ce qu'un XCom ?", None) == "Qu'est-ce qu'un XCom ?"
    assert chain.condense_question("Qu'est-ce qu'un XCom ?", []) == "Qu'est-ce qu'un XCom ?"


def test_conversation_dataset_is_valid():
    from evaluation.dataset import load_eval_dataset
    from evaluation.run_conversation_eval import load_conversations, validate_conversations
    items = {it["id"]: it for it in load_eval_dataset()}
    conversations = load_conversations()
    assert validate_conversations(conversations, items) == []
    assert len(conversations) >= 15
    broken = [{"id": "x", "history": ["inconnu"], "question": "?", "target": "dbt_incr_04"}]
    assert validate_conversations(broken, items)


def test_is_caveat():
    assert is_caveat({"source": "dbt_03_tests.md", "text": "Intro\n\nNote: data tests run after the build."})
    assert is_caveat({"source": "airflow_05_best_practices.md", "text": "Avoid top-level code."})
    assert not is_caveat({"source": "dbt_03_tests.md", "text": "A note about tests: nothing special."})


def test_parse_complements_keeps_only_grounded_items():
    caveats = [{"source": "airflow_04_xcoms.md", "section": "XComs",
                "text": "Warning: XComs are designed for small amounts of data. Use `xcom_backend` for more."}]
    raw = json.dumps({"complements": [
        {"type": "mise_en_garde", "texte": "Les XComs sont faits pour de petites données.",
         "source": "[airflow_04_xcoms.md]"},
        {"type": "mise_en_garde", "texte": "Inventé.", "source": "dbt_03_tests.md"},          # document absent
        {"type": "alternative", "texte": "Utiliser `custom_storage`.", "source": "airflow_04_xcoms.md"},  # code absent
        {"type": "a_savoir", "texte": "La doc dit « XComs are unlimited and fast ».",            # citation inventée
         "source": "airflow_04_xcoms.md"},
        {"type": "opinion", "texte": "Type inconnu.", "source": "airflow_04_xcoms.md"},
        {"type": "alternative", "texte": "Configurer `xcom_backend`.", "source": "airflow_04_xcoms.md"},
    ]})
    out = parse_complements(raw, caveats, max_items=3)
    assert [(c["type"], c["source"]) for c in out] == [("mise_en_garde", "airflow_04_xcoms.md"),
                                                       ("alternative", "airflow_04_xcoms.md")]
    assert out[0]["label"] == "Mise en garde" and out[0]["section"] == "XComs"
    question = json.dumps({"complements": [{"type": "a_savoir", "texte": "Quelle est la taille maximale ?",
                                            "source": "airflow_04_xcoms.md"}]})
    assert parse_complements(question, caveats) == []                 # une question n'est pas une information
    assert parse_complements("pas du JSON", caveats) == []
    assert parse_complements('{"complements": "texte"}', caveats) == []


def test_keep_grounded_uses_nli():
    class FakeNli:  # implication forte seulement si la phrase parle de "petites"
        def entailment(self, premises, hypothesis):
            return [0.9 if "petites" in hypothesis else 0.1 for _ in premises]

    caveats = [{"source": "airflow_04_xcoms.md", "text": "XComs are designed for small amounts of data."}]
    doc = "airflow_04_xcoms.md"
    items = [{"text": "Les XComs sont faits pour de petites quantités de données.", "source": doc},
             {"text": "Il faut toujours utiliser des variables globales pour partager.", "source": doc}]
    kept = keep_grounded(items, caveats, FakeNli(), min_entailment=0.5)
    assert [k["text"] for k in kept] == [items[0]["text"]] and kept[0]["grounding"] == 0.9
