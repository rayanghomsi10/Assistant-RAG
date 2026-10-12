"""Tests du pipeline DPO sans modèle : récompense, sélection des paires, anti-fuite, export."""
import json

import pytest

from evaluation.dataset import load_eval_dataset
from rag_chain.checks import REFUSAL_MESSAGE
from training.build_preferences import select_pair
from training.export_ollama import adapter_modelfile, export_adapter
from training.generate_questions import OUT_OF_SCOPE_PATH, LeakageFilter, content_tokens, dedupe, jaccard
from training.reward import answer_sentences, compute_reward, concision
from training.train_dpo import OLLAMA_LLAMA3_CHAT_TEMPLATE, split_pairs

CTX = ("XComs are explicitly pushed and pulled using the `xcom_push` and `xcom_pull` methods on Task Instances. "
       "They are only designed for small amounts of data.")
GOOD = ("Pour pousser une valeur, on appelle la méthode `xcom_push` de l'instance de tâche, avec une clé et une "
        "valeur [airflow_04_xcoms.md]. Les XComs sont prévus pour de petites quantités de données.")


@pytest.fixture
def hits(make_hit):
    return [make_hit(CTX)]


def test_reward_good_answer(hits):
    r = compute_reward(GOOD, "in_scope", hits, "airflow_04_xcoms.md")
    assert r.passed and r.total == 1.0


@pytest.mark.parametrize("answer, reason", [
    (REFUSAL_MESSAGE, "false_refusal"),
    ("Utilisez xcom_push [airflow_04_xcoms.md].", "too_short"),
    (GOOD.replace("airflow_04_xcoms.md", "dbt_03_tests.md"), "cites_unretrieved_document"),
    (GOOD + " Comme le dit la doc : « XComs can store any dataframe of any size ».", "fabricated_quote"),
])
def test_reward_hard_constraints(hits, answer, reason):
    r = compute_reward(answer, "in_scope", hits, "airflow_04_xcoms.md")
    assert not r.passed and r.hard_fail == reason and r.total == 0.0


def test_reward_out_of_scope(hits):
    assert compute_reward(REFUSAL_MESSAGE, "out_of_scope", hits).total == 1.0
    assert compute_reward(GOOD, "out_of_scope", hits).hard_fail == "answered_out_of_scope"
    rambling = REFUSAL_MESSAGE + " Mais " + "voici une réponse inventée " * 10
    assert compute_reward(rambling, "out_of_scope", hits).total == 0.5


def test_reward_soft_components(hits):
    no_source = GOOD.replace(" [airflow_04_xcoms.md]", "")
    r = compute_reward(no_source, "in_scope", hits, "airflow_04_xcoms.md")
    assert r.passed and r.components["cites_source"] == 0.0 and r.total < 1.0
    bad_code = GOOD.replace("`xcom_push`", "`push_xcom_value`")
    assert compute_reward(bad_code, "in_scope", hits, "airflow_04_xcoms.md").components["code_support"] == 0.0


def test_reward_uses_nli_when_given(hits):
    class FakeNli:
        def entailment(self, premises, hypothesis):
            return [0.0 for _ in premises]
    r = compute_reward(GOOD, "in_scope", hits, "airflow_04_xcoms.md", nli=FakeNli())
    assert r.components["grounding"] == 0.0 and r.total < 1.0


def test_concision_and_sentences():
    assert concision(100) == 1.0 and concision(400) == 0.0 and 0 < concision(250) < 1
    assert answer_sentences("Première phrase assez longue ici. Court. [doc.md] Deuxième phrase également longue.") == [
        "Première phrase assez longue ici.", "Deuxième phrase également longue."]


def test_select_pair(hits):
    answers = [GOOD, REFUSAL_MESSAGE, GOOD.replace(" [airflow_04_xcoms.md]", "")]
    scored = [(a, compute_reward(a, "in_scope", hits, "airflow_04_xcoms.md")) for a in answers]
    (chosen, rc), (rejected, rr) = select_pair(scored, min_margin=0.2)
    assert chosen == GOOD and rejected == REFUSAL_MESSAGE
    # Aucune réponse valide -> pas de paire.
    bad = [(REFUSAL_MESSAGE, compute_reward(REFUSAL_MESSAGE, "in_scope", hits, "airflow_04_xcoms.md"))]
    assert select_pair(bad, 0.2) is None
    # Écart insuffisant -> pas de paire.
    same = [(GOOD, scored[0][1]), (GOOD + " ", scored[0][1])]
    assert select_pair(same, 0.2) is None


def test_out_of_scope_training_questions_disjoint_from_eval():
    train = json.loads(OUT_OF_SCOPE_PATH.read_text(encoding="utf-8"))
    eval_qs = [it["question"] for it in load_eval_dataset()]
    assert len(train) == len(set(train)) >= 50
    flt = LeakageFilter(eval_qs, use_embeddings=False)
    assert all(flt.keep(train))


def test_leakage_filter_catches_paraphrase():
    eval_qs = [it["question"] for it in load_eval_dataset()]
    flt = LeakageFilter(eval_qs, use_embeddings=False)
    near_copy = "Quels sont les cinq types de matérialisations intégrés dans dbt ?"
    assert flt.keep([near_copy, "Comment déclarer une source dans un fichier YAML dbt ?"]) == [False, True]


def test_dedupe():
    rows = [{"question": "Comment fonctionne xcom_push dans Airflow ?"},
            {"question": "Comment fonctionne xcom_push dans Airflow ?"},
            {"question": "Qu'est-ce qu'un modèle ephemeral dans dbt ?"}]
    assert len(dedupe(rows)) == 2
    assert jaccard(content_tokens("a b c"), set()) == 0.0


def test_split_pairs_is_deterministic():
    pairs = [{"id": str(i)} for i in range(50)]
    a, b = split_pairs(pairs, 0.1, seed=1), split_pairs(pairs, 0.1, seed=1)
    assert a == b and len(a[1]) == 5 and len(a[0]) == 45


def test_export_adapter(tmp_path):
    adapter = tmp_path / "dpo-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    modelfile = export_adapter(adapter, tmp_path / "ollama", "llama3.1:8b")
    text = modelfile.read_text()
    assert text.startswith("FROM llama3.1:8b\nADAPTER ./adapter\n")
    assert (tmp_path / "ollama" / "adapter" / "adapter_model.safetensors").exists()
    assert "num_ctx" in adapter_modelfile("x")


def test_chat_template_matches_ollama_format():
    jinja2 = pytest.importorskip("jinja2")
    tpl = jinja2.Environment().from_string(OLLAMA_LLAMA3_CHAT_TEMPLATE)
    out = tpl.render(bos_token="<|begin_of_text|>", add_generation_prompt=True,
                     messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}])
    assert out == ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>"
                   "<|start_header_id|>user<|end_header_id|>\n\nQ<|eot_id|>"
                   "<|start_header_id|>assistant<|end_header_id|>\n\n")
