import numpy as np

from evaluation.metrics import bootstrap_ci, generation_metrics, paired_bootstrap, retrieval_metrics, suggest_threshold

ITEM = {"id": "x", "type": "in_scope", "expected_sources": ["airflow_04_xcoms.md"],
        "evidence": ["only designed for small amounts of data", "xcom_push and xcom_pull methods"]}


def test_retrieval_metrics_rank_and_recall(make_hit):
    hits = [make_hit("unrelated text", source="dbt_01_sql_models.md", idx=0),
            make_hit("They are **only** designed for small amounts of data.", idx=1)]
    m = retrieval_metrics(ITEM, hits)
    assert m["evidence_recall"] == 0.5
    assert m["evidence_hit"] == 1.0 and m["source_hit"] == 1.0
    assert m["mrr"] == 0.5


def test_retrieval_metrics_miss(make_hit):
    m = retrieval_metrics(ITEM, [make_hit("nothing here", source="dbt_01_sql_models.md")])
    assert m == {"evidence_recall": 0.0, "evidence_hit": 0.0, "source_hit": 0.0, "mrr": 0.0}


def test_false_refusal_only_when_evidence_was_retrieved(make_hit):
    hits = [make_hit("They are only designed for small amounts of data.")]
    refusal = "Je ne trouve pas cette information dans les documents fournis."
    assert generation_metrics(ITEM, refusal, hits, evidence_hit=1.0)["false_refusal"] == 1.0
    assert generation_metrics(ITEM, refusal, hits, evidence_hit=0.0)["false_refusal"] == 0.0
    oos = {"id": "o", "type": "out_of_scope", "expected_sources": [], "evidence": []}
    assert generation_metrics(oos, refusal, hits) == {"correct_refusal": 1.0, "clean_refusal": 1.0}


def test_bootstrap_ci_contains_mean_and_ignores_nan():
    mean, lo, hi, n = bootstrap_ci([1, 0, 1, 1, np.nan, 0, 1, 1])
    assert n == 7 and lo <= mean <= hi
    assert np.isnan(bootstrap_ci([np.nan])[0])


def test_paired_bootstrap_detects_consistent_improvement():
    a = np.zeros(40)
    b = np.r_[np.ones(30), np.zeros(10)]
    res = paired_bootstrap(a, b)
    assert res["delta"] == 0.75 and res["ci_low"] > 0 and res["p_value"] < 0.01


def test_paired_bootstrap_no_effect():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 40).astype(float)
    res = paired_bootstrap(a, a.copy())
    assert res["delta"] == 0 and res["ci_low"] <= 0 <= res["ci_high"]


def test_suggest_threshold_separates_scores():
    best = suggest_threshold([0.8, 0.9, 0.7], [0.1, 0.3, 0.2])
    assert best["balanced_accuracy"] == 1.0 and 0.3 < best["threshold"] <= 0.7
