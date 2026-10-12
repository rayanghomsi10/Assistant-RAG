from rag_chain.checks import (REFUSAL_MESSAGE, check_answer, cited_documents, extract_code_spans,
                              extract_quotes, is_clean_refusal, is_refusal)

CONTEXT = ('XComs are explicitly "pushed" and "pulled" to/from their storage using the `xcom_push` '
           "and `xcom_pull` methods on Task Instances. They are only designed for small amounts of data.")


def test_refusal_detection():
    assert is_refusal(REFUSAL_MESSAGE)
    assert is_refusal("Je ne trouve pas d’information à ce sujet.")
    assert is_refusal("Les documents fournis ne mentionnent pas cette option.")
    assert not is_refusal("On utilise xcom_push [airflow_04_xcoms.md].")


def test_clean_refusal_rejects_refusal_followed_by_an_answer():
    long = REFUSAL_MESSAGE + " Cependant, " + "en général on fait ceci " * 10
    assert is_refusal(long) and not is_clean_refusal(long)
    assert is_clean_refusal(REFUSAL_MESSAGE)


def test_cited_documents_normalizes_extension():
    answer = "Voir [airflow_04_xcoms.md] et airflow_01_dags, ainsi que [dbt_03_tests]."
    assert cited_documents(answer) == {"airflow_04_xcoms.md", "airflow_01_dags.md", "dbt_03_tests.md"}


def test_quotes_short_terms_ignored():
    answer = 'Les tâches sont des "views". Selon la doc : « They are only designed for small amounts of data ».'
    assert extract_quotes(answer) == ["They are only designed for small amounts of data"]


def test_code_spans_exclude_document_names():
    assert extract_code_spans("Utilise `xcom_push()` (voir `airflow_04_xcoms.md`).") == ["xcom_push()"]


def test_check_answer_valid():
    answer = ("On utilise la méthode `xcom_push()` de l'instance de tâche [airflow_04_xcoms.md] ; "
              "« They are only designed for small amounts of data ».")
    chk = check_answer(answer, [CONTEXT], ["airflow_04_xcoms.md"])
    assert not chk.refusal
    assert chk.citations_valid and chk.has_citation
    assert chk.quotes_faithful
    assert chk.code_support == 1.0


def test_check_answer_detects_fabrications():
    answer = ("Utilisez `push_value` [dbt_03_tests.md] : « XComs can store dataframes of any size ».")
    chk = check_answer(answer, [CONTEXT], ["airflow_04_xcoms.md"])
    assert chk.cited_not_retrieved == ["dbt_03_tests.md"]
    assert chk.unsupported_quotes == ["XComs can store dataframes of any size"]
    assert chk.unsupported_code == ["push_value"]
    assert chk.code_support == 0.0
