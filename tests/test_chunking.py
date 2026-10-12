from ingestion.chunking import chunk_document, corpus_fingerprint, load_corpus_documents, parse_sections

DOC = """# Apache Airflow — XComs

Source : https://example.org/xcoms (version des sources : test)

Intro paragraph about XComs.

## Pushing

Use xcom_push.

```python
# this is a comment, not a heading
task_instance.xcom_push(key="k", value=1)
```

### Details

More details.

## Backends

""" + "Long backend text. " * 300


def test_parse_sections_ignores_hashes_in_code():
    url, sections = parse_sections(DOC)
    assert url == "https://example.org/xcoms"
    paths = [s.path for s in sections]
    assert paths == [["Apache Airflow — XComs"],
                     ["Apache Airflow — XComs", "Pushing"],
                     ["Apache Airflow — XComs", "Pushing", "Details"],
                     ["Apache Airflow — XComs", "Backends"]]
    assert "# this is a comment" in sections[1].body


def test_chunks_respect_size_and_carry_breadcrumb(word_counter):
    chunks = chunk_document("airflow_04_xcoms.md", DOC, word_counter, chunk_size=120, chunk_overlap=20)
    assert all(c.n_tokens <= 120 for c in chunks)
    # Les petites sous-sections d'une même section de niveau 2 sont regroupées.
    pushing = [c for c in chunks if c.section.endswith("Pushing")]
    assert len(pushing) == 1 and "Details" in pushing[0].text and "More details." in pushing[0].text
    # La longue section est recoupée, chaque morceau garde le fil d'Ariane.
    backends = [c for c in chunks if c.section.endswith("Backends")]
    assert len(backends) > 1
    assert all(c.text.startswith("Apache Airflow — XComs > Backends\n\n") for c in backends)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.url == "https://example.org/xcoms" for c in chunks)


def test_real_corpus_chunks_within_budget(word_counter):
    docs = load_corpus_documents()
    assert len(docs) == 10
    for doc in docs:
        chunks = chunk_document(doc["source"], doc["text"], word_counter, chunk_size=400, chunk_overlap=50)
        assert chunks and all(c.n_tokens <= 400 for c in chunks)
        assert all(c.text.split("\n", 1)[0].startswith(("dbt — ", "Apache Airflow — ")) for c in chunks)


def test_corpus_fingerprint_changes_with_content():
    a = [{"source": "a.md", "text": "x"}]
    b = [{"source": "a.md", "text": "y"}]
    assert corpus_fingerprint(a) != corpus_fingerprint(b)
    assert corpus_fingerprint(a) == corpus_fingerprint(list(a))
