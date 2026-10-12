from ingestion.build_corpus import MdxConverter, RstConverter, _drop_sections, rst_inline

RAW = "https://raw.example"


def fake_fetch(files: dict[str, str]):
    def _fetch(url: str) -> str:
        return files[url]
    return _fetch


def test_rst_inline_roles_and_links():
    assert rst_inline(":class:`~airflow.sdk.bases.xcom.BaseXCom`") == "`BaseXCom`"
    assert rst_inline(":ref:`TaskFlow arguments <concepts:args>`") == "TaskFlow arguments"
    assert rst_inline("``xcom_pull``") == "`xcom_pull`"
    assert rst_inline("see `Jinja docs <https://jinja.example>`_.") == "see Jinja docs."


def test_rst_converter_headings_directives_and_comments():
    rst = """ .. Licensed to the Apache Software Foundation
    under the License.

.. _concepts:xcom:

XComs
=====

Use ``xcom_push``::

    task_instance.xcom_push(key="k", value=1)

.. note::

  ``xcom_pull()`` without ``task_ids`` pulls only from the current task.

.. image:: /img/x.png
    :width: 100

Custom Backends
---------------

Subclass :class:`~airflow.sdk.bases.xcom.BaseXCom`.
"""
    title, md = RstConverter(fake_fetch({}), raw_base=RAW).convert(rst)
    assert title == "XComs"
    assert "Licensed" not in md and "concepts:xcom" not in md and "image" not in md
    assert "Use `xcom_push`:" in md
    assert '```\ntask_instance.xcom_push(key="k", value=1)\n```' in md
    assert "Note: `xcom_pull()` without `task_ids` pulls only from the current task." in md
    assert "## Custom Backends" in md and "Subclass `BaseXCom`." in md


def test_rst_exampleinclude_is_resolved():
    rst = """Title
=====

.. exampleinclude:: /../src/example.py
    :language: python
    :start-after: [START x]
    :end-before: [END x]
"""
    files = {f"{RAW}/airflow-core/src/example.py": "import a\n# [START x]\nprint(1)\n# [END x]\n"}
    _, md = RstConverter(fake_fetch(files), raw_base=RAW).convert(rst)
    assert "```python\nprint(1)\n```" in md


def test_mdx_converter_components():
    mdx = """---
title: "SQL models"
---

import Snip from '/snippets/_snip.md';

<Snip resource='models' />

A model is a `select` statement in <Constant name="dbt" />, see [ref docs](/reference/ref).

:::info Good to know

Models live in <Term id="data-warehouse" /> schemas &mdash; always.

:::

<File name='models/a.sql'>

```sql title="models/a.sql"
select 1
```

</File>

<FAQ path="Models/q" />
"""
    files = {
        f"{RAW}/constants.js": "export const CONSTANTS = {\n  dbt: 'dbt',\n  core: 'dbt v1',\n};",
        f"{RAW}/snippets/_snip.md": "Generate {props.resource} quickly.",
        f"{RAW}/docs/faqs/Models/q.md": "---\ntitle: Can I?\n---\nYes you can.",
    }
    title, md = MdxConverter(fake_fetch(files), raw_base=RAW).convert(mdx)
    assert title == "SQL models"
    assert "Generate models quickly." in md
    assert "A model is a `select` statement in dbt, see ref docs." in md
    assert "Info: Good to know" in md and ":::" not in md
    assert "data warehouse schemas — always." in md
    assert "File `models/a.sql`:" in md and "```sql\nselect 1\n```" in md
    assert "#### FAQ: Can I?" in md and "Yes you can." in md
    assert "<" not in md.replace("<!--", "")


def test_drop_navigation_section_keeps_following_paragraph():
    md = "## Related reference docs\n* [a](/a)\n* [b](/b)\n\nImportant: keep me.\n\n## Overview\ntext"
    out = _drop_sections(md, {"related reference docs"})
    assert "Related" not in out and "* [a]" not in out
    assert "Important: keep me." in out and "## Overview" in out
