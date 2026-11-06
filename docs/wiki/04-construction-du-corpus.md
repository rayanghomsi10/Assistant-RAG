# 4. Construction du corpus

`ingestion/build_corpus.py` (484 lignes) transforme les sources de 10 pages de documentation officielle en 10 fichiers Markdown propres, reproductibles au caractère près.

## 4.1 Rôle et sources

Le corpus couvre un domaine cohérent, l'outillage data engineering :

|Fichier|Page|Mots|
|---|---|---|
|`dbt_01_sql_models.md`|Modèles SQL, configuration, `ref`, FAQ|2 739|
|`dbt_02_jinja_macros.md`|Jinja, macros, packages, bonnes pratiques|1 819|
|`dbt_03_tests.md`|Data tests singuliers et génériques, stockage des échecs|3 189|
|`dbt_04_incremental_models.md`|Modèles incrémentaux, `unique_key`, `on_schema_change`|2 316|
|`dbt_05_materializations.md`|view, table, incremental, ephemeral, materialized view|1 433|
|`airflow_01_dags.md`|Déclaration, dépendances, branchements, trigger rules|5 749|
|`airflow_02_operators.md`|Opérateurs, templating Jinja|1 809|
|`airflow_03_scheduler.md`|Scheduler, haute disponibilité, réglages|2 129|
|`airflow_04_xcoms.md`|XComs, backends|794|
|`airflow_05_best_practices.md`|Bonnes pratiques d'écriture et de test des Dags|7 701|

Les sources sont lues dans les dépôts GitHub officiels, à des versions **figées dans le code** : le commit `3aea3292…` de `dbt-labs/docs.getdbt.com` (fichiers MDX) et le tag `3.3.1` d' `apache/airflow` (fichiers reStructuredText). Chaque document commence par son titre, préfixé du produit (« dbt — », « Apache Airflow — »), et une ligne `Source :` qui indique l'URL publique et la version.

## 4.2 Fonctionnement

### Côté dbt : convertir du MDX

Le site de dbt est écrit en MDX : du Markdown mêlé de composants React. Le convertisseur `MdxConverter` :

lit le _front matter_ YAML (titre) ;

- sépare le texte en segments « prose » et « bloc de code » (`_split_fences`), pour ne jamais modifier le code ;

- parcourt les balises JSX avec un petit analyseur (`_scan_jsx_tags`) qui gère les balises multilignes et les props entre accolades ;

- remplace chaque composant par son équivalent textuel : `<Constant name="core"/>` par sa valeur lue dans `constants.js` du site (« dbt v1 ») ; `<Term id="data-warehouse"/>` par « data warehouse » ; `<File` `name="…">` par « File `…`: » ; `<TabItem>` par le libellé de l'onglet ; `<FAQ path="…"/>` par le contenu de la FAQ, téléchargé et inséré comme sous-section ; un fragment importé depuis `/snippets/` par son contenu, avec substitution des `{props.x}` ;

convertit les encadrés `:::info` en « Info: … », supprime les images, garde le texte des liens.

### Côté Airflow : convertir du reStructuredText

`RstConverter` parcourt le fichier ligne par ligne :

- **titres** : en RST, un titre est une ligne soulignée (`=====`, `-----` …), et le niveau dépend de l'ordre d'apparition des styles de soulignement. Le convertisseur mémorise cet ordre pour produire # , `##`, `###` ;

- **directives** (`.. code-block::`, `.. note::`, `.. versionadded::` …) : converties en blocs de code ou en paragraphes « Note: … » ; les images et tables des matières sont supprimées ;

- **inclusions** : `.. include::` et `.. exampleinclude::` (exemples de code tirés des Dags d'exemple d'Airflow) sont résolues en téléchargeant le fichier cible et en extrayant la portion entre les marqueurs `:start-after:` et `:end-before:` ;

- **rôles en ligne** (`rst_inline`) : `:class:`~airflow.sdk.BaseXCom devient ``BaseXCom , `:ref:`texte` `<cible> devient « texte », ```code` devient ``code`` ;

- **blocs littéraux** introduits par `::` en fin de paragraphe : convertis en bloc de code ;

- **commentaires et cibles** (en-tête de licence, `.. _label:`) : supprimés.

Enfin, `build_document` supprime les sections de pure navigation (« Related reference docs », listes de liens), ajoute l'en-tête et normalise les espaces (`_tidy`).

## 4.3 Fonctions

|Fonction|Rôle|
|---|---|
|**`http_fetch`** `(url) → str`|Télécharge une URL (requests, timeout 30 s). Injectable : les tests passent un faux `fetch`.|
|**`_split_fences`** `(text) → [(est_code, segment)]`|Sépare prose et blocs de code délimités par `ou `~~~`.|
|**`_drop_sections`** `(markdown, titles) → str`|Supprime un titre de navigation et la liste de liens qui le suit ; s'arrête au premier paragraphe non-liste.|
|**`_tidy`** `(text) → str`|Supprime les espaces de fin de ligne et les lignes vides multiples.|
|**`parse_frontmatter`** `(text) → (meta, corps)`|Extrait le bloc YAML entre `---`(clés simples).|
|**`_scan_jsx_tags`** `(text) → itérateur`|Repère les balises JSX ouvrantes, fermantes ou auto-fermantes, en suivant guillemets et accolades.|
|**`_prop`**,**`_term_label`**|Lit un attribut de balise ; transforme un identifiant de terme en libellé (« cte » → « CTE »).|
|**`MdxConverter`** `(`fetch`, raw_base)`|Convertisseur MDX. Méthodes : `constants`(lit `constants.js`, en cache), `convert(text, heading_shift)`→ (titre, markdown), `_clean_code_fence`, `_convert_prose`, `_faq`.|
|**`rst_inline`** `(text) → str`|Convertit les rôles et liens RST en ligne.|
|**`RstConverter`**|Convertisseur RST. Méthodes : `convert`, `_resolve`(chemins Sphinx relatifs),|
|`(fetch, raw_base, docs_root)`|`_slice`(marqueurs start/end), `_convert_lines`(parcours principal), `_directive`.|
|**`build_document`**|Produit un document complet : conversion, suppression de la navigation, en-tête,|
|`(name, public_url, source_url,` `fetch) → str`|nettoyage.|
|**`main`** `()`|Construit les 10 documents de PAGESdans `corpus/`.|

## 4.4 Choix et alternatives

> [!TIP]
> **POURQUOI PARTIR DES SOURCES PLUTÔT QUE DU SITE**
> La V1 téléchargeait le HTML rendu des sites, en extrayait le texte avec BeautifulSoup, puis générait des PDF avec ReportLab, que le chunking relisait avec pypdf. Trois défauts : chaque élément en ligne (lien, `code`) devenait une ligne à part et coupait les phrases ; des artefacts d'interface (« ↑↓ Navigate ■ Select Esc Close ») se retrouvaient dans le texte ; la structure en sections était perdue, ce qui interdisait un chunking structurel. Le passage par PDF n'apportait rien : on écrivait un format de mise en page pour le relire aussitôt. Les sources donnent un texte propre, structuré et versionné ; figer les versions rend le corpus reproductible, ce que la CI vérifie à chaque push en le reconstruisant et en exigeant un `git diff` vide.

> [!NOTE]
> **CE QU'ON AURAIT PU FAIRE**
>
> - **Construire les sites avec leurs propres outils** (Docusaurus pour dbt, Sphinx pour Airflow) et convertir le HTML produit : fidèle à 100 %, mais il faut installer deux chaînes de build complètes. Les convertisseurs maison couvrent les constructions effectivement présentes dans ces 10 pages, et les tests vérifient chacune d'elles.
>
> - **Pandoc** pour le RST : convertisseur générique et robuste, mais il ne connaît pas les directives propres à Sphinx et Airflow (`exampleinclude`, rôles `:class:`) et ne résout pas les inclusions.
>
> - **Unstructured, Docling ou un parseur de PDF avancé** : utiles quand on n'a que des PDF ou des documents bureautiques. Ici, on a mieux : la source.
>
> - **Garder le HTML et extraire avec trafilatura** : plus simple, mais dépend de la mise en page du site, qui change sans prévenir.

> [!WARNING]
> **ATTENTION**
>
> Les convertisseurs sont spécialisés : une page qui utiliserait un composant MDX ou une directive RST non prévus laisserait passer des balises brutes. Ajouter une page au corpus demande de relire le Markdown produit.
