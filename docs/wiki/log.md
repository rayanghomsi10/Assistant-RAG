# Journal du wiki

Une entrée par évolution, la plus récente en bas. Format : `## [AAAA-MM-JJ] type | sujet`, où
type vaut `migration`, `mise-a-jour`, `resultat` ou `verification`.

## [2026-10-18] historique | Contenu repris du PDF

Avant la migration, la documentation était un PDF retouché à la main. Ses derniers ajouts :
chapitre 19 (baseline V2 mesurée les 12 et 13 octobre 2026), chapitre 20 (Langfuse, interface,
mémoire de conversation, compléments, reprise de `build_preferences.py`) et section 20.6
(révision de la sélection des paires DPO, second run du 18 octobre 2026).

## [2026-11-06] migration | PDF vers wiki Markdown

Le PDF de 69 pages devient un wiki Markdown (`docs/wiki/`), source unique de la documentation ; le
PDF est désormais généré par `docs/build_pdf.py`. Conversion automatique (texte, tableaux,
encadrés, code en ligne, figures recadrées dans `assets/`), puis relecture page par page.
Corrections faites à cette occasion : arborescence du dépôt (3.3) et variables de configuration
(3.5) mises à jour ; nombre de tests (66, chapitre 12) et de lignes par module ; signatures de
`chain.py` (8.2) ; route `GET /` (9.1) ; mode d'emploi (18) réécrit, avec les commandes de
relecture des paires et de génération du PDF ; restes de texte masqué par les retouches du PDF
supprimés (12, 16).
