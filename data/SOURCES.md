# Data sources

All grounding data is a dated snapshot, frozen before evaluation. Large files (`raw/`, `images/`,
`index/`) are rebuilt by the scripts in `scripts/`; the small curated files are committed.

| Data | Source | Licence / terms | Snapshot | Script | Used by |
|---|---|---|---|---|---|
| Product catalog for retrieval (furniture subset of ABO) | Amazon Berkeley Objects, public S3 bucket `amazon-berkeley-objects` | CC BY 4.0 (credit: Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022) | 2026-10-22 | `fetch_abo.py`, `build_abo_subset.py` | retrieval eval, catalog |
| SG price catalog (`sg_catalog.csv`) | IKEA Singapore (public search API behind ikea.com/sg; robots.txt on that host allows all), Castlery SG category pages (JSON-LD; robots.txt allows) | Prices, titles and links quoted for a private student project; images cached locally only to compute embeddings, not redistributed | 2026-10-22 | `collect_sg_catalog.py` | Style agent demo, budget filter, Pricing |
| Taobao items (`taobao_manual.csv`) | Taobao listings, entered by hand (login wall) | as above | pending | manual | Pricing |
| Style eval mood boards (`boards/`) | Unsplash, official API | Unsplash licence; photographer attribution in each board's `meta.json`; private eval set, not redistributed | pending (needs API key) | `fetch_unsplash_boards.py` | style eval |
| Retrieval eval boards (`evals/cases/retrieval.jsonl`) | built from the ABO subset (alternate images) | CC BY 4.0 | with ABO | `make_retrieval_boards.py` | retrieval eval |

## Known data issues (and what we did)

- **Leaky retailer categories.** Castlery's floor-lamp page lists table lamps, and IKEA's "side table" search
  returns bedside tables. Every product is re-typed from its own title (`renopilot/catalog/taxonomy.py`);
  each mistype we found is a test case in `tests/test_taxonomy.py`.
- **ABO accessories labelled as furniture** (fabric swatches under `SOFA`, slipcovers, replacement legs) are
  excluded by title pattern. Colour variants of one model are collapsed to one family.
- **IKEA product names arrive with U+FFFD** in place of Swedish letters (e.g. `PO�NG`): the API sends them
  that way. Harmless for search; the reranker strips non-ASCII.
- **CNY to SGD** for Taobao rows uses a fixed indicative rate (`CNY_TO_SGD` in the collector), and only for the
  budget filter. Landed cost (FX date, shipping, GST) is the Pricing agent's job.
- Castlery had no matching category page for bar stools, ottomans and bed frames in this snapshot.
