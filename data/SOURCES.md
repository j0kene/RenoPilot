# Data sources

All grounding data is a dated snapshot, frozen before evaluation. Large files (`raw/`, `images/`,
`index/`) are rebuilt by the scripts in `scripts/`; the small curated files are committed.

| Data | Source | Licence / terms | Snapshot | Script | Used by |
|---|---|---|---|---|---|
| Product catalog for retrieval (furniture subset of ABO) | Amazon Berkeley Objects, public S3 bucket `amazon-berkeley-objects` | CC BY 4.0 (credit: Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022) | 2026-10-22 | `fetch_abo.py`, `build_abo_subset.py` | retrieval eval, catalog |
| SG price catalog (`sg_catalog.csv`) | IKEA Singapore (public search API behind ikea.com/sg; robots.txt on that host allows all), Castlery SG category pages (JSON-LD; robots.txt allows) | Prices, titles and links quoted for a private student project; images cached locally only to compute embeddings, not redistributed | 2026-10-22 | `collect_sg_catalog.py` | Style agent demo, budget filter, Pricing |
| Overseas listings (`taobao_manual.csv`) | **18 made-up illustrations** of the overseas landed-cost method (`synthetic=true`, `[SAMPLE]` titles, no URLs), not observed prices. Taobao sits behind a login; real rows can be entered by hand in the same columns | n/a (synthetic) | 2026-10-25 | manual | Pricing |
| Pricing rules (`pricing/rules.json`) | IRAS (GST 9% on imports), Singapore Customs (low-value goods / OVR, S$400 air/post import relief, no relief by sea, dutiable goods), IKEA SG and Castlery delivery pages; Cainiao sea tariff via PropertyGuru (2020, **assumption**); air rate (**assumption**) | public pages; each entry carries its URL and date checked | 2026-10-25 | hand-compiled | Pricing |
| FX snapshot (`pricing/fx_snapshot.json`) | Frankfurter API (ECB reference rates) | free, no key | rates of 2026-10-23, fetched 2026-10-25 | `fx.py` | Pricing evals |
| HDB renovation rule cards (`compliance/hdb_rules.jsonl`) | HDB Renovation Guidelines for Building Works, Important Information on Renovations, Application for a Renovation Permit (hdb.gov.sg); MyNiceHome (gov.sg); 8 contractor/blog guides as **secondary** sources (`compliance/sources.json` lists each URL, status and whether it is official) | public pages, quoted for a private student project; raw pages not redistributed | 2026-10-26 | `fetch_hdb_sources.py`; cards hand-written, quotes checked by `tests/test_compliance_kb.py` | Compliance guard |
| Style eval mood boards (`boards/`) | Unsplash, official API | Unsplash licence; photographer attribution in each board's `meta.json`; private eval set, not redistributed | 2026-10-22/23 (27 boards x 4 photos) | `fetch_unsplash_boards.py` | style eval |
| Retrieval eval boards (`evals/cases/retrieval.jsonl`) | built from the ABO subset (alternate images) | CC BY 4.0 | with ABO | `make_retrieval_boards.py` | retrieval eval |

## Known data issues (and what we did)

- **Leaky retailer categories.** Castlery's floor-lamp page lists table lamps, and IKEA's "side table" search
  returns bedside tables. Every product is re-typed from its own title (`renopilot/catalog/taxonomy.py`);
  each mistype we found is a test case in `tests/test_taxonomy.py`.
- **ABO accessories labelled as furniture** (fabric swatches under `SOFA`, slipcovers, replacement legs) are
  excluded by title pattern. Colour variants of one model are collapsed to one family.
- **IKEA product names arrive with U+FFFD** in place of Swedish letters (e.g. `PO�NG`): the API sends them
  that way. Harmless for search; the reranker strips non-ASCII.
- **CNY to SGD** is never hard-coded: overseas rows are priced by the Pricing agent with a dated FX rate
  (live Frankfurter or the frozen snapshot), plus shipping and GST.
- Castlery had no matching category page for bar stools, ottomans and bed frames in this snapshot.
- **Delivery fees are per order**, not per item. A quote shows the fee "if bought alone"; totals across items
  (`basket_total`) charge each local retailer's fee once (Castlery's tier is re-read on the order total).
- **IKEA rows have no weights**, so the IKEA delivery tier (S$10 parcel vs S$35 home) uses a typical weight per
  item type and the quote is marked estimated.
- **Overseas shipping rates are the weakest data**: no authoritative current rate was found; quotes using the
  estimate list it as an assumption, and a seller-quoted fee on the row always wins.
