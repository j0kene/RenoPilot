"""Hard filters must hold whatever the rankings say. Uses a tiny synthetic catalog and
fake embeddings, so no model is loaded."""
import numpy as np

from renopilot.catalog.search import CatalogIndex
from renopilot.schemas import Product


def _index():
    rows = [
        ("a", "sofa", "Oak frame linen sofa beige", 1200.0, "ikea_sg"),
        ("b", "sofa", "Velvet chesterfield sofa green", 2400.0, "castlery"),
        ("c", "sofa", "Linen sofa light oak legs", None, "abo"),
        ("d", "coffee_table", "Round oak coffee table", 300.0, "ikea_sg"),
        ("e", "sofa", "Boucle curved sofa cream", 900.0, "castlery"),
    ]
    products = [Product(id=i, item_type=t, title=title, price_sgd=p, source=s) for i, t, title, p, s in rows]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(rows), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return CatalogIndex(products, emb, "fake:none"), emb


def test_item_type_and_price_cap_are_never_violated():
    idx, _ = _index()
    hits = idx.search("linen sofa", item_type="sofa", max_price=1500, signals=("bm25",))
    ids = [h.product.id for h in hits]
    assert set(ids) == {"a", "e"}                     # b over cap, c unknown price, d wrong type
    assert all(h.product.price_sgd <= 1500 for h in hits)


def test_no_cap_keeps_unpriced_items():
    idx, _ = _index()
    ids = {h.product.id for h in idx.search("linen sofa", item_type="sofa", signals=("bm25",))}
    assert ids == {"a", "b", "c", "e"}


def test_exclude_and_sources():
    idx, _ = _index()
    hits = idx.search("sofa", item_type="sofa", exclude_ids={"a"}, sources=("ikea_sg", "castlery"),
                      signals=("bm25",))
    assert {h.product.id for h in hits} == {"b", "e"}


def test_board_image_signal_ranks_the_matching_product_first():
    idx, emb = _index()
    hits = idx.search("", item_type="sofa", board_emb=emb[[4]], signals=("board_img",))
    assert hits[0].product.id == "e"
    assert hits[0].ranks == {"board_img": 1}


def test_empty_filter_result_is_empty_not_an_error():
    idx, _ = _index()
    assert idx.search("sofa", item_type="sofa", max_price=10, signals=("bm25",)) == []
    assert idx.count(item_type="sofa", max_price=1000) == 1
