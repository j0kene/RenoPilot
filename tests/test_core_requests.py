"""Request identity, quantity-aware costs and cross-room catalog constraints."""
from datetime import date
import csv
import io

import numpy as np
import pytest

from renopilot.basket import offers_for, selection, summary, work_totals
from renopilot.catalog.search import CatalogIndex
from renopilot.nodes.approval import _apply_work_updates, apply_routes
from renopilot.nodes import approval as approval_node
from renopilot.nodes.guard import compliance_guard, request_from
from renopilot.nodes.respond import shopping_list_csv
from renopilot.nodes.style import build_shortlist, identify_requests, make_shortlist
from renopilot.pricing.fx import FxProvider
from renopilot.pricing.landed_cost import landed_cost
from renopilot.pricing.node import make_pricing
from renopilot.schemas import ItemSpec, Product, ProjectProfile, StyleProfile, Suggestion, WorkEstimate
from renopilot.tools.budget import request_caps
from tests.test_pricing_node import FakeJudge


def profile(*items):
    return StyleProfile(style_tags=["japandi"], palette=["beige"], summary="Pale wood", items=list(items))


def test_room_requests_keep_independent_choices_and_quantity_aware_budget_and_delivery():
    p = Product(id="ikea_sg:sofa", source="ikea_sg", item_type="sofa", title="Linen sofa",
                price=100, price_sgd=100, currency="SGD", dims="W 120 cm D 80 cm H 75 cm")
    too_big = p.model_copy(update={"id": "ikea_sg:large", "dims": "W 240 cm D 100 cm H 90 cm"})
    requests = profile(ItemSpec(item_type="sofa", description="linen sofa", room="bedroom", room_label="Bedroom 1",
                                quantity=2, max_width_cm=180),
                       ItemSpec(item_type="sofa", description="linen sofa", room="bedroom", room_label="Bedroom 2",
                                max_width_cm=180))
    project = ProjectProfile(total_budget=1000)
    rows, _ = build_shortlist(requests, project, [], index=CatalogIndex([p, too_big], np.zeros((2, 4)), "none"),
                              signals=("bm25",), use_rerank=False)
    assert len(rows) == 2 and len({s.request_id for s in rows}) == 2
    assert all(s.product_id == p.id and s.fit_note.startswith("Fits:") for s in rows)
    assert all(cap == pytest.approx(1000 / 3) for cap in request_caps(project, rows).values())
    quote = landed_cost(p, fx=FxProvider(mode="snapshot"))
    checked = compliance_guard({"suggestions": [s.model_copy(update={"price_quote": quote}) for s in rows]})["suggestions"]
    chosen, total = selection(checked, {s.selection_key: p.id for s in checked}, project)
    assert total["total_sgd"] == summary(chosen)["total_sgd"] == 335
    assert len(offers_for(checked)) == 2


def test_comparable_rejection_and_size_caps_are_scoped_to_the_request():
    original = Product(id="castlery:sofa", source="castlery", item_type="sofa", title="Linen sofa",
                       price=500, price_sgd=500, currency="SGD", dims="W 150 cm D 80 cm H 75 cm")
    candidate = original.model_copy(update={"id": "ikea_sg:sofa", "source": "ikea_sg", "price": 100})
    requests = profile(*[ItemSpec(request_id=key, item_type="sofa", description="sofa", max_width_cm=180)
                         for key in ("a", "b")])
    rows = [Suggestion(request_id=key, product_id=original.id, item_type="sofa", work_type="loose_furniture")
            for key in ("a", "b")]
    pricing = make_pricing(products={p.id: p for p in (original, candidate)}, pool=[original, candidate],
        fx=FxProvider(mode="snapshot"), judge_model=FakeJudge([{"shortlisted_id": original.id,
            "candidate_id": candidate.id, "comparable": True, "why": "Similar linen sofa"}]))
    out = pricing({"suggestions": rows, "style_profile": requests, "project": ProjectProfile(),
                   "rejected": [{"request_id": "a", "product_id": candidate.id, "by": "user"}]})["suggestions"]
    assert not out[0].comparables and out[1].comparables[0].product_id == candidate.id
    too_big = requests.items[1].model_copy(update={"max_width_cm": 120})
    requests = requests.model_copy(update={"items": [requests.items[0], too_big]})
    out = pricing({"suggestions": rows, "style_profile": requests, "project": ProjectProfile()})["suggestions"]
    assert out[0].comparables and not out[1].comparables


def test_work_edits_and_guard_coverage_do_not_merge_two_bathrooms():
    a, b = [ItemSpec(request_id=key, item_type="floor_tiles", description="Floor tiles", room="bathroom",
                     room_label=key, work_details={"overlay_thickness_mm": 10, "existing_floor_layers": 1})
            for key in ("Bathroom 1", "Bathroom 2")]
    state = {"style_profile": profile(a, b), "suggestions": [request_from(a), request_from(b)]}
    changed, rows, _, _ = _apply_work_updates(state, {a.request_id: {"quantity": 2}})
    assert [r.quantity for r in rows] == [2, 1]
    guarded = compliance_guard({"style_profile": changed, "suggestions": rows[:1]})["suggestions"]
    assert {s.request_id for s in guarded} == {a.request_id, b.request_id}
    with pytest.raises(ValueError, match="separate request"):
        _apply_work_updates(state, {a.request_id: {"item_type": "sofa"}})


def test_work_budget_is_enforced_and_blocked_requests_remain_in_export():
    work = Suggestion(request_id="work", item_type="wall_paint", work_type="finish", title="Paint bedroom",
                      verdict={"decision": "allow"})
    project = ProjectProfile(works_budget=100, work_estimates=[WorkEstimate(request_id="work", description="Painting",
        quantity=2, unit_rate_sgd=60, source="Owner quote", checked_at=date(2026, 11, 1))])
    with pytest.raises(ValueError, match="works budget"):
        selection([work], {}, project)
    blocked = work.model_copy(update={
        "status": "rejected", "verdict": work.verdict.model_copy(update={"decision": "block"})})
    rows = list(csv.reader(io.StringIO(shopping_list_csv({"project": project, "suggestions": [blocked]}))))
    assert any(len(row) > 3 and row[2] == "Paint bedroom" and row[3] == "rejected" for row in rows)


def test_choosing_a_route_keeps_the_work_estimate_and_its_budget_check():
    board = identify_requests(profile(ItemSpec(item_type="floor_tiles", description="Floor tiles", room="living",
        work_details={"overlay_thickness_mm": 10, "existing_floor_layers": 1})))
    work = compliance_guard({"project": ProjectProfile(), "style_profile": board, "suggestions": []})["suggestions"][0]
    project = ProjectProfile(works_budget=1000, work_estimates=[WorkEstimate(request_id=work.key, description="Tiling",
        unit_rate_sgd=5000, source="Contractor", checked_at=date(2026, 10, 30))])
    routed = apply_routes([work], {work.key: "HDB-FLOOR-OVERLAY"}, project)
    assert work_totals(project, routed)["subtotal_sgd"] == 5000
    with pytest.raises(ValueError, match="works budget"):
        selection(routed, {}, project)


def test_request_with_no_offers_can_be_edited():
    a = ItemSpec(request_id="a", item_type="sofa", description="sofa", room="bedroom")
    b = a.model_copy(update={"request_id": "b"})
    state = {"style_profile": profile(a, b), "suggestions": [Suggestion(
        request_id="a", product_id="p", item_type="sofa", work_type="loose_furniture")]}
    changed, _, redo, _ = _apply_work_updates(state, {"b": {"max_width_cm": 200}})
    assert changed.items[1].max_width_cm == 200 and redo == ["sofa"]
    with pytest.raises(ValueError, match="no longer on this shortlist"):
        _apply_work_updates(state, {"gone": {"quantity": 2}})


def test_affordable_quantity_basket_survives_pricing_and_approval():
    chair = Product(id="ikea_sg:chair", source="ikea_sg", item_type="dining_chair", title="Chair", price=100,
                    price_sgd=100, currency="SGD", dims="W 50 cm D 50 cm H 80 cm", weight_kg=10)
    spec = ItemSpec(request_id="chairs", item_type="dining_chair", description="chair", quantity=4)
    pricing = make_pricing(products={chair.id: chair}, pool=[chair], fx=FxProvider(mode="snapshot"), judge_model=None)
    row = Suggestion(request_id="chairs", product_id=chair.id, item_type=chair.item_type, work_type="loose_furniture",
                     quantity=4, source=chair.source, title="Chair")
    for budget, kept in ((435, True), (434, False)):
        project = ProjectProfile(total_budget=budget)
        out = pricing({"suggestions": [row], "style_profile": profile(spec), "project": project})["suggestions"]
        assert (out[0].status != "rejected") is kept
    project = ProjectProfile(total_budget=435)
    out = pricing({"suggestions": [row], "style_profile": profile(spec), "project": project})["suggestions"]
    checked = compliance_guard({"suggestions": out, "project": project})["suggestions"]
    assert selection(checked, {"chairs": chair.id}, project)[1]["total_sgd"] == 435


def test_refinement_stays_on_the_rejected_request_and_out_of_the_saved_profile(monkeypatch):
    # This test checks request-scoped query text, not the live shopper.
    monkeypatch.setattr("renopilot.nodes.shopper.shop", lambda *args, **kwargs: [])
    class Index:
        embed_spec = "fake"
        queries = []

        def search(self, query, **_):
            self.queries.append(query)
            return []

    a = ItemSpec(request_id="bed-1", item_type="sofa", description="sofa one", room="bedroom")
    b = ItemSpec(request_id="bed-2", item_type="sofa", description="sofa two", room="bedroom")
    refinements = {"bed-1": "narrower"}
    out = make_shortlist(index=Index(), signals=("bm25",))({
        "style_profile": profile(a, b), "project": ProjectProfile(), "board_images": [], "shortlist_types": ["sofa"],
        "suggestions": [], "rejected": [], "refinements": refinements})
    assert ["narrower" in q for q in Index.queries] == [True, False]
    assert [i.description for i in out["style_profile"].items] == ["sofa one", "sofa two"]


def test_delivery_is_shared_when_a_category_is_split_across_rooms():
    chair = Product(id="ikea_sg:chair", source="ikea_sg", item_type="dining_chair", title="Chair", price=100,
                    price_sgd=100, currency="SGD", dims="W 50 cm D 50 cm H 80 cm", weight_kg=10)
    specs = [ItemSpec(request_id=r, item_type="dining_chair", description="chair", quantity=2) for r in ("a", "b")]
    rows = [Suggestion(request_id=r, product_id=chair.id, item_type=chair.item_type, work_type="loose_furniture",
                       quantity=2, source=chair.source, title="Chair") for r in ("a", "b")]
    pricing = make_pricing(products={chair.id: chair}, pool=[chair], fx=FxProvider(mode="snapshot"), judge_model=None)
    for budget, kept in ((435, True), (434, False)):
        project = ProjectProfile(total_budget=budget)
        out = pricing({"suggestions": rows, "style_profile": profile(*specs), "project": project})["suggestions"]
        assert all(s.status != "rejected" for s in out) is kept
    project = ProjectProfile(total_budget=435)
    out = pricing({"suggestions": rows, "style_profile": profile(*specs), "project": project})["suggestions"]
    checked = compliance_guard({"suggestions": out, "project": project})["suggestions"]
    assert selection(checked, {"a": chair.id, "b": chair.id}, project)[1]["total_sgd"] == 435


def test_delivery_tier_uses_combined_weight_for_multiple_units():
    chair = Product(id="ikea_sg:chair", source="ikea_sg", item_type="dining_chair", title="Chair",
                    price=100, currency="SGD", dims="W 50 cm D 50 cm H 80 cm", weight_kg=10)
    quote = landed_cost(chair, fx=FxProvider(mode="snapshot"))
    assert quote.local_delivery_sgd == 10
    picked = Suggestion(product_id=chair.id, item_type=chair.item_type, work_type="loose_furniture", source=chair.source,
                        quantity=4, price_quote=quote)
    assert summary([picked])["total_sgd"] == 435  # 40kg requires one home delivery


def test_quantity_edit_reallocates_other_categories_and_preserves_their_rejections(monkeypatch):
    sofa = ItemSpec(request_id="sofa", item_type="sofa", description="Sofa")
    rug = ItemSpec(request_id="rug", item_type="rug", description="Rug")
    state = {"project": ProjectProfile(total_budget=1000), "style_profile": profile(sofa, rug),
             "suggestions": [Suggestion(request_id=i.request_id, item_type=i.item_type, product_id=i.request_id,
                                        work_type="loose_furniture") for i in (sofa, rug)],
             "rejected": [{"request_id": "rug", "item_type": "rug", "product_id": "old-rug", "by": "user"}]}
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"work_updates": {"sofa": {"quantity": 2}}})
    command = approval_node.approval(state)
    assert set(command.update["redo_types"]) == {"sofa", "rug"}
    assert command.update["rejected"].value == state["rejected"]


def test_replacing_brief_clears_estimates_from_the_previous_scope(monkeypatch):
    project = ProjectProfile(work_estimates=[WorkEstimate(description="Old quote", unit_rate_sgd=100,
                            source="Contractor", checked_at=date(2026, 11, 1))])
    monkeypatch.setattr(approval_node, "interrupt", lambda _: {"brief": "A different room and scope"})
    command = approval_node.approval({"project": project, "suggestions": []})
    assert command.update["project"].work_estimates == []
