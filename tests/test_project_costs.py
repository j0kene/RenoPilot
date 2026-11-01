import csv
import io
from datetime import date

from renopilot.nodes.respond import render, shopping_list_csv
from renopilot.schemas import PriceQuote, ProjectProfile, Suggestion, Verdict, WorkEstimate


def furniture():
    return Suggestion(request_id="chairs-living", product_id="ikea:chair", item_type="dining_chair",
                      work_type="loose_furniture", status="approved", title="Dining chair", quantity=4,
                      source="ikea_sg",
                      room_label="Living room", price_quote=PriceQuote(listed=100, currency="SGD", item_sgd=100,
                          local_delivery_sgd=35, total_sgd=135, lines=[]), retrieved_at="2026-10-30",
                      fit_note="Fits within the supplied width limit.")


def floor_work(status="proposed"):
    return Suggestion(request_id="floor-living", item_type="floor_tiles", work_type="finish", title="Floor tile work",
                      status=status, room_label="Living room", verdict=Verdict(decision="allow_with_permit",
                      reason="Confirm the existing layers.", route="Overlay", rule_id="floor", citation="HDB",
                      conditions=["Confirm the existing floor layers before work."]))


def rows(state):
    return list(csv.reader(io.StringIO(shopping_list_csv(state))))


def test_quantity_room_and_shared_delivery_totals_match_screen_and_csv():
    state = {"approved": True, "suggestions": [furniture()], "project": ProjectProfile()}
    text = render(state)
    table = rows(state)
    basket_row = next(row for row in table if row and row[0] == "BASKET TOTAL")
    chair = next(row for row in table if row and row[2] == "Dining chair")
    assert "Dining chair" in text and "Living room" in text and "× 4" in text
    assert "Selected basket total: S$435.00" in text
    assert basket_row[8] == "435.0" and chair[12:16] == ["4", "chairs-living", "2026-10-30", "Fits within the supplied width limit."]


def test_unknown_work_cost_keeps_project_estimate_incomplete_and_conditions_visible():
    work = floor_work()
    project = ProjectProfile(work_estimates=[WorkEstimate(request_id="floor-living", description="Floor tile work",
                                   )])
    state = {"approved": True, "suggestions": [furniture(), work], "project": project}
    text = render(state)
    table = rows(state)
    work_row = next(row for row in table if row and row[0] == "WORK ESTIMATE")
    works_total = next(row for row in table if row and row[0] == "WORKS SUBTOTAL")
    project_total = next(row for row in table if row and row[0] == "PROJECT ESTIMATE")
    assert "amount unknown" in text and "source unknown, check date unknown" in text
    assert "not cleared to proceed" in text and "Confirm the existing floor layers" in text
    assert work_row[4] == work_row[8] == work_row[10] == work_row[14] == ""
    assert works_total[8] == project_total[8] == ""
    assert "incomplete" in project_total[2].lower()


def test_verified_work_estimate_is_included_in_project_total():
    work = floor_work(status="approved")
    project = ProjectProfile(work_estimates=[WorkEstimate(request_id="floor-living", description="Tile install",
                                   quantity=8, unit="m2", unit_rate_sgd=10, source="Contractor quote",
                                   checked_at=date(2026, 10, 30))])
    state = {"approved": True, "suggestions": [furniture(), work], "project": project}
    text = render(state)
    table = rows(state)
    works_total = next(row for row in table if row and row[0] == "WORKS SUBTOTAL")
    project_total = next(row for row in table if row and row[0] == "PROJECT ESTIMATE")
    assert works_total[8] == "80.0" and project_total[8] == "515.0"
    assert "Project estimate: S$515.00" in text
