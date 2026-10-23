from renopilot.schemas import ProjectProfile, Suggestion
from renopilot.tools.budget import allocate, caps_for, fmt_sgd, total, within


def test_allocate_is_deterministic_and_never_overspends():
    caps = allocate(8000, ["sofa", "coffee_table", "rug"])
    assert caps == allocate(8000, ["sofa", "coffee_table", "rug"])
    assert sum(caps.values()) <= 8000
    assert caps["sofa"] > caps["coffee_table"] > 0


def test_allocate_dedupes_and_handles_empty():
    assert allocate(1000, []) == {}
    assert allocate(0, ["sofa"]) == {}
    assert list(allocate(1000, ["sofa", "sofa"])) == ["sofa"]


def test_category_budget_wins_and_total_covers_the_rest():
    p = ProjectProfile(total_budget=5000, budget_by_category={"sofa": 2000})
    caps = caps_for(p, ["sofa", "coffee_table", "rug"])
    assert caps["sofa"] == 2000
    assert caps["coffee_table"] + caps["rug"] <= 3000


def test_no_budget_means_no_cap():
    assert caps_for(ProjectProfile(), ["sofa"]) == {"sofa": None}
    assert caps_for(None, ["sofa"]) == {"sofa": None}


def test_overspent_categories_leave_zero_not_negative():
    p = ProjectProfile(total_budget=1000, budget_by_category={"sofa": 1500})
    assert caps_for(p, ["sofa", "rug"])["rug"] == 0


def test_within_is_a_hard_rule():
    assert within(1500, 1500)
    assert not within(1500.01, 1500)
    assert not within(None, 1500)          # unknown price never passes a real cap
    assert within(None, None)


def test_total_and_format():
    s = [Suggestion(item_type="sofa", work_type="loose_furniture", price_sgd=1299.0, status="approved"),
         Suggestion(item_type="rug", work_type="loose_furniture", price_sgd=89.9, status="rejected")]
    assert total(s) == 1299.0
    assert total(s, status=None) == 1388.9
    assert fmt_sgd(1299) == "S$1,299" and fmt_sgd(89.9) == "S$89.90" and fmt_sgd(None) == "price n/a"
