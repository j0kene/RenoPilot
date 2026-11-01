from renopilot.catalog.fit import axis_dimensions_cm, canonical_dimensions, fit_note
from renopilot.schemas import ItemSpec, Product


def product(dims):
    return Product(id="p", source="ikea_sg", item_type="desk", title="Desk", dims=dims)


def spec(**limits):
    return ItemSpec(item_type="desk", description="desk", **limits)


def test_axis_dimensions_convert_cm_mm_and_inches():
    assert axis_dimensions_cm('W 80 cm D: 500 mm H=28"') == {"width": 80, "depth": 50, "height": 71.12}
    assert axis_dimensions_cm("Width 31.5 inches Depth 20 in Height 28 in") == {
        "width": 80.01, "depth": 50.8, "height": 71.12}
    assert canonical_dimensions("Width: 800 mm, Depth: 50 cm") == "W 80 cm D 50 cm"


def test_bare_product_dimensions_do_not_guess_axis_order():
    fit, note = fit_note(product("90x55 cm"), spec(max_width_cm=100, max_depth_cm=60))
    assert fit is None and "width, depth" in note
    assert axis_dimensions_cm("W x D x H: 90 x 55 x 75 cm") == {
        "width": 90, "depth": 55, "height": 75}
    assert axis_dimensions_cm("90 x 55 x 75 cm (W x D x H)") == {
        "width": 90, "depth": 55, "height": 75}


def test_missing_or_partial_axis_is_unknown_and_no_limits_is_not_checked():
    fit, note = fit_note(product("W 80 cm D 50 cm"), spec(max_width_cm=100, max_depth_cm=60, max_height_cm=75))
    assert fit is None and "height" in note
    assert fit_note(product(""), spec()) == (None, "Fit not checked: no size limits supplied.")


def test_over_limit_rejects_even_if_another_constrained_axis_is_missing():
    fit, note = fit_note(product("W 110 cm"), spec(max_width_cm=100, max_depth_cm=60))
    assert fit is False and "width is 110 cm" in note


def test_true_requires_every_constrained_axis_to_be_verified():
    assert fit_note(product("W 80 cm D 50 cm H 70 cm"),
                    spec(max_width_cm=80, max_depth_cm=50, max_height_cm=70))[0] is True
