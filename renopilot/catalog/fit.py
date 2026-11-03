"""Conservative checks against owner-supplied furniture size limits."""
from __future__ import annotations

import re

from renopilot.schemas import ItemSpec, Product

_AXIS = r"(?:width|depth|height|w|d|h)"
_UNIT = r"cm|mm|in(?:ch(?:es)?)?|\"|″"


def axis_dimensions_cm(dims: str) -> dict[str, float]:
    """Parse only explicit axis labels with explicit units; never assign bare WxD values."""
    dims = dims or ""
    unit = rf"(?P<unit>{_UNIT})(?![a-z])"
    number = r"(\d+(?:\.\d+)?)"
    descriptor = r"W\s*[x×]\s*D\s*[x×]\s*H"
    patterns = (
        rf"{descriptor}\s*[:=]\s*{number}\s*[x×]\s*{number}\s*[x×]\s*{number}\s*{unit}",
        rf"{number}\s*[x×]\s*{number}\s*[x×]\s*{number}\s*{unit}\s*\(?\s*{descriptor}\s*\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, dims, re.I)
        if match:
            factor = _cm_factor(match["unit"])
            return {axis: float(match[i]) * factor for i, axis in enumerate(("width", "depth", "height"), 1)}

    found = {}
    for match in re.finditer(
        rf"(?<![a-z])(?P<axis>{_AXIS})\s*[:=]?\s*(?P<value>\d+(?:\.\d+)?)\s*{unit}", dims, re.I,
    ):
        axis = {"w": "width", "width": "width", "d": "depth", "depth": "depth",
                "h": "height", "height": "height"}[match["axis"].lower()]
        value = float(match["value"])
        found[axis] = value * _cm_factor(match["unit"])
    return found


def _cm_factor(unit: str) -> float:
    unit = unit.lower()
    return 0.1 if unit == "mm" else 2.54 if unit.startswith("in") or unit in {'"', "″"} else 1


def canonical_dimensions(dims: str) -> str:
    """Normalize explicitly labelled dimensions for the catalog, retaining only known axes."""
    values = axis_dimensions_cm(dims)
    return " ".join(f"{axis[0].upper()} {values[axis]:g} cm" for axis in ("width", "depth", "height")
                    if axis in values)


def fit_note(product: Product, spec: ItemSpec) -> tuple[bool | None, str]:
    limits = {"width": spec.max_width_cm, "depth": spec.max_depth_cm, "height": spec.max_height_cm}
    limits = {axis: cap for axis, cap in limits.items() if cap is not None}
    if not limits:
        return None, "Fit not checked: no size limits supplied."

    dimensions = axis_dimensions_cm(product.dims)
    exceeded = [axis for axis, cap in limits.items() if axis in dimensions and dimensions[axis] > cap]
    if exceeded:
        axis = exceeded[0]
        return False, f"Does not fit: verified {axis} is {dimensions[axis]:g} cm; limit is {limits[axis]:g} cm."

    missing = [axis for axis in limits if axis not in dimensions]
    if missing:
        names = ", ".join(missing)
        return None, f"Fit unknown: verified axis dimensions missing for {names}."
    return True, "Fits: all constrained axes have verified dimensions within the supplied limits."
