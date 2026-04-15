from __future__ import annotations


LOT_SUBITEM_TO_COLUMN = {
    "avg": "lot_avg_fab_value",
    "std": "lot_std_fab_value",
    "min": "lot_min_fab_value",
    "max": "lot_max_fab_value",
    "range": "lot_range_fab_value",
}

JUDGE_PRIORITY = {"normal": 0, "BOB": 1, "WOW": 2}
CONTROLLED_JUDGE_PRIORITY = {"normal": 0, "BOB": 1, "DPM_Controlled": 2}


def safe_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
