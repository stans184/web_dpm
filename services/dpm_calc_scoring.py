from __future__ import annotations

import math

import pandas as pd

from services.dpm_calc_common import CONTROLLED_JUDGE_PRIORITY, JUDGE_PRIORITY, safe_float


def resolve_wow_count(total_count: int, wow_portion) -> int:
    if total_count <= 0:
        return 0

    wow_portion_value = safe_float(wow_portion)
    if wow_portion_value is None or wow_portion_value <= 0:
        return 0

    if wow_portion_value <= 1:
        wow_count = math.ceil(total_count * wow_portion_value)
    elif wow_portion_value <= 100:
        wow_count = math.ceil(total_count * (wow_portion_value / 100.0))
    else:
        wow_count = math.ceil(wow_portion_value)

    return min(total_count, max(0, wow_count))


def resolve_thresholds(
    df: pd.DataFrame,
    value_col: str,
    *,
    control_type: str,
    sigma,
    lsl,
    usl,
    percentile,
) -> tuple[float | None, float | None]:
    if df.empty or value_col not in df.columns:
        return None, None

    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        return None, None

    if control_type == "sigma":
        sigma_value = safe_float(sigma)
        if sigma_value is None:
            return None, None

        avg = values.mean()
        std = values.std()
        if pd.isna(std):
            return None, None
        return float(avg - sigma_value * std), float(avg + sigma_value * std)

    if control_type == "spec":
        return safe_float(lsl), safe_float(usl)

    percentile_value = safe_float(percentile)
    if percentile_value is None:
        return None, None
    percentile_value = max(0.0, min(100.0, percentile_value))
    return float(values.quantile(percentile_value / 100.0)), None


def classify_outliers(
    df: pd.DataFrame,
    value_col: str,
    *,
    control_type: str,
    sigma,
    lsl,
    usl,
    percentile,
) -> tuple[pd.DataFrame, dict]:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None), {
            "value_col": value_col,
            "lower_spec": None,
            "upper_spec": None,
            "control_type": control_type,
        }

    working_df = df.copy()
    numeric_values = pd.to_numeric(working_df[value_col], errors="coerce")
    lower_spec, upper_spec = resolve_thresholds(
        working_df,
        value_col,
        control_type=control_type,
        sigma=sigma,
        lsl=lsl,
        usl=usl,
        percentile=percentile,
    )

    status = pd.Series("normal", index=working_df.index)
    if lower_spec is not None:
        status = status.mask(numeric_values < lower_spec, "lower_out")
    if upper_spec is not None:
        status = status.mask(numeric_values > upper_spec, "upper_out")

    working_df["outlier_status"] = status
    return working_df.reset_index(drop=True), {
        "value_col": value_col,
        "lower_spec": lower_spec,
        "upper_spec": upper_spec,
        "control_type": control_type,
    }


def score_window_summary(
    summary_df: pd.DataFrame,
    *,
    incoming_value: float,
    incoming_avg: float,
    target_avg: float,
    positive_correlation: bool,
    use_absolute_formula: bool,
    use_ascending_rank: bool,
    wow_portion,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "fab_value_count",
                "fab_value_med",
                "window_start_time",
                "window_end_time",
                "dpm_score",
                "dpm_rank",
                "dpm_judge",
            ]
        )

    working_df = summary_df.copy()
    b_value = 1.0 if target_avg == 0 else incoming_avg / target_avg
    c_value = -2 * incoming_avg if positive_correlation else 0.0
    score = (1.0 * incoming_value) + (b_value * working_df["fab_value_med"].astype(float)) + c_value
    if use_absolute_formula:
        score = score.abs()

    working_df["dpm_score"] = score
    working_df["dpm_rank"] = working_df["dpm_score"].rank(
        method="dense",
        ascending=use_ascending_rank,
    ).astype(int)
    working_df["dpm_judge"] = "BOB"

    if len(working_df) < 3:
        return working_df.reset_index(drop=True)

    wow_count = resolve_wow_count(len(working_df), wow_portion)
    if wow_count <= 0:
        return working_df.reset_index(drop=True)

    worst_rows = working_df.sort_values(by=["dpm_rank", "dpm_score"], ascending=[False, False]).head(wow_count)
    working_df.loc[worst_rows.index, "dpm_judge"] = "WOW"
    return working_df.reset_index(drop=True)


def apply_equipment_judge_to_target_rows(
    target_rows: pd.DataFrame,
    equipment_summary_df: pd.DataFrame,
    *,
    equipment_col: str,
    correction_value: float,
) -> pd.DataFrame:
    working_df = target_rows.copy()
    if working_df.empty:
        return working_df

    if equipment_summary_df.empty or equipment_col not in working_df.columns:
        return working_df

    judge_map = equipment_summary_df.set_index(equipment_col)["dpm_judge"].to_dict()
    rank_map = equipment_summary_df.set_index(equipment_col)["dpm_rank"].to_dict()
    score_map = equipment_summary_df.set_index(equipment_col)["dpm_score"].to_dict()

    working_df["dpm_judge"] = working_df[equipment_col].map(judge_map).fillna("normal")
    working_df["dpm_rank"] = working_df[equipment_col].map(rank_map)
    working_df["dpm_score"] = working_df[equipment_col].map(score_map)
    working_df["dpm_controlled_judge"] = working_df["dpm_judge"].replace({"WOW": "DPM_Controlled"})

    wow_mask = working_df["dpm_judge"] == "WOW"
    if correction_value and wow_mask.any():
        working_df.loc[wow_mask, "dpm_controlled_value"] = working_df.loc[wow_mask, "fab_value"] - correction_value

    return working_df


def select_highest_priority(series: pd.Series, priority_map: dict[str, int]) -> str:
    labels = [label for label in series.dropna().astype(str).tolist() if label in priority_map]
    if not labels:
        return "normal"
    return max(labels, key=lambda label: priority_map[label])


def merge_updates_into_target(base_target_df: pd.DataFrame, updates_df: pd.DataFrame) -> pd.DataFrame:
    if base_target_df.empty:
        return pd.DataFrame()
    if updates_df.empty:
        return base_target_df

    agg_map = {
        "dpm_judge": ("dpm_judge", lambda s: select_highest_priority(s, JUDGE_PRIORITY)),
        "dpm_controlled_judge": (
            "dpm_controlled_judge",
            lambda s: select_highest_priority(s, CONTROLLED_JUDGE_PRIORITY),
        ),
        "dpm_rank": ("dpm_rank", "min"),
        "dpm_score": ("dpm_score", "mean"),
        "dpm_controlled_value": ("dpm_controlled_value", "mean"),
        "lot_control_value": ("lot_control_value", "mean"),
        "outlier_status": ("outlier_status", "first"),
        "control_group": ("control_group", "first"),
        "control_entity": ("control_entity", "first"),
        "source_wafer_id": ("source_wafer_id", "first"),
        "incoming_wafer_value": ("incoming_wafer_value", "mean"),
    }
    agg_map = {key: value for key, value in agg_map.items() if value[0] in updates_df.columns}
    aggregated_df = updates_df.groupby("_target_row_id", as_index=False).agg(**agg_map)

    merged_df = base_target_df.merge(
        aggregated_df,
        on="_target_row_id",
        how="left",
        suffixes=("", "_upd"),
    )

    for col in [
        "dpm_judge",
        "dpm_controlled_judge",
        "dpm_rank",
        "dpm_score",
        "dpm_controlled_value",
        "lot_control_value",
        "outlier_status",
        "control_group",
        "control_entity",
        "source_wafer_id",
        "incoming_wafer_value",
    ]:
        update_col = f"{col}_upd"
        if update_col in merged_df.columns:
            merged_df[col] = merged_df[update_col].combine_first(merged_df.get(col))
            merged_df = merged_df.drop(columns=[update_col])

    return merged_df
