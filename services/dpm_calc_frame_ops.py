from __future__ import annotations

import pandas as pd


def build_lot_representative_df(incoming_df: pd.DataFrame) -> pd.DataFrame:
    if incoming_df.empty:
        return pd.DataFrame()
    if any(col not in incoming_df.columns for col in ["lot_id", "tkout_time"]):
        return pd.DataFrame()

    working_df = incoming_df.sort_values("tkout_time").copy()
    return working_df.groupby("lot_id", as_index=False).first().reset_index(drop=True)


def build_joined_wafer_df(incoming_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    if incoming_df.empty or target_df.empty:
        return pd.DataFrame()

    incoming_cols = [col for col in incoming_df.columns if col not in {"eqp_id", "eqp_chamber"}]
    return incoming_df[incoming_cols].copy().merge(
        target_df.copy(),
        on=["root_lot_id", "wafer_id"],
        how="inner",
        suffixes=("_incoming", "_target"),
    ).reset_index(drop=True)


def is_positive_correlation(joined_df: pd.DataFrame) -> bool:
    if joined_df.empty:
        return True
    if "fab_value_incoming" not in joined_df.columns or "fab_value_target" not in joined_df.columns:
        return True

    corr = pd.to_numeric(joined_df["fab_value_incoming"], errors="coerce").corr(
        pd.to_numeric(joined_df["fab_value_target"], errors="coerce")
    )
    return False if pd.notna(corr) and corr < 0 else True


def build_window_summary(
    target_df: pd.DataFrame,
    event_time,
    equipment_col: str,
    *,
    window_days: int,
    min_wafer_qty: int,
) -> pd.DataFrame:
    if target_df.empty or equipment_col not in target_df.columns:
        return pd.DataFrame()
    if pd.isna(event_time):
        return pd.DataFrame()

    end_time = pd.Timestamp(event_time)
    start_time = end_time - pd.Timedelta(days=window_days)
    window_df = target_df[
        (target_df["tkout_time"] >= start_time) & (target_df["tkout_time"] <= end_time)
    ].copy()
    if window_df.empty:
        return pd.DataFrame()

    summary_df = (
        window_df.groupby(equipment_col, dropna=False)
        .agg(
            fab_value_count=("fab_value", "count"),
            fab_value_med=("fab_value", "median"),
        )
        .reset_index()
    )
    summary_df = summary_df[summary_df["fab_value_count"] >= min_wafer_qty].reset_index(drop=True)
    summary_df["window_start_time"] = start_time
    summary_df["window_end_time"] = end_time
    return summary_df


def initialize_target_result_df(target_df: pd.DataFrame) -> pd.DataFrame:
    if target_df.empty:
        return pd.DataFrame()

    working_df = target_df.copy().reset_index(drop=False).rename(columns={"index": "_target_row_id"})
    working_df["dpm_judge"] = "normal"
    working_df["dpm_rank"] = pd.NA
    working_df["dpm_score"] = pd.NA
    working_df["dpm_controlled_value"] = working_df["fab_value"]
    working_df["dpm_controlled_judge"] = "normal"
    return working_df


def summarize_box_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["series", "count", "avg", "med", "std"])

    summary_rows = []
    for series_name, value_col in [("Target", "fab_value"), ("DPM Controlled", "dpm_controlled_value")]:
        if value_col not in df.columns:
            continue
        values = pd.to_numeric(df[value_col], errors="coerce").dropna()
        summary_rows.append(
            {
                "series": series_name,
                "count": int(values.count()),
                "avg": float(values.mean()) if not values.empty else None,
                "med": float(values.median()) if not values.empty else None,
                "std": float(values.std()) if not values.empty else None,
            }
        )
    return pd.DataFrame(summary_rows)
