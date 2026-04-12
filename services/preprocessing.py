"""DPM 계산 전처리 로직 모음."""

import pandas as pd


def remove_fab_value_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """fab_value 기준 3배 IQR 바깥 값을 제거한다."""
    if df is None or df.empty or "fab_value" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()

    working_df = df.copy()
    q1 = working_df["fab_value"].quantile(0.25)
    q3 = working_df["fab_value"].quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return working_df.reset_index(drop=True)

    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    filtered_df = working_df[working_df["fab_value"].between(lower_bound, upper_bound, inclusive="both")]
    return filtered_df.reset_index(drop=True)


def preprocess_comparison_trend_data(
    incoming_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """비교 산점도용 incoming/target 데이터를 정리한다."""
    return remove_fab_value_outliers_iqr(incoming_df), remove_fab_value_outliers_iqr(target_df)


def filter_trend_data(
    df: pd.DataFrame,
    *,
    ppid: str = "",
    lot_type: str = "",
    lot_filter: str = "",
) -> pd.DataFrame:
    """화면 입력 조건으로 추세 데이터를 필터링한다."""
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    working_df = df.copy()

    if ppid and "ppid" in working_df.columns:
        working_df = working_df[working_df["ppid"].astype(str).str.contains(ppid, case=False, na=False)]

    normalized_lot_type = str(lot_type).strip().lower()
    if normalized_lot_type == "all":
        normalized_lot_type = ""
    elif normalized_lot_type == "p type":
        normalized_lot_type = "p_type"

    if normalized_lot_type and "lot_type" in working_df.columns:
        working_df = working_df[
            working_df["lot_type"].astype(str).str.contains(normalized_lot_type, case=False, na=False)
        ]

    if lot_filter:
        # 사용자는 lot_id와 root_lot_id 둘 중 어느 값으로도 검색할 수 있다.
        lot_mask = pd.Series(False, index=working_df.index)
        for col in ["lot_id", "root_lot_id"]:
            if col in working_df.columns:
                lot_mask = lot_mask | working_df[col].astype(str).str.contains(
                    lot_filter,
                    case=False,
                    na=False,
                )
        working_df = working_df[lot_mask]

    return working_df.reset_index(drop=True)


def add_lot_subitem_values(df: pd.DataFrame) -> pd.DataFrame:
    """lot 단위 집계 컬럼을 추가한다."""
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    working_df = df.copy()
    if "lot_id" not in working_df.columns or "fab_value" not in working_df.columns:
        return working_df.reset_index(drop=True)

    grouped = working_df.groupby("lot_id")["fab_value"]
    working_df["lot_avg_fab_value"] = grouped.transform("mean")
    working_df["lot_std_fab_value"] = grouped.transform("std").fillna(0.0)
    working_df["lot_min_fab_value"] = grouped.transform("min")
    working_df["lot_max_fab_value"] = grouped.transform("max")
    working_df["lot_range_fab_value"] = working_df["lot_max_fab_value"] - working_df["lot_min_fab_value"]
    return working_df.reset_index(drop=True)


def preprocess_trend_data(
    df: pd.DataFrame,
    *,
    ppid: str = "",
    lot_type: str = "",
    lot_filter: str = "",
    add_lot_subitems: bool = False,
) -> pd.DataFrame:
    """단일 추세 데이터에 대한 표준 전처리를 수행한다."""
    processed_df = remove_fab_value_outliers_iqr(df)
    processed_df = filter_trend_data(
        processed_df,
        ppid=ppid,
        lot_type=lot_type,
        lot_filter=lot_filter,
    )

    if add_lot_subitems:
        processed_df = add_lot_subitem_values(processed_df)

    return processed_df.reset_index(drop=True)


def preprocess_dpm_input_data(
    df_incoming: pd.DataFrame,
    df_target: pd.DataFrame,
    inputs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """계산 단계 직전에 incoming/target 데이터를 정리한다."""
    processed_incoming_df = preprocess_trend_data(
        df_incoming,
        ppid=inputs.get("incoming_ppid", ""),
        lot_type=inputs.get("incoming_lot_type", ""),
        lot_filter=inputs.get("incoming_lot_filter", ""),
        add_lot_subitems=True,
    )
    processed_target_df = preprocess_trend_data(
        df_target,
        ppid=inputs.get("target_ppid", ""),
        lot_type=inputs.get("target_lot_type", ""),
        lot_filter="",
        add_lot_subitems=False,
    )
    return processed_incoming_df, processed_target_df
