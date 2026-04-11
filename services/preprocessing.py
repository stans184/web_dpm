import pandas as pd


# preprocessing 계층은 raw data를 계산 가능한 형태로 다듬는 역할을 한다.
# 현재 요구사항은 다음 3단계다.
# 1. fab_value 기준 3*IQR outlier 제거
# 2. ppid / lot_type / lot_filter 기준 filtering
# 3. incoming 데이터에 lot_id 기준 lot mean 값 추가


def remove_fab_value_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    # fab_value 분포에서 3*IQR 범위를 벗어나는 값을 제거한다.
    # 데이터가 없거나 필요한 컬럼이 없으면 원본 형태를 최대한 유지해서 반환한다.
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
    filtered_df = working_df[
        working_df["fab_value"].between(lower_bound, upper_bound, inclusive="both")
    ]
    return filtered_df.reset_index(drop=True)


def preprocess_comparison_trend_data(
    incoming_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # incoming vs target comparison scatter는 raw trend 전체를 그대로 쓰지 않고
    # 양쪽 dataframe에 3*IQR outlier 제거만 적용한 뒤 비교하는 용도로 사용한다.
    # 이 로직을 view가 아니라 preprocessing 서비스 안에 두어 역할을 분리한다.
    filtered_incoming_df = remove_fab_value_outliers_iqr(incoming_df)
    filtered_target_df = remove_fab_value_outliers_iqr(target_df)
    return filtered_incoming_df, filtered_target_df


def filter_trend_data(
    df: pd.DataFrame,
    *,
    ppid: str = "",
    lot_type: str = "",
    lot_filter: str = "",
) -> pd.DataFrame:
    # DPM setting popup에서 받은 ppid / lot_type / lot_filter 조건으로
    # raw trend data를 한 번 더 세밀하게 걸러낸다.
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    working_df = df.copy()

    if ppid and "ppid" in working_df.columns:
        working_df = working_df[
            working_df["ppid"].astype(str).str.contains(ppid, case=False, na=False)
        ]

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
        # lot_filter는 lot_id 또는 root_lot_id 어느 쪽에도 걸릴 수 있도록 처리한다.
        lot_mask = pd.Series(False, index=working_df.index)
        for col in ["lot_id", "root_lot_id"]:
            if col in working_df.columns:
                lot_mask = lot_mask | working_df[col].astype(str).str.contains(
                    lot_filter, case=False, na=False
                )
        working_df = working_df[lot_mask]

    return working_df.reset_index(drop=True)


def add_lot_mean_fab_value(df: pd.DataFrame) -> pd.DataFrame:
    # incoming dataframe에서는 같은 lot_id 묶음별 fab_value 평균을 계산해서
    # lot_mean_fab_value 컬럼으로 추가한다.
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    working_df = df.copy()
    if "lot_id" not in working_df.columns or "fab_value" not in working_df.columns:
        return working_df.reset_index(drop=True)

    working_df["lot_mean_fab_value"] = working_df.groupby("lot_id")["fab_value"].transform("mean")
    return working_df.reset_index(drop=True)


def preprocess_trend_data(
    df: pd.DataFrame,
    *,
    ppid: str = "",
    lot_type: str = "",
    lot_filter: str = "",
    add_lot_mean: bool = False,
) -> pd.DataFrame:
    # preprocessing의 메인 진입 함수다.
    # 서비스 순서를 한 곳에 모아 두면 뷰에서는 이 함수만 호출하면 된다.
    processed_df = remove_fab_value_outliers_iqr(df)
    processed_df = filter_trend_data(
        processed_df,
        ppid=ppid,
        lot_type=lot_type,
        lot_filter=lot_filter,
    )

    if add_lot_mean:
        processed_df = add_lot_mean_fab_value(processed_df)

    return processed_df.reset_index(drop=True)


def preprocess_dpm_input_data(
    df_incoming: pd.DataFrame,
    df_target: pd.DataFrame,
    inputs: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # DPM 계산 직전에 사용하는 메인 preprocessing 진입 함수다.
    # get_data.py에서 불러온 raw dataframe을 받아 아래 순서로 처리한다.
    # 1. df_incoming, df_target 각각 fab_value 기준 3*IQR outlier 제거
    # 2. df_incoming은 incoming ppid / incoming lot type / incoming lot filter 기준 filtering
    # 3. df_target은 target ppid / target lot type 기준 filtering
    # 4. df_incoming에는 lot_id 기준 fab_value 평균 컬럼(lot_mean_fab_value) 추가
    # 이렇게 만들어진 결과를 dpm_calculation.py에 전달한다.
    processed_incoming_df = preprocess_trend_data(
        df_incoming,
        ppid=inputs.get("incoming_ppid", ""),
        lot_type=inputs.get("incoming_lot_type", ""),
        lot_filter=inputs.get("incoming_lot_filter", ""),
        add_lot_mean=True,
    )
    processed_target_df = preprocess_trend_data(
        df_target,
        ppid=inputs.get("target_ppid", ""),
        lot_type=inputs.get("target_lot_type", ""),
        lot_filter="",
        add_lot_mean=False,
    )
    return processed_incoming_df, processed_target_df
