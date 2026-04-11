import pandas as pd


# dpm_calculation 계층은 preprocessing이 끝난 incoming / target 데이터를 받아
# DPM setting 기준의 이상치 판정과 제어용 결과 데이터셋을 만드는 역할을 한다.
# 현재 구현 범위는 df_incoming 기준 outlier 판정과 시각화용 dataframe 생성까지다.


def _safe_float(value) -> float | None:
    # text_input으로 들어온 숫자 문자열을 float으로 안전하게 변환한다.
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_lot_representative_df(incoming_df: pd.DataFrame) -> pd.DataFrame:
    # preprocessing에서 lot_mean_fab_value 컬럼을 이미 추가해 두었으므로
    # lot 단위 제어용 dataframe은 lot_id 기준 대표 row 하나만 남겨 사용한다.
    if incoming_df is None or incoming_df.empty:
        return pd.DataFrame()

    working_df = incoming_df.copy()
    required_cols = ["lot_id", "tkout_time", "lot_mean_fab_value"]
    existing_required_cols = [col for col in required_cols if col in working_df.columns]
    if len(existing_required_cols) < 3:
        return pd.DataFrame()

    working_df = working_df.sort_values("tkout_time")
    representative_df = working_df.groupby("lot_id", as_index=False).first()
    return representative_df.reset_index(drop=True)


def _resolve_thresholds(df: pd.DataFrame, value_col: str, inputs: dict) -> tuple[float | None, float | None]:
    # control_type에 따라 lower / upper 판정 기준을 계산한다.
    # percentile은 "그 값보다 작은 물량"을 outlier로 보므로 lower threshold만 사용한다.
    if df is None or df.empty or value_col not in df.columns:
        return None, None

    control_type = inputs.get("control_type", "sigma")
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        return None, None

    if control_type == "sigma":
        sigma = _safe_float(inputs.get("sigma"))
        if sigma is None:
            return None, None

        avg = values.mean()
        std = values.std()
        if pd.isna(std):
            return None, None
        upper_spec = avg + sigma * std
        lower_spec = avg - sigma * std
        return float(lower_spec), float(upper_spec)

    if control_type == "spec":
        lower_spec = _safe_float(inputs.get("lsl"))
        upper_spec = _safe_float(inputs.get("usl"))
        return lower_spec, upper_spec

    percentile = _safe_float(inputs.get("percentile"))
    if percentile is None:
        return None, None

    percentile = max(0.0, min(100.0, percentile))
    lower_spec = values.quantile(percentile / 100.0)
    return float(lower_spec), None


def _classify_outliers(df: pd.DataFrame, value_col: str, inputs: dict) -> tuple[pd.DataFrame, dict]:
    # dataframe의 특정 값 컬럼을 기준으로 normal / lower_out / upper_out을 판정한다.
    if df is None or df.empty or value_col not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else None), {
            "value_col": value_col,
            "lower_spec": None,
            "upper_spec": None,
            "control_type": inputs.get("control_type"),
        }

    working_df = df.copy()
    numeric_values = pd.to_numeric(working_df[value_col], errors="coerce")
    lower_spec, upper_spec = _resolve_thresholds(working_df, value_col, inputs)

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
        "control_type": inputs.get("control_type"),
    }


def run_dpm_calculation(
    incoming_df: pd.DataFrame,
    target_df: pd.DataFrame,
    inputs: dict,
    df_side: pd.DataFrame | None = None,
    df_l2: pd.DataFrame | None = None,
    df_l3: pd.DataFrame | None = None,
) -> dict:
    """
    preprocessing 결과를 받아 제어용 outlier 판정 결과를 만든다.
    현재 target/side/l2/l3 계산은 placeholder이고, incoming 기반 제어 차트 데이터셋을 우선 구성한다.
    """
    wafer_control_df, wafer_meta = _classify_outliers(incoming_df, "fab_value", inputs)

    lot_representative_df = _build_lot_representative_df(incoming_df)
    lot_control_df, lot_meta = _classify_outliers(lot_representative_df, "lot_mean_fab_value", inputs)

    # 현재 dummy incoming 데이터에는 eqp_id / eqp_chamber 컬럼이 없으므로
    # Lot - EQP, Lot - Chamber는 동일한 lot 대표값 dataframe을 사용한다.
    # 추후 repositories의 incoming schema가 확장되면 여기서 그룹 기준만 분기하면 된다.
    lot_eqp_df = lot_control_df.copy()
    lot_chamber_df = lot_control_df.copy()
    wafer_chamber_df = wafer_control_df.copy()

    return {
        "status": "implemented_for_outlier_judgement",
        "message": "Incoming outlier judgement is ready for control charts.",
        "incoming_rows": 0 if incoming_df is None else len(incoming_df),
        "target_rows": 0 if target_df is None else len(target_df),
        "side_rows": 0 if df_side is None else len(df_side),
        "l2_rows": 0 if df_l2 is None else len(df_l2),
        "l3_rows": 0 if df_l3 is None else len(df_l3),
        "fab_type": inputs.get("fab_type"),
        "control_scope": inputs.get("control_scope"),
        "control_type": inputs.get("control_type"),
        "lot_eqp_df": lot_eqp_df,
        "lot_chamber_df": lot_chamber_df,
        "wafer_chamber_df": wafer_chamber_df,
        "lot_meta": lot_meta,
        "wafer_meta": wafer_meta,
    }
