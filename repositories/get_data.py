"""DPM 원본 데이터를 불러오는 저장소 모듈."""

from pathlib import Path

import pandas as pd


REPOSITORY_DIR = Path(__file__).resolve().parent
TEMP_INCOMING_PATH = REPOSITORY_DIR / "incoming_data.csv"
TEMP_TARGET_PATH = REPOSITORY_DIR / "target_data.csv"


def _load_temp_csv(path: Path) -> pd.DataFrame:
    """임시 CSV 파일을 읽고 시간 컬럼을 정규화한다."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "tkout_time" in df.columns:
        df["tkout_time"] = pd.to_datetime(df["tkout_time"], errors="coerce")
    return df


def _apply_common_query_filters(df: pd.DataFrame, query: dict, prefix: str) -> pd.DataFrame:
    """뷰에서 넘어온 공통 조회 조건을 데이터프레임에 적용한다."""
    working_df = df.copy()

    prc_step = query.get(f"{prefix}_prc_step")
    metro_step = query.get(f"{prefix}_metro_step")
    item = query.get(f"{prefix}_item")
    from_date = query.get("from_date")
    to_date = query.get("to_date")

    if prc_step and "step_seq" in working_df.columns:
        working_df = working_df[working_df["step_seq"].astype(str) == str(prc_step)]

    if metro_step and "metro_step_seq" in working_df.columns:
        metro_tokens = [token.strip() for token in str(metro_step).replace(";", ",").split(",") if token.strip()]
        if metro_tokens:
            working_df = working_df[working_df["metro_step_seq"].astype(str).isin(metro_tokens)]

    if item and "item_id" in working_df.columns:
        working_df = working_df[working_df["item_id"].astype(str).str.contains(str(item), case=False, na=False)]

    if from_date and "tkout_time" in working_df.columns:
        working_df = working_df[working_df["tkout_time"] >= pd.Timestamp(from_date)]

    if to_date and "tkout_time" in working_df.columns:
        working_df = working_df[working_df["tkout_time"] < pd.Timestamp(to_date) + pd.Timedelta(days=1)]

    return working_df.reset_index(drop=True)


def _empty_placeholder_df(name: str) -> pd.DataFrame:
    """아직 구현되지 않은 데이터 소스 대신 빈 결과를 반환한다."""
    return pd.DataFrame({"source_name": [name]}).iloc[0:0]


def load_mem_incoming(query: dict) -> pd.DataFrame:
    """메모리 incoming 데이터를 불러온다."""
    return _apply_common_query_filters(_load_temp_csv(TEMP_INCOMING_PATH), query, "incoming")


def load_mem_target(query: dict) -> pd.DataFrame:
    """메모리 target 데이터를 불러온다."""
    return _apply_common_query_filters(_load_temp_csv(TEMP_TARGET_PATH), query, "target")


def load_mem_side(query: dict) -> pd.DataFrame:
    """메모리 side 데이터를 위한 빈 결과를 반환한다."""
    return _empty_placeholder_df("mem_side")


def load_mem_l2(query: dict) -> pd.DataFrame:
    """메모리 L2 데이터를 위한 빈 결과를 반환한다."""
    return _empty_placeholder_df("mem_l2")


def load_mem_l3(query: dict) -> pd.DataFrame:
    """메모리 L3 데이터를 위한 빈 결과를 반환한다."""
    return _empty_placeholder_df("mem_l3")


def load_fdry_incoming(query: dict) -> pd.DataFrame:
    """파운드리 incoming 데이터를 불러온다."""
    return _apply_common_query_filters(_load_temp_csv(TEMP_INCOMING_PATH), query, "incoming")


def load_fdry_target(query: dict) -> pd.DataFrame:
    """파운드리 target 데이터를 불러온다."""
    return _apply_common_query_filters(_load_temp_csv(TEMP_TARGET_PATH), query, "target")


def load_fdry_side(query: dict) -> pd.DataFrame:
    """파운드리 side 데이터를 위한 빈 결과를 반환한다."""
    return _empty_placeholder_df("fdry_side")


def load_fdry_l2(query: dict) -> pd.DataFrame:
    """파운드리 L2 데이터를 위한 빈 결과를 반환한다."""
    return _empty_placeholder_df("fdry_l2")


def load_fdry_l3(query: dict) -> pd.DataFrame:
    """파운드리 L3 데이터를 위한 빈 결과를 반환한다."""
    return _empty_placeholder_df("fdry_l3")


def load_dpm_raw_data(query: dict) -> dict[str, pd.DataFrame]:
    """DPM 계산에 필요한 원본 데이터 묶음을 불러온다."""
    fab_type = query.get("fab_type", "memory")

    # 향후 DB 연동 시에도 동일한 함수 시그니처를 유지하기 위해
    # 현재 단계부터 필요한 조회 파라미터를 모두 경계에 실어 나른다.
    _ = (
        query.get("line_id", []),
        query.get("process_id", []),
        query.get("side_metro_step"),
        query.get("side_metro_item", ""),
        query.get("l2_step_seq", ""),
        query.get("l2_item_id", ""),
        query.get("l3_step_seq", ""),
        query.get("l3_item_id", ""),
    )

    if fab_type == "memory":
        df_incoming = load_mem_incoming(query)
        df_target = load_mem_target(query)
        df_side = load_mem_side(query)
        df_l2 = load_mem_l2(query)
        df_l3 = load_mem_l3(query)
    else:
        df_incoming = load_fdry_incoming(query)
        df_target = load_fdry_target(query)
        df_side = load_fdry_side(query)
        df_l2 = load_fdry_l2(query)
        df_l3 = load_fdry_l3(query)

    return {
        "fab_type": fab_type,
        "df_incoming": df_incoming,
        "df_target": df_target,
        "df_side": df_side,
        "df_l2": df_l2,
        "df_l3": df_l3,
    }
