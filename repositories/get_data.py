from pathlib import Path

import pandas as pd


# repositories 계층은 실제 사내 DB 또는 파일 저장소와 통신하는 자리를 의미한다.
# 현재는 사내 환경이 아니어서 DB 점검이 어려우므로 임시 CSV를 사용한다.
# 추후 사내 환경으로 돌아가면 아래 load_mem_* / load_fdry_* 함수 내부의
# "TEMP CSV PLACEHOLDER" 구간을 실제 SQL 실행 코드로 교체하면 된다.
REPOSITORY_DIR = Path(__file__).resolve().parent
TEMP_INCOMING_PATH = REPOSITORY_DIR / "incoming_data.csv"
TEMP_TARGET_PATH = REPOSITORY_DIR / "target_data.csv"


def _load_temp_csv(path: Path) -> pd.DataFrame:
    # 현재 개발 환경에서는 임시 CSV를 읽어 dataframe을 만든다.
    # tkout_time은 이후 시각화와 기간 필터링에서 사용되므로 datetime으로 변환한다.
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "tkout_time" in df.columns:
        df["tkout_time"] = pd.to_datetime(df["tkout_time"], errors="coerce")
    return df


def _apply_common_query_filters(df: pd.DataFrame, query: dict, prefix: str) -> pd.DataFrame:
    # repositories 단계에서는 step, metro step, item, 기간처럼
    # raw data volume 자체를 줄일 수 있는 조건만 먼저 적용한다.
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
    # side / l2 / l3는 아직 사내 DB를 붙이지 못했으므로 빈 dataframe을 반환한다.
    # 추후 실제 쿼리로 교체되면 이 함수는 제거 가능하다.
    return pd.DataFrame({"source_name": [name]}).iloc[0:0]


def load_mem_incoming(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 memory incoming SQL로 교체할 함수다.
    # 현재는 repositories/incoming_data.csv를 임시 참조한다.
    df_incoming = _load_temp_csv(TEMP_INCOMING_PATH)
    return _apply_common_query_filters(df_incoming, query, "incoming")


def load_mem_target(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 memory target SQL로 교체할 함수다.
    # 현재는 repositories/target_data.csv를 임시 참조한다.
    df_target = _load_temp_csv(TEMP_TARGET_PATH)
    return _apply_common_query_filters(df_target, query, "target")


def load_mem_side(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 memory side용 SQL 결과를 반환하도록 교체한다.
    return _empty_placeholder_df("mem_side")


def load_mem_l2(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 memory l2용 SQL 결과를 반환하도록 교체한다.
    return _empty_placeholder_df("mem_l2")


def load_mem_l3(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 memory l3용 SQL 결과를 반환하도록 교체한다.
    return _empty_placeholder_df("mem_l3")


def load_fdry_incoming(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 foundry incoming SQL로 교체할 함수다.
    # 현재는 repositories/incoming_data.csv를 임시 참조한다.
    df_incoming = _load_temp_csv(TEMP_INCOMING_PATH)
    return _apply_common_query_filters(df_incoming, query, "incoming")


def load_fdry_target(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 foundry target SQL로 교체할 함수다.
    # 현재는 repositories/target_data.csv를 임시 참조한다.
    df_target = _load_temp_csv(TEMP_TARGET_PATH)
    return _apply_common_query_filters(df_target, query, "target")


def load_fdry_side(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 foundry side용 SQL 결과를 반환하도록 교체한다.
    return _empty_placeholder_df("fdry_side")


def load_fdry_l2(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 foundry l2용 SQL 결과를 반환하도록 교체한다.
    return _empty_placeholder_df("fdry_l2")


def load_fdry_l3(query: dict) -> pd.DataFrame:
    # TEMP CSV PLACEHOLDER:
    # 추후 foundry l3용 SQL 결과를 반환하도록 교체한다.
    return _empty_placeholder_df("fdry_l3")


def load_dpm_raw_data(query: dict) -> dict[str, pd.DataFrame]:
    """
    UI 입력값(query)을 받아 repositories 계층의 raw dataframe들을 로드한다.
    반환 형식은 services 계층에서 그대로 사용할 수 있도록 df_incoming, df_target,
    df_side, df_l2, df_l3 이름으로 고정한다.
    """
    fab_type = query.get("fab_type", "memory")
    line_id = query.get("line_id", [])
    process_id = query.get("process_id", [])
    side_metro_step = query.get("side_metro_step")
    side_metro_item = query.get("side_metro_item", "")
    l2_step_seq = query.get("l2_step_seq", "")
    l2_item_id = query.get("l2_item_id", "")
    l3_step_seq = query.get("l3_step_seq", "")
    l3_item_id = query.get("l3_item_id", "")

    # DB QUERY PLACEHOLDER:
    # 추후 사내 환경에서는 아래 분기 내부에서 각 함수가 실제 DB query를 수행하게 된다.
    # 함수명은 유지하고 내부 구현만 SQL / connector 코드로 교체하는 것을 권장한다.
    # query dict에는 fab_type, line_id, process_id 외에도 side / l2 / l3 popup 값이 함께 들어온다.
    # 현재 임시 CSV 경로에서는 이 값들을 실제 조회 조건에 쓰지 않지만,
    # 사내 전환 시 각 load_* 함수의 SQL where 절 또는 parameter binding에 그대로 연결하면 된다.
    _ = (line_id, process_id, side_metro_step, side_metro_item, l2_step_seq, l2_item_id, l3_step_seq, l3_item_id)
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
