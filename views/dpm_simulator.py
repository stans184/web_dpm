# file: view/dpm_simulator.py

# 이 뷰 파일은 DPM Simulator 화면 전체를 담당한다.
# 화면 입력값을 세션에 저장하고, Run all / Apply DPM setting only 버튼에 따라
# repositories -> services -> visualization 흐름을 연결하는 역할을 한다.

from pathlib import Path

import pandas as pd
import streamlit as st

from components.download_button import render_download_button
from repositories.get_data import load_dpm_raw_data
from services.dpm_calculation import run_dpm_calculation
from services.preprocessing import preprocess_comparison_trend_data, preprocess_dpm_input_data
from services.visualization import (
    build_comparison_scatter_with_regression,
    build_outlier_scatter,
    build_trend_scatter,
    join_incoming_target_for_comparison,
)

# INPUT_DEFAULTS는 화면 입력의 기본 구조를 정의한다.
# 이 dict 자체가 이후 DB 조회 파라미터의 기본 골격이 되므로
# UI에서 입력받는 값과 동일한 이름으로 유지한다.
INPUT_DEFAULTS = {
    "fab_type": "memory",
    "line_id": [],
    "process_id": [],
    "incoming_prc_step": None,
    "incoming_metro_step": "",
    "incoming_item": "",
    "incoming_ppid": "",
    "incoming_lot_type": "ALL",
    "incoming_lot_filter": "",
    "target_prc_step": None,
    "target_metro_step": "",
    "target_item": "",
    "target_ppid": "",
    "target_lot_type": "ALL",
    "side_metro_step": None,
    "side_metro_item": "",
    "l2_step_seq": "",
    "l2_item_id": "",
    "l3_step_seq": "",
    "l3_item_id": "",
    "control_scope": "망목제어",
    "from_date": None,
    "to_date": None,
    "control_type": "sigma",
    "sigma": "",
    "usl": "",
    "lsl": "",
    "percentile": "",
    "wow_portion": "",
    "window_days": "",
    "min_wafer_qty": "",
}

STEP_MAPPING_FILES = {
    "memory": "mem_prc_with_met.csv",
    "foundry": "fdry_prc_with_met.csv",
}

LOT_TYPE_OPTIONS = ["ALL", "P type"]
CONTROL_SCOPE_OPTIONS = ["망목제어", "망소제어 (큰 설비 제어)", "망대제어 (작은 설비 제어)"]


def _compact_value(value) -> str:
    # 캡션에 표시할 때 빈값은 "-"로 통일해서 보여주기 위한 유틸 함수다.
    if value in (None, ""):
        return "-"
    return str(value)


def _normalize_step_tokens(raw_value: str) -> list[str]:
    # 사용자가 metro step을 여러 개 입력할 수 있도록
    # 줄바꿈, 세미콜론, 콤마를 모두 동일한 구분자로 정규화한다.
    if not raw_value:
        return []

    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip().lower() for token in normalized.split(",") if token.strip()]


def _normalize_line_ids(raw_value: str | list[str] | None) -> list[str]:
    # line_id는 화면에서는 콤마 구분 문자열로 입력받되,
    # 내부 query parameter에서는 list 형태로 통일해서 사용한다.
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]

    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _line_ids_to_text(line_ids: str | list[str] | None) -> str:
    # 세션에 저장된 list 형태 line_id를 text_input에 다시 넣기 위해
    # 화면 표시용 문자열로 변환한다.
    return ", ".join(_normalize_line_ids(line_ids))


def _normalize_process_ids(raw_value: str | list[str] | None) -> list[str]:
    # process_id도 line_id와 같은 방식으로 콤마 구분 문자열을 list로 정규화한다.
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]

    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _process_ids_to_text(process_ids: str | list[str] | None) -> str:
    # 세션에 저장된 process_id list를 text_input 표시용 문자열로 바꾼다.
    return ", ".join(_normalize_process_ids(process_ids))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    # step mapping dataframe의 실제 컬럼명이 조금 달라도
    # 후보 이름 목록 안에서 대응되는 컬럼을 찾기 위한 함수다.
    normalized_columns = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def _load_step_mapping_df_for_fab(fab_type: str) -> pd.DataFrame:
    # selected_fab_type에 따라 서로 다른 step mapping csv를 읽는다.
    # memory는 mem_prc_with_met.csv, foundry는 fdry_prc_with_met.csv를 참조한다.
    repository_dir = Path(__file__).resolve().parents[1] / "repositories"
    filename = STEP_MAPPING_FILES.get(fab_type)
    if not filename:
        return pd.DataFrame()

    csv_path = repository_dir / filename
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def _get_prc_step_candidates(
    step_mapping_df: pd.DataFrame,
    metro_step_text: str,
    line_ids: list[str] | None = None,
    process_ids: list[str] | None = None,
) -> list[str]:
    # 사용자가 입력한 metro step을 기준으로
    # 매핑 dataframe에서 선택 가능한 prc step 후보 목록을 만든다.
    if step_mapping_df is None or step_mapping_df.empty:
        return []

    prc_col = _find_column(
        step_mapping_df,
        ["prc_step", "prc step", "prc_step_seq", "prc step seq", "step_seq", "step"],
    )
    metro_col = _find_column(
        step_mapping_df,
        ["metro_step", "metro step", "metro_step_seq", "met_step_seq", "metro"],
    )
    line_col = _find_column(step_mapping_df, ["line_id", "line id"])
    process_col = _find_column(step_mapping_df, ["process_id", "process id"])
    if prc_col is None or metro_col is None:
        return []

    metro_tokens = _normalize_step_tokens(metro_step_text)
    if not metro_tokens:
        return []

    selected_columns = [prc_col, metro_col]
    if line_col is not None:
        selected_columns.append(line_col)
    if process_col is not None:
        selected_columns.append(process_col)

    working_df = step_mapping_df[selected_columns].dropna(subset=[prc_col, metro_col]).copy()
    working_df[metro_col] = working_df[metro_col].astype(str).str.strip()
    working_df[prc_col] = working_df[prc_col].astype(str).str.strip()
    if line_col is not None:
        working_df[line_col] = working_df[line_col].astype(str).str.strip()
        normalized_line_ids = [line_id.strip().lower() for line_id in (line_ids or []) if line_id.strip()]
        if normalized_line_ids:
            working_df = working_df[working_df[line_col].str.lower().isin(normalized_line_ids)]
    if process_col is not None:
        working_df[process_col] = working_df[process_col].astype(str).str.strip()
        normalized_process_ids = [process_id.strip().lower() for process_id in (process_ids or []) if process_id.strip()]
        if normalized_process_ids:
            working_df = working_df[working_df[process_col].str.lower().isin(normalized_process_ids)]

    matched_rows = working_df[working_df[metro_col].str.lower().isin(metro_tokens)]
    return sorted(matched_rows[prc_col].drop_duplicates().tolist())


def _get_side_metro_candidates(
    step_mapping_df: pd.DataFrame,
    target_prc_step: str | None,
    line_ids: list[str] | None = None,
    process_ids: list[str] | None = None,
) -> list[str]:
    # Side Effect popup은 선택된 target prc step을 기준으로 연관 met_step_seq 후보를 보여준다.
    # 여기서 fab_type, line_id, process_id를 함께 적용하면 실제 설비/라인 조건에 맞는 side step만 고를 수 있다.
    if step_mapping_df is None or step_mapping_df.empty or not target_prc_step:
        return []

    prc_col = _find_column(
        step_mapping_df,
        ["prc_step", "prc step", "prc_step_seq", "prc step seq", "step_seq", "step"],
    )
    metro_col = _find_column(
        step_mapping_df,
        ["metro_step", "metro step", "metro_step_seq", "met_step_seq", "metro"],
    )
    line_col = _find_column(step_mapping_df, ["line_id", "line id"])
    process_col = _find_column(step_mapping_df, ["process_id", "process id"])
    if prc_col is None or metro_col is None:
        return []

    selected_columns = [prc_col, metro_col]
    if line_col is not None:
        selected_columns.append(line_col)
    if process_col is not None:
        selected_columns.append(process_col)

    working_df = step_mapping_df[selected_columns].dropna(subset=[prc_col, metro_col]).copy()
    working_df[prc_col] = working_df[prc_col].astype(str).str.strip()
    working_df[metro_col] = working_df[metro_col].astype(str).str.strip()
    working_df = working_df[working_df[prc_col].str.lower() == str(target_prc_step).strip().lower()]

    if line_col is not None:
        working_df[line_col] = working_df[line_col].astype(str).str.strip()
        normalized_line_ids = [line_id.strip().lower() for line_id in (line_ids or []) if line_id.strip()]
        if normalized_line_ids:
            working_df = working_df[working_df[line_col].str.lower().isin(normalized_line_ids)]

    if process_col is not None:
        working_df[process_col] = working_df[process_col].astype(str).str.strip()
        normalized_process_ids = [process_id.strip().lower() for process_id in (process_ids or []) if process_id.strip()]
        if normalized_process_ids:
            working_df = working_df[working_df[process_col].str.lower().isin(normalized_process_ids)]

    return sorted(working_df[metro_col].drop_duplicates().tolist())


def _get_legend_options(*dfs: pd.DataFrame) -> list[str]:
    # trend scatter에서 legend로 쓸 수 있는 후보 컬럼 목록을 만든다.
    # 사용자 입장에서 많이 바꿔볼 만한 대표 컬럼만 노출한다.
    candidate_columns = ["ppid", "lot_type", "item_id", "root_lot_id", "lot_id"]
    options: list[str] = []
    for col in candidate_columns:
        if any(df is not None and not df.empty and col in df.columns for df in dfs):
            options.append(col)
    return options or ["ppid"]


def _get_joined_legend_options(joined_df: pd.DataFrame) -> list[str]:
    # incoming/target join 결과에서는 suffix가 붙은 컬럼명을 기준으로
    # legend 후보를 별도로 만들어 준다.
    candidate_columns = [
        "ppid_incoming",
        "ppid_target",
        "lot_type_incoming",
        "lot_type_target",
        "item_id_incoming",
        "item_id_target",
        "root_lot_id",
    ]
    return [col for col in candidate_columns if joined_df is not None and col in joined_df.columns] or [
        "ppid_incoming"
    ]


def _run_preprocessing_and_calculation(
    df_incoming: pd.DataFrame,
    df_target: pd.DataFrame,
    query_inputs: dict,
    df_side: pd.DataFrame | None = None,
    df_l2: pd.DataFrame | None = None,
    df_l3: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # services 단계의 공통 파이프라인이다.
    # Run all에서는 get_data 이후 이 함수를 호출하고,
    # Apply DPM setting only에서는 이미 보관된 raw data를 그대로 넣어서 다시 실행한다.
    # 실제 3IQR outlier 제거, incoming/target filtering, incoming lot 평균 추가는
    # preprocessing 서비스 내부 함수가 전담하고 여기서는 호출만 한다.
    incoming_preprocessed, target_preprocessed = preprocess_dpm_input_data(
        df_incoming,
        df_target,
        query_inputs,
    )
    calculation_result = run_dpm_calculation(
        incoming_preprocessed,
        target_preprocessed,
        query_inputs,
        df_side=df_side,
        df_l2=df_l2,
        df_l3=df_l3,
    )
    return incoming_preprocessed, target_preprocessed, calculation_result


def render_dpm_simulator():
    # Streamlit 탭 안에 그려지는 DPM Simulator 메인 화면 함수다.
    st.markdown("### DPM Simulator")

    with st.container(border=True):
        st.markdown("#### Input Section")

        # memory / foundry 선택값도 query parameter의 일부이므로
        # 이후 repositories.get_data 로 전달되는 입력값에 포함된다.
        # 상단의 fab_type, line_id, process_id는 서로 연관된 조회 조건이므로
        # 같은 줄의 왼쪽에 몰아서 배치해 한 눈에 설정할 수 있게 한다.
        top_col_fab, top_col_line, top_col_process, top_col_spacer = st.columns([1.1, 1.2, 1.2, 4.5])
        with top_col_fab:
            selected_fab_type = st.radio(
                "Fab Type",
                ["memory", "foundry"],
                horizontal=True,
                key="fab_type_selector",
                index=0 if st.session_state["current_fab_type"] == "memory" else 1,
                label_visibility="collapsed",
            )
        st.session_state["current_fab_type"] = selected_fab_type

        # 세션에 기존 입력값이 있어도 최신 기본 스키마와 merge해서
        # 신규 필드가 누락되지 않도록 보정한다.
        st.session_state["dpm_inputs"][selected_fab_type] = {
            **INPUT_DEFAULTS,
            **st.session_state["dpm_inputs"][selected_fab_type],
        }
        current_inputs = st.session_state["dpm_inputs"][selected_fab_type]
        with top_col_line:
            line_ids = _normalize_line_ids(current_inputs.get("line_id"))
            line_id = st.text_input(
                "Line ID",
                value=_line_ids_to_text(line_ids),
                key=f"line_id_{selected_fab_type}",
                placeholder="Line ID ex)KFBN,P3DF",
                label_visibility="collapsed",
            )
        line_ids = _normalize_line_ids(line_id)
        with top_col_process:
            process_ids = _normalize_process_ids(current_inputs.get("process_id"))
            process_id = st.text_input(
                "Process ID",
                value=_process_ids_to_text(process_ids),
                key=f"process_id_{selected_fab_type}",
                placeholder="Process ID ex)PRC_01,PRC_02",
                label_visibility="collapsed",
            )
        process_ids = _normalize_process_ids(process_id)
        step_mapping_df = _load_step_mapping_df_for_fab(selected_fab_type)

        st.markdown("---")

        # 입력창 크기를 조금 더 컴팩트하게 보이도록 공통 스타일을 지정한다.
        st.markdown(
            """
            <style>
            div[data-testid="stTextInput"] input,
            div[data-testid="stDateInput"] input,
            div[data-testid="stSelectbox"] input {
                font-size: 0.9rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # 현재 저장된 metro step 기준으로 prc step 후보를 먼저 계산해 둔다.
        # 사용자가 metro step을 수정하면 popover 내부에서 다시 한 번 즉시 계산한다.
        incoming_metro_step = current_inputs["incoming_metro_step"]
        incoming_prc_candidates = _get_prc_step_candidates(
            step_mapping_df,
            incoming_metro_step,
            line_ids,
            process_ids,
        )
        if current_inputs["incoming_prc_step"] not in incoming_prc_candidates:
            current_inputs["incoming_prc_step"] = None

        target_metro_step = current_inputs["target_metro_step"]
        target_prc_candidates = _get_prc_step_candidates(
            step_mapping_df,
            target_metro_step,
            line_ids,
            process_ids,
        )
        if current_inputs["target_prc_step"] not in target_prc_candidates:
            current_inputs["target_prc_step"] = None

        side_metro_candidates = _get_side_metro_candidates(
            step_mapping_df,
            current_inputs["target_prc_step"],
            line_ids,
            process_ids,
        )
        if current_inputs["side_metro_step"] not in side_metro_candidates:
            current_inputs["side_metro_step"] = None

        # popup은 조회 조건 종류별로 분리한다.
        # Side Effect와 L2-L3를 별도로 두는 이유는 repositories 계층에
        # 각각 다른 raw query parameter로 넘겨 이후 개별 데이터셋을 로드할 수 있게 하기 위함이다.
        col_incoming, col_target, col_side, col_l2l3, col_dpm, col_action = st.columns([1, 1, 1, 1, 1, 0.95])

        with col_incoming:
            # Incoming popover는 incoming 쿼리의 핵심 식별값만 받는다.
            with st.popover("Incoming", use_container_width=True):
                incoming_metro_step = st.text_input(
                    "Incoming metro step",
                    value=current_inputs["incoming_metro_step"],
                    key=f"in_metro_step_{selected_fab_type}",
                    placeholder="Incoming metro step",
                    label_visibility="collapsed",
                )
                refreshed_incoming_candidates = _get_prc_step_candidates(
                    step_mapping_df,
                    incoming_metro_step,
                    line_ids,
                    process_ids,
                )
                incoming_prc_step = st.selectbox(
                    "Incoming prc step",
                    options=refreshed_incoming_candidates,
                    index=(
                        refreshed_incoming_candidates.index(current_inputs["incoming_prc_step"])
                        if current_inputs["incoming_prc_step"] in refreshed_incoming_candidates
                        else None
                    ),
                    key=f"in_prc_step_{selected_fab_type}",
                    placeholder="Incoming prc step",
                    label_visibility="collapsed",
                )
                incoming_item = st.text_input(
                    "Incoming item",
                    value=current_inputs["incoming_item"],
                    key=f"in_item_{selected_fab_type}",
                    placeholder="Incoming item",
                    label_visibility="collapsed",
                )

            st.caption(
                "prc: "
                f"{_compact_value(incoming_prc_step)} | metro: {_compact_value(incoming_metro_step)} | "
                f"item: {_compact_value(incoming_item)}"
            )

        with col_target:
            # Target popover도 incoming과 동일한 패턴으로 구성하되
            # 조회 기간 입력까지 함께 받는다.
            with st.popover("Target", use_container_width=True):
                target_metro_step = st.text_input(
                    "Target metro step",
                    value=current_inputs["target_metro_step"],
                    key=f"tar_metro_step_{selected_fab_type}",
                    placeholder="Target metro step",
                    label_visibility="collapsed",
                )
                refreshed_target_candidates = _get_prc_step_candidates(
                    step_mapping_df,
                    target_metro_step,
                    line_ids,
                    process_ids,
                )
                target_prc_step = st.selectbox(
                    "Target prc step",
                    options=refreshed_target_candidates,
                    index=(
                        refreshed_target_candidates.index(current_inputs["target_prc_step"])
                        if current_inputs["target_prc_step"] in refreshed_target_candidates
                        else None
                    ),
                    key=f"tar_prc_step_{selected_fab_type}",
                    placeholder="Target prc step",
                    label_visibility="collapsed",
                )
                target_item = st.text_input(
                    "Target item",
                    value=current_inputs["target_item"],
                    key=f"tar_item_{selected_fab_type}",
                    placeholder="Target item",
                    label_visibility="collapsed",
                )
                date_col_from, date_col_to = st.columns(2)
                with date_col_from:
                    from_date = st.date_input(
                        "From Date",
                        value=current_inputs["from_date"],
                        key=f"from_date_{selected_fab_type}",
                    )
                with date_col_to:
                    to_date = st.date_input(
                        "To Date",
                        value=current_inputs["to_date"],
                        key=f"to_date_{selected_fab_type}",
                    )

            st.caption(
                "prc: "
                f"{_compact_value(target_prc_step)} | metro: {_compact_value(target_metro_step)} | "
                f"{_compact_value(from_date)} ~ {_compact_value(to_date)}"
            )

        with col_side:
            # Side Effect popup은 target prc step이 정해졌을 때만 side met step 후보를 계산한다.
            # 이 값들은 side effect용 dataframe을 repositories에서 조회할 때 필요한 핵심 key가 된다.
            with st.popover("Side Effect", use_container_width=True):
                refreshed_side_candidates = _get_side_metro_candidates(
                    step_mapping_df,
                    target_prc_step,
                    line_ids,
                    process_ids,
                )
                side_metro_step = st.selectbox(
                    "Side metro step",
                    options=refreshed_side_candidates,
                    index=(
                        refreshed_side_candidates.index(current_inputs["side_metro_step"])
                        if current_inputs["side_metro_step"] in refreshed_side_candidates
                        else None
                    ),
                    key=f"side_metro_step_{selected_fab_type}",
                    placeholder="Side metro step",
                    label_visibility="collapsed",
                )
                side_metro_item = st.text_input(
                    "Side metro item",
                    value=current_inputs["side_metro_item"],
                    key=f"side_metro_item_{selected_fab_type}",
                    placeholder="Side metro item",
                    label_visibility="collapsed",
                )

            st.caption(
                f"metro: {_compact_value(side_metro_step)} | item: {_compact_value(side_metro_item)}"
            )

        with col_l2l3:
            # L2 / L3 popup은 추가 원인 분석 또는 후속 계산용 입력을 받는다.
            # repositories에 step_seq와 item_id를 별도 전달하면 추후 l2/l3 전용 query로 확장하기 쉽다.
            with st.popover("L2-L3", use_container_width=True):
                l2_step_seq = st.text_input(
                    "L2 step seq",
                    value=current_inputs["l2_step_seq"],
                    key=f"l2_step_seq_{selected_fab_type}",
                    placeholder="L2 step seq",
                    label_visibility="collapsed",
                )
                l2_item_id = st.text_input(
                    "L2 item id",
                    value=current_inputs["l2_item_id"],
                    key=f"l2_item_id_{selected_fab_type}",
                    placeholder="L2 item id",
                    label_visibility="collapsed",
                )
                l3_step_seq = st.text_input(
                    "L3 step seq",
                    value=current_inputs["l3_step_seq"],
                    key=f"l3_step_seq_{selected_fab_type}",
                    placeholder="L3 step seq",
                    label_visibility="collapsed",
                )
                l3_item_id = st.text_input(
                    "L3 item id",
                    value=current_inputs["l3_item_id"],
                    key=f"l3_item_id_{selected_fab_type}",
                    placeholder="L3 item id",
                    label_visibility="collapsed",
                )

            st.caption(
                f"L2: {_compact_value(l2_step_seq)} / {_compact_value(l2_item_id)} | "
                f"L3: {_compact_value(l3_step_seq)} / {_compact_value(l3_item_id)}"
            )

        with col_dpm:
            # DPM Setting popover에는 PPID, lot 관련 필터와
            # DPM 계산 파라미터를 함께 모아 둔다.
            with st.popover("DPM Setting", use_container_width=True):
                incoming_ppid = st.text_input(
                    "Incoming ppid",
                    value=current_inputs["incoming_ppid"],
                    key=f"in_ppid_{selected_fab_type}",
                    placeholder="Incoming ppid",
                    label_visibility="collapsed",
                )
                incoming_lot_type = st.selectbox(
                    "Incoming lot type",
                    options=LOT_TYPE_OPTIONS,
                    index=(
                        LOT_TYPE_OPTIONS.index(current_inputs["incoming_lot_type"])
                        if current_inputs["incoming_lot_type"] in LOT_TYPE_OPTIONS
                        else 0
                    ),
                    key=f"in_lot_type_{selected_fab_type}",
                    placeholder="Incoming lot type",
                    label_visibility="collapsed",
                )
                target_ppid = st.text_input(
                    "Target ppid",
                    value=current_inputs["target_ppid"],
                    key=f"tar_ppid_{selected_fab_type}",
                    placeholder="Target ppid",
                    label_visibility="collapsed",
                )
                target_lot_type = st.selectbox(
                    "Target lot type",
                    options=LOT_TYPE_OPTIONS,
                    index=(
                        LOT_TYPE_OPTIONS.index(current_inputs["target_lot_type"])
                        if current_inputs["target_lot_type"] in LOT_TYPE_OPTIONS
                        else 0
                    ),
                    key=f"tar_lot_type_{selected_fab_type}",
                    placeholder="Target lot type",
                    label_visibility="collapsed",
                )
                incoming_lot_filter = st.text_input(
                    "Incoming lot filter",
                    value=current_inputs["incoming_lot_filter"],
                    key=f"in_lot_filter_{selected_fab_type}",
                    placeholder="Incoming lot filter",
                    label_visibility="collapsed",
                )

                st.markdown("---")

                control_scope = st.selectbox(
                    "Control scope",
                    options=CONTROL_SCOPE_OPTIONS,
                    index=(
                        CONTROL_SCOPE_OPTIONS.index(current_inputs["control_scope"])
                        if current_inputs["control_scope"] in CONTROL_SCOPE_OPTIONS
                        else 0
                    ),
                    key=f"control_scope_{selected_fab_type}",
                    label_visibility="collapsed",
                )
                control_type = st.radio(
                    "Control Type",
                    ["sigma", "spec", "percentile"],
                    index=["sigma", "spec", "percentile"].index(current_inputs["control_type"]),
                    key=f"control_type_{selected_fab_type}",
                    horizontal=True,
                    label_visibility="collapsed",
                )
                sigma = st.text_input(
                    "Sigma",
                    value=current_inputs["sigma"],
                    key=f"sigma_{selected_fab_type}",
                    placeholder="Sigma",
                    label_visibility="collapsed",
                )
                spec_col_usl, spec_col_lsl = st.columns(2)
                with spec_col_usl:
                    usl = st.text_input(
                        "USL",
                        value=current_inputs["usl"],
                        key=f"usl_{selected_fab_type}",
                        placeholder="USL",
                        label_visibility="collapsed",
                    )
                with spec_col_lsl:
                    lsl = st.text_input(
                        "LSL",
                        value=current_inputs["lsl"],
                        key=f"lsl_{selected_fab_type}",
                        placeholder="LSL",
                        label_visibility="collapsed",
                    )
                percentile = st.text_input(
                    "Percentile",
                    value=current_inputs["percentile"],
                    key=f"percentile_{selected_fab_type}",
                    placeholder="Percentile",
                    label_visibility="collapsed",
                )
                wow_portion = st.text_input(
                    "WOW portion",
                    value=current_inputs["wow_portion"],
                    key=f"wow_portion_{selected_fab_type}",
                    placeholder="WOW portion",
                    label_visibility="collapsed",
                )
                window_days = st.text_input(
                    "Window days",
                    value=current_inputs["window_days"],
                    key=f"window_days_{selected_fab_type}",
                    placeholder="Window days",
                    label_visibility="collapsed",
                )
                min_wafer_qty = st.text_input(
                    "Min wafer qty",
                    value=current_inputs["min_wafer_qty"],
                    key=f"min_wafer_qty_{selected_fab_type}",
                    placeholder="Min wafer qty",
                    label_visibility="collapsed",
                )

            if control_type == "sigma":
                dpm_summary = f"sigma: {_compact_value(sigma)}"
            elif control_type == "spec":
                dpm_summary = f"spec: {_compact_value(usl)} / {_compact_value(lsl)}"
            else:
                dpm_summary = f"pct: {_compact_value(percentile)}"
            st.caption(dpm_summary)

        with col_action:
            # Run all은 raw data load부터 다시 시작하고,
            # Apply DPM setting only는 기존 raw data를 재사용한다.
            st.markdown("<div style='height: 0.2rem;'></div>", unsafe_allow_html=True)
            run_all_btn = st.button(
                "Run all - Data Load and DPM Simulation",
                key=f"btn_run_{selected_fab_type}",
                type="primary",
                use_container_width=True,
            )
            update_btn = st.button(
                "Apply new DPM setting",
                key=f"btn_upd_{selected_fab_type}",
                use_container_width=True,
            )

        # query_inputs는 현재 화면에서 사용자가 설정한 모든 값을 모은 결과다.
        # 이후 repositories / services 함수로 그대로 전달된다.
        query_inputs = {
            "fab_type": selected_fab_type,
            "line_id": line_ids,
            "process_id": process_ids,
            "incoming_prc_step": incoming_prc_step,
            "incoming_metro_step": incoming_metro_step,
            "incoming_item": incoming_item,
            "incoming_ppid": incoming_ppid,
            "incoming_lot_type": incoming_lot_type,
            "incoming_lot_filter": incoming_lot_filter,
            "target_prc_step": target_prc_step,
            "target_metro_step": target_metro_step,
            "target_item": target_item,
            "target_ppid": target_ppid,
            "target_lot_type": target_lot_type,
            "side_metro_step": side_metro_step,
            "side_metro_item": side_metro_item,
            "l2_step_seq": l2_step_seq,
            "l2_item_id": l2_item_id,
            "l3_step_seq": l3_step_seq,
            "l3_item_id": l3_item_id,
            "control_scope": control_scope,
            "from_date": from_date,
            "to_date": to_date,
            "control_type": control_type,
            "sigma": sigma,
            "usl": usl,
            "lsl": lsl,
            "percentile": percentile,
            "wow_portion": wow_portion,
            "window_days": window_days,
            "min_wafer_qty": min_wafer_qty,
        }

        # 입력값은 매 렌더링마다 세션에 저장해 두어
        # 버튼 클릭이나 rerun 이후에도 값이 유지되도록 한다.
        st.session_state["dpm_inputs"][selected_fab_type].update(query_inputs)

        if step_mapping_df.empty:
            expected_file = STEP_MAPPING_FILES.get(selected_fab_type, "step mapping csv")
            st.caption(f"Step mapping dataframe is not loaded: {expected_file}")

    if run_all_btn:
        # Run all은 repositories.get_data를 다시 호출해
        # 최신 query 기준 raw data를 새로 로드한다.
        with st.spinner("Loading raw data and starting preprocessing..."):
            raw_data = load_dpm_raw_data(query_inputs)
            df_incoming = raw_data["df_incoming"]
            df_target = raw_data["df_target"]
            df_side = raw_data["df_side"]
            df_l2 = raw_data["df_l2"]
            df_l3 = raw_data["df_l3"]
            joined_raw = join_incoming_target_for_comparison(df_incoming, df_target)
            incoming_preprocessed, target_preprocessed, calculation_result = _run_preprocessing_and_calculation(
                df_incoming,
                df_target,
                query_inputs,
                df_side=df_side,
                df_l2=df_l2,
                df_l3=df_l3,
            )

            st.session_state["dpm_loaded_data"][selected_fab_type] = {
                "query_inputs": query_inputs,
                "df_incoming": df_incoming,
                "df_target": df_target,
                "df_side": df_side,
                "df_l2": df_l2,
                "df_l3": df_l3,
                "joined_raw": joined_raw,
                "incoming_preprocessed": incoming_preprocessed,
                "target_preprocessed": target_preprocessed,
            }
            st.session_state["dpm_results"][selected_fab_type] = calculation_result
            st.success(f"{selected_fab_type.upper()} raw data load completed.")

    elif update_btn:
        # Apply DPM setting only는 이미 로드된 incoming / target raw data를 그대로 사용한다.
        # 즉, get_data는 다시 타지 않고 preprocessing부터 다시 시작한다.
        loaded_data = st.session_state["dpm_loaded_data"][selected_fab_type]
        if loaded_data is None:
            st.warning("Run all을 먼저 실행해서 raw data를 불러와 주세요.")
        else:
            with st.spinner("Re-running preprocessing and DPM calculation..."):
                df_incoming = loaded_data["df_incoming"]
                df_target = loaded_data["df_target"]
                df_side = loaded_data.get("df_side")
                df_l2 = loaded_data.get("df_l2")
                df_l3 = loaded_data.get("df_l3")
                incoming_preprocessed, target_preprocessed, calculation_result = _run_preprocessing_and_calculation(
                    df_incoming,
                    df_target,
                    query_inputs,
                    df_side=df_side,
                    df_l2=df_l2,
                    df_l3=df_l3,
                )

                # raw data는 그대로 두고, query / preprocessed / calculation 결과만 갱신한다.
                loaded_data["query_inputs"] = query_inputs
                loaded_data["incoming_preprocessed"] = incoming_preprocessed
                loaded_data["target_preprocessed"] = target_preprocessed
                st.session_state["dpm_loaded_data"][selected_fab_type] = loaded_data
                st.session_state["dpm_results"][selected_fab_type] = calculation_result
                st.success("DPM setting changes applied from preprocessing stage.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Result Section")
        loaded_data = st.session_state["dpm_loaded_data"][selected_fab_type]
        calculation_result = st.session_state["dpm_results"][selected_fab_type]

        if loaded_data is None:
            st.info("Fill the inputs above and run the simulator to load data.")
            return

        # Result Section에서는 세션에 보관된 raw / preprocessed 데이터를 꺼내 사용한다.
        # 이 구조 덕분에 Apply DPM setting only가 raw data를 재조회하지 않아도 된다.
        incoming_raw = loaded_data["df_incoming"]
        target_raw = loaded_data["df_target"]
        # incoming/target 개별 trend는 get_data에서 받은 raw dataframe 그대로 그린다.
        # comparison용 outlier 제거는 preprocessing 서비스 내부 함수가 담당한다.
        incoming_corr_base, target_corr_base = preprocess_comparison_trend_data(
            incoming_raw,
            target_raw,
        )
        joined_corr = join_incoming_target_for_comparison(incoming_corr_base, target_corr_base)
        incoming_preprocessed = loaded_data["incoming_preprocessed"]
        target_preprocessed = loaded_data["target_preprocessed"]

        legend_options = _get_legend_options(incoming_raw, target_raw)
        joined_legend_options = _get_joined_legend_options(joined_corr)

        # 사용자가 scatterplot의 color 기준을 바꿔볼 수 있도록 legend selector를 제공한다.
        control_col_left, control_col_right = st.columns([1, 1])
        with control_col_left:
            trend_legend = st.selectbox(
                "Trend legend",
                options=legend_options,
                key=f"trend_legend_{selected_fab_type}",
            )
        with control_col_right:
            comparison_legend = st.selectbox(
                "Comparison legend",
                options=joined_legend_options,
                key=f"comparison_legend_{selected_fab_type}",
            )

        # 첫 줄은 Incoming -> Target -> Comparison 순서로 배치한다.
        plot_col_in, plot_col_tar, plot_col_cmp = st.columns(3)
        with plot_col_in:
            incoming_fig = build_trend_scatter(
                incoming_raw,
                title="Incoming Metro Trend",
                legend_column=trend_legend,
            )
            st.plotly_chart(incoming_fig, use_container_width=True)
            st.caption(f"rows: {len(incoming_raw)}")

        with plot_col_tar:
            target_fig = build_trend_scatter(
                target_raw,
                title="Target Metro Trend",
                legend_column=trend_legend,
            )
            st.plotly_chart(target_fig, use_container_width=True)
            st.caption(f"rows: {len(target_raw)}")

        with plot_col_cmp:
            comparison_fig, r_squared = build_comparison_scatter_with_regression(
                joined_corr,
                legend_column=comparison_legend,
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            st.caption(
                f"joined rows: {len(joined_corr)} | Pearson R^2: "
                f"{'-' if r_squared is None else f'{r_squared:.4f}'}"
            )

        # 제어 결과 섹션은 dpm_calculation.py에서 판정된 outlier 결과를 사용한다.
        # Lot 단위는 lot_mean_fab_value, Wafer 단위는 fab_value를 y축으로 사용한다.
        lot_eqp_df = calculation_result.get("lot_eqp_df") if calculation_result else pd.DataFrame()
        lot_chamber_df = calculation_result.get("lot_chamber_df") if calculation_result else pd.DataFrame()
        wafer_chamber_df = calculation_result.get("wafer_chamber_df") if calculation_result else pd.DataFrame()

        control_sections = [
            ("Lot - EQP 제어", lot_eqp_df, "lot_mean_fab_value"),
            ("Lot - Chamber 제어", lot_chamber_df, "lot_mean_fab_value"),
            ("Wafer - Chamber 제어", wafer_chamber_df, "fab_value"),
        ]
        for section_title, section_df, y_col in control_sections:
            with st.container(border=True):
                st.markdown(f"**{section_title}**")
                section_plot_col, section_info_col = st.columns([2.2, 1])
                with section_plot_col:
                    section_fig = build_outlier_scatter(
                        section_df,
                        x_col="tkout_time",
                        y_col=y_col,
                        title=section_title,
                    )
                    st.plotly_chart(section_fig, use_container_width=True)
                with section_info_col:
                    if section_df is None or section_df.empty:
                        st.caption("판정 결과가 없습니다.")
                    else:
                        outlier_counts = (
                            section_df["outlier_status"].value_counts()
                            if "outlier_status" in section_df.columns
                            else pd.Series(dtype=int)
                        )
                        st.metric("Total", len(section_df))
                        st.metric("Upper Out", int(outlier_counts.get("upper_out", 0)))
                        st.metric("Lower Out", int(outlier_counts.get("lower_out", 0)))
