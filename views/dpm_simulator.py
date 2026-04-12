"""DPM Simulator 메인 화면을 구성하는 Streamlit 뷰 모듈."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components.download_button import render_download_button
from models.dpm_models import (
    CONTROL_SCOPE_OPTIONS,
    INCOMING_SUBITEM_OPTIONS,
    LOT_TYPE_OPTIONS,
    DpmInputModel,
)
from repositories.get_data import load_dpm_raw_data
from services.dpm_calculation import run_dpm_calculation
from services.preprocessing import preprocess_comparison_trend_data, preprocess_dpm_input_data
from services.visualization import (
    build_comparison_scatter_with_regression,
    build_judged_trend_scatter,
    build_outlier_scatter,
    build_target_control_boxplot,
    build_trend_scatter,
    join_incoming_target_for_comparison,
)


STEP_MAPPING_FILES = {
    "memory": "mem_prc_with_met.csv",
    "foundry": "fdry_prc_with_met.csv",
}


def _default_inputs(fab_type: str) -> dict:
    """뷰가 사용할 기본 입력값을 공통 입력 모델에서 생성한다."""
    return DpmInputModel(fab_type=fab_type).to_dict()


def _compact_value(value) -> str:
    """캡션에서 비어 있는 값을 보기 좋은 문자로 치환한다."""
    if value in (None, ""):
        return "-"
    return str(value)


def _normalize_step_tokens(raw_value: str) -> list[str]:
    """여러 구분자를 허용해 step 입력을 소문자 토큰 목록으로 정규화한다."""
    if not raw_value:
        return []
    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip().lower() for token in normalized.split(",") if token.strip()]


def _normalize_line_ids(raw_value: str | list[str] | None) -> list[str]:
    """문자열 또는 목록 형태의 line id 입력을 동일한 목록 구조로 맞춘다."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]
    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _line_ids_to_text(line_ids: str | list[str] | None) -> str:
    """세션에 저장된 line id 목록을 텍스트 입력용 문자열로 변환한다."""
    return ", ".join(_normalize_line_ids(line_ids))


def _normalize_process_ids(raw_value: str | list[str] | None) -> list[str]:
    """문자열 또는 목록 형태의 process id 입력을 동일한 목록 구조로 맞춘다."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]
    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _process_ids_to_text(process_ids: str | list[str] | None) -> str:
    """세션에 저장된 process id 목록을 텍스트 입력용 문자열로 변환한다."""
    return ", ".join(_normalize_process_ids(process_ids))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """후보 컬럼명 목록 중 실제 데이터프레임에 존재하는 첫 컬럼을 찾는다."""
    normalized_columns = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def _load_step_mapping_df_for_fab(fab_type: str) -> pd.DataFrame:
    """선택된 fab 유형에 맞는 step mapping CSV를 불러온다."""
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
    """metro step, line, process 조건으로 선택 가능한 prc step 후보를 만든다."""
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
        normalized_process_ids = [
            process_id.strip().lower() for process_id in (process_ids or []) if process_id.strip()
        ]
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
    """target prc step 기준으로 side effect용 metro step 후보를 만든다."""
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
        normalized_process_ids = [
            process_id.strip().lower() for process_id in (process_ids or []) if process_id.strip()
        ]
        if normalized_process_ids:
            working_df = working_df[working_df[process_col].str.lower().isin(normalized_process_ids)]

    return sorted(working_df[metro_col].drop_duplicates().tolist())


def _get_legend_options(*dfs: pd.DataFrame) -> list[str]:
    """원본 trend 산점도에서 사용할 legend 후보 컬럼을 반환한다."""
    candidate_columns = ["ppid", "lot_type", "item_id", "root_lot_id", "lot_id"]
    options: list[str] = []
    for col in candidate_columns:
        if any(df is not None and not df.empty and col in df.columns for df in dfs):
            options.append(col)
    return options or ["ppid"]


def _get_joined_legend_options(joined_df: pd.DataFrame) -> list[str]:
    """incoming/target 조인 결과에서 사용할 legend 후보 컬럼을 반환한다."""
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
    """전처리와 DPM 계산을 한 번에 실행한다."""
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


def _build_dpm_export_df(control_sections: list[tuple[str, dict | None]]) -> pd.DataFrame:
    """제어 결과를 하나의 다운로드용 데이터프레임으로 합친다."""
    export_frames: list[pd.DataFrame] = []
    for section_title, section_result in control_sections:
        if not section_result:
            continue

        section_df = section_result.get("controlled_trend_df", pd.DataFrame())
        if section_df is None or section_df.empty:
            continue

        export_df = section_df.copy()
        export_df.insert(0, "control_section", section_title)
        export_frames.append(export_df)

    if not export_frames:
        return pd.DataFrame()
    return pd.concat(export_frames, ignore_index=True)


def _merge_current_inputs(selected_fab_type: str) -> dict:
    """세션에 저장된 입력값을 최신 입력 모델 스키마와 동기화한다."""
    merged_inputs = {
        **_default_inputs(selected_fab_type),
        **st.session_state["dpm_inputs"][selected_fab_type],
    }
    st.session_state["dpm_inputs"][selected_fab_type] = merged_inputs
    return merged_inputs


def _resolve_control_scope_mode(control_scope: str) -> str:
    """제어 범위 라벨을 계산용 모드 문자열로 변환한다."""
    if control_scope == CONTROL_SCOPE_OPTIONS[0]:
        return "mesh"
    if control_scope == CONTROL_SCOPE_OPTIONS[1]:
        return "small"
    return "large"


def _build_query_inputs(
    *,
    selected_fab_type: str,
    line_ids: list[str],
    process_ids: list[str],
    incoming_prc_step,
    incoming_metro_step,
    incoming_item,
    incoming_ppid,
    incoming_lot_type,
    incoming_lot_filter,
    incoming_subitem,
    target_prc_step,
    target_metro_step,
    target_item,
    target_ppid,
    target_lot_type,
    side_metro_step,
    side_metro_item,
    l2_step_seq,
    l2_item_id,
    l3_step_seq,
    l3_item_id,
    control_scope,
    from_date,
    to_date,
    control_type,
    sigma,
    usl,
    lsl,
    percentile,
    wow_portion,
    window_days,
    min_wafer_qty,
) -> dict:
    """뷰 입력값을 계산 서비스가 사용하는 공통 입력 모델 구조로 만든다."""
    return DpmInputModel(
        fab_type=selected_fab_type,
        line_id=line_ids,
        process_id=process_ids,
        incoming_prc_step=incoming_prc_step,
        incoming_metro_step=incoming_metro_step,
        incoming_item=incoming_item,
        incoming_ppid=incoming_ppid,
        incoming_lot_type=incoming_lot_type,
        incoming_lot_filter=incoming_lot_filter,
        incoming_subitem=incoming_subitem,
        target_prc_step=target_prc_step,
        target_metro_step=target_metro_step,
        target_item=target_item,
        target_ppid=target_ppid,
        target_lot_type=target_lot_type,
        side_metro_step=side_metro_step,
        side_metro_item=side_metro_item,
        l2_step_seq=l2_step_seq,
        l2_item_id=l2_item_id,
        l3_step_seq=l3_step_seq,
        l3_item_id=l3_item_id,
        control_scope=control_scope,
        control_scope_mode=_resolve_control_scope_mode(control_scope),
        from_date=from_date,
        to_date=to_date,
        control_type=control_type,
        sigma=sigma,
        usl=usl,
        lsl=lsl,
        percentile=percentile,
        wow_portion=wow_portion,
        window_days=window_days,
        min_wafer_qty=min_wafer_qty,
    ).to_dict()


def _render_result_sections(selected_fab_type: str):
    """계산 결과와 다운로드 영역을 한 곳에서 렌더링한다."""
    loaded_data = st.session_state["dpm_loaded_data"][selected_fab_type]
    calculation_result = st.session_state["dpm_results"][selected_fab_type]

    if loaded_data is None:
        st.info("입력값을 설정한 뒤 시뮬레이터를 실행해 주세요.")
        return

    incoming_raw = loaded_data["df_incoming"]
    target_raw = loaded_data["df_target"]
    incoming_corr_base, target_corr_base = preprocess_comparison_trend_data(incoming_raw, target_raw)
    joined_corr = join_incoming_target_for_comparison(incoming_corr_base, target_corr_base)

    legend_options = _get_legend_options(incoming_raw, target_raw)
    joined_legend_options = _get_joined_legend_options(joined_corr)

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

    plot_col_in, plot_col_tar, plot_col_cmp = st.columns(3)
    with plot_col_in:
        st.plotly_chart(
            build_trend_scatter(incoming_raw, title="Incoming Metro Trend", legend_column=trend_legend),
            use_container_width=True,
        )
        st.caption(f"rows: {len(incoming_raw)}")

    with plot_col_tar:
        st.plotly_chart(
            build_trend_scatter(target_raw, title="Target Metro Trend", legend_column=trend_legend),
            use_container_width=True,
        )
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

    control_sections = [
        ("Lot - EQP 제어", calculation_result.get("lot_eqp_result") if calculation_result else None),
        ("Lot - Chamber 제어", calculation_result.get("lot_chamber_result") if calculation_result else None),
        ("Wafer - Chamber 제어", calculation_result.get("wafer_chamber_result") if calculation_result else None),
    ]
    target_color_map = {"normal": "lightgrey", "BOB": "green", "WOW": "red"}
    controlled_color_map = {"normal": "lightgrey", "BOB": "green", "DPM_Controlled": "red"}

    for section_title, section_result in control_sections:
        with st.container(border=True):
            st.markdown(f"**{section_title}**")
            if not section_result:
                st.info("계산 결과가 아직 없습니다.")
                continue

            incoming_outlier_df = section_result.get("incoming_outlier_df", pd.DataFrame())
            target_trend_df = section_result.get("target_trend_df", pd.DataFrame())
            controlled_trend_df = section_result.get("controlled_trend_df", pd.DataFrame())
            boxplot_df = section_result.get("boxplot_df", pd.DataFrame())
            boxplot_stats_df = section_result.get("boxplot_stats_df", pd.DataFrame())
            value_col = section_result.get("value_col", "fab_value")

            col_incoming, col_target, col_controlled, col_box = st.columns(4)
            with col_incoming:
                st.plotly_chart(
                    build_outlier_scatter(
                        incoming_outlier_df,
                        x_col="tkout_time",
                        y_col=value_col,
                        title=f"{section_title} - Incoming Outlier",
                    ),
                    use_container_width=True,
                )
            with col_target:
                st.plotly_chart(
                    build_judged_trend_scatter(
                        target_trend_df,
                        x_col="tkout_time",
                        y_col="fab_value",
                        judge_col="dpm_judge",
                        title=f"{section_title} - Target Trend",
                        color_map=target_color_map,
                        category_order=["normal", "BOB", "WOW"],
                    ),
                    use_container_width=True,
                )
            with col_controlled:
                st.plotly_chart(
                    build_judged_trend_scatter(
                        controlled_trend_df,
                        x_col="tkout_time",
                        y_col="dpm_controlled_value",
                        judge_col="dpm_controlled_judge",
                        title=f"{section_title} - DPM Controlled Trend",
                        color_map=controlled_color_map,
                        category_order=["normal", "BOB", "DPM_Controlled"],
                    ),
                    use_container_width=True,
                )
            with col_box:
                st.plotly_chart(
                    build_target_control_boxplot(boxplot_df, title=f"{section_title} - Box Plot"),
                    use_container_width=True,
                )
                if boxplot_stats_df is None or boxplot_stats_df.empty:
                    st.caption("통계 결과가 없습니다.")
                else:
                    st.dataframe(boxplot_stats_df, use_container_width=True, hide_index=True)

    export_df = _build_dpm_export_df(control_sections)
    st.markdown("#### Raw Data Download")
    render_download_button(
        export_df,
        filename=f"dpm_control_result_{selected_fab_type}.csv",
        label="Download DPM Raw Data",
    )


def render_dpm_simulator():
    """DPM Simulator 메인 화면을 그린다."""
    st.markdown("### DPM Simulator")

    with st.container(border=True):
        st.markdown("#### Input Section")

        top_col_fab, top_col_line, top_col_process, _ = st.columns([1.1, 1.2, 1.2, 4.5])
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

        current_inputs = _merge_current_inputs(selected_fab_type)
        with top_col_line:
            line_id = st.text_input(
                "Line ID",
                value=_line_ids_to_text(current_inputs.get("line_id")),
                key=f"line_id_{selected_fab_type}",
                placeholder="Line ID ex) KFBN,P3DF",
                label_visibility="collapsed",
            )
        line_ids = _normalize_line_ids(line_id)

        with top_col_process:
            process_id = st.text_input(
                "Process ID",
                value=_process_ids_to_text(current_inputs.get("process_id")),
                key=f"process_id_{selected_fab_type}",
                placeholder="Process ID ex) PRC_01,PRC_02",
                label_visibility="collapsed",
            )
        process_ids = _normalize_process_ids(process_id)
        step_mapping_df = _load_step_mapping_df_for_fab(selected_fab_type)

        st.markdown("---")
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

        incoming_prc_candidates = _get_prc_step_candidates(
            step_mapping_df,
            current_inputs["incoming_metro_step"],
            line_ids,
            process_ids,
        )
        if current_inputs["incoming_prc_step"] not in incoming_prc_candidates:
            current_inputs["incoming_prc_step"] = None

        target_prc_candidates = _get_prc_step_candidates(
            step_mapping_df,
            current_inputs["target_metro_step"],
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

        col_incoming, col_target, col_side, col_l2l3, col_dpm, col_action = st.columns([1, 1, 1, 1, 1, 0.95])

        with col_incoming:
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
            st.caption(f"metro: {_compact_value(side_metro_step)} | item: {_compact_value(side_metro_item)}")

        with col_l2l3:
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
                    index=LOT_TYPE_OPTIONS.index(current_inputs["incoming_lot_type"])
                    if current_inputs["incoming_lot_type"] in LOT_TYPE_OPTIONS
                    else 0,
                    key=f"in_lot_type_{selected_fab_type}",
                    placeholder="Incoming lot type",
                )
                incoming_lot_filter = st.text_input(
                    "Incoming lot filter",
                    value=current_inputs["incoming_lot_filter"],
                    key=f"in_lot_filter_{selected_fab_type}",
                    placeholder="Incoming lot filter",
                    label_visibility="collapsed",
                )
                incoming_subitem = st.selectbox(
                    "Incoming subitem",
                    options=INCOMING_SUBITEM_OPTIONS,
                    index=INCOMING_SUBITEM_OPTIONS.index(current_inputs["incoming_subitem"])
                    if current_inputs["incoming_subitem"] in INCOMING_SUBITEM_OPTIONS
                    else 0,
                    key=f"in_subitem_{selected_fab_type}",
                    placeholder="Incoming subitem",
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
                    index=LOT_TYPE_OPTIONS.index(current_inputs["target_lot_type"])
                    if current_inputs["target_lot_type"] in LOT_TYPE_OPTIONS
                    else 0,
                    key=f"tar_lot_type_{selected_fab_type}",
                    placeholder="Target lot type",
                )

                st.markdown("---")

                control_scope = st.selectbox(
                    "Control scope",
                    options=CONTROL_SCOPE_OPTIONS,
                    index=CONTROL_SCOPE_OPTIONS.index(current_inputs["control_scope"])
                    if current_inputs["control_scope"] in CONTROL_SCOPE_OPTIONS
                    else 0,
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

        query_inputs = _build_query_inputs(
            selected_fab_type=selected_fab_type,
            line_ids=line_ids,
            process_ids=process_ids,
            incoming_prc_step=incoming_prc_step,
            incoming_metro_step=incoming_metro_step,
            incoming_item=incoming_item,
            incoming_ppid=incoming_ppid,
            incoming_lot_type=incoming_lot_type,
            incoming_lot_filter=incoming_lot_filter,
            incoming_subitem=incoming_subitem,
            target_prc_step=target_prc_step,
            target_metro_step=target_metro_step,
            target_item=target_item,
            target_ppid=target_ppid,
            target_lot_type=target_lot_type,
            side_metro_step=side_metro_step,
            side_metro_item=side_metro_item,
            l2_step_seq=l2_step_seq,
            l2_item_id=l2_item_id,
            l3_step_seq=l3_step_seq,
            l3_item_id=l3_item_id,
            control_scope=control_scope,
            from_date=from_date,
            to_date=to_date,
            control_type=control_type,
            sigma=sigma,
            usl=usl,
            lsl=lsl,
            percentile=percentile,
            wow_portion=wow_portion,
            window_days=window_days,
            min_wafer_qty=min_wafer_qty,
        )
        st.session_state["dpm_inputs"][selected_fab_type] = query_inputs

        if step_mapping_df.empty:
            expected_file = STEP_MAPPING_FILES.get(selected_fab_type, "step mapping csv")
            st.caption(f"Step mapping dataframe is not loaded: {expected_file}")

    if run_all_btn:
        with st.spinner("Loading raw data and starting preprocessing..."):
            raw_data = load_dpm_raw_data(query_inputs)
            df_incoming = raw_data["df_incoming"]
            df_target = raw_data["df_target"]
            df_side = raw_data["df_side"]
            df_l2 = raw_data["df_l2"]
            df_l3 = raw_data["df_l3"]
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
                "incoming_preprocessed": incoming_preprocessed,
                "target_preprocessed": target_preprocessed,
            }
            st.session_state["dpm_results"][selected_fab_type] = calculation_result
            st.success(f"{selected_fab_type.upper()} raw data load completed.")

    elif update_btn:
        loaded_data = st.session_state["dpm_loaded_data"][selected_fab_type]
        if loaded_data is None:
            st.warning("먼저 Run all을 실행해 raw data를 불러와 주세요.")
        else:
            with st.spinner("Re-running preprocessing and DPM calculation..."):
                incoming_preprocessed, target_preprocessed, calculation_result = _run_preprocessing_and_calculation(
                    loaded_data["df_incoming"],
                    loaded_data["df_target"],
                    query_inputs,
                    df_side=loaded_data.get("df_side"),
                    df_l2=loaded_data.get("df_l2"),
                    df_l3=loaded_data.get("df_l3"),
                )
                loaded_data["query_inputs"] = query_inputs
                loaded_data["incoming_preprocessed"] = incoming_preprocessed
                loaded_data["target_preprocessed"] = target_preprocessed
                st.session_state["dpm_loaded_data"][selected_fab_type] = loaded_data
                st.session_state["dpm_results"][selected_fab_type] = calculation_result
                st.success("DPM setting changes applied from preprocessing stage.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Result Section")
        _render_result_sections(selected_fab_type)
