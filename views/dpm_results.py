from __future__ import annotations

import pandas as pd
import streamlit as st

from components.download_button import render_download_button
from services.preprocessing import preprocess_comparison_trend_data
from services.visualization import (
    build_comparison_scatter_with_regression,
    build_judged_trend_scatter,
    build_outlier_scatter,
    build_target_control_boxplot,
    build_trend_scatter,
    join_incoming_target_for_comparison,
)


def get_legend_options(*dfs: pd.DataFrame) -> list[str]:
    candidate_columns = ["ppid", "lot_type", "item_id", "root_lot_id", "lot_id"]
    options: list[str] = []
    for col in candidate_columns:
        if any(df is not None and not df.empty and col in df.columns for df in dfs):
            options.append(col)
    return options or ["ppid"]


def get_joined_legend_options(joined_df: pd.DataFrame) -> list[str]:
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


def build_dpm_export_df(control_sections: list[tuple[str, dict | None]]) -> pd.DataFrame:
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


def render_result_sections(selected_fab_type: str):
    loaded_data = st.session_state["dpm_loaded_data"][selected_fab_type]
    calculation_result = st.session_state["dpm_results"][selected_fab_type]

    if loaded_data is None:
        st.info("입력값을 설정한 뒤 시뮬레이션을 실행해 주세요.")
        return

    incoming_raw = loaded_data["df_incoming"]
    target_raw = loaded_data["df_target"]
    incoming_corr_base, target_corr_base = preprocess_comparison_trend_data(incoming_raw, target_raw)
    joined_corr = join_incoming_target_for_comparison(incoming_corr_base, target_corr_base)

    legend_options = get_legend_options(incoming_raw, target_raw)
    joined_legend_options = get_joined_legend_options(joined_corr)

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

    export_df = build_dpm_export_df(control_sections)
    st.markdown("#### Raw Data Download")
    render_download_button(
        export_df,
        filename=f"dpm_control_result_{selected_fab_type}.csv",
        label="Download DPM Raw Data",
    )
