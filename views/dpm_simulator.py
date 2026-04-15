"""DPM Simulator main page composition."""

from __future__ import annotations

import streamlit as st

from repositories.get_data import load_dpm_raw_data
from views.dpm_form import render_input_section
from views.dpm_results import render_result_sections
from views.dpm_shared import run_preprocessing_and_calculation


def render_dpm_simulator():
    form_state = render_input_section()
    selected_fab_type = form_state.selected_fab_type
    query_inputs = form_state.query_inputs

    if form_state.run_all_btn:
        with st.spinner("Loading raw data and starting preprocessing..."):
            raw_data = load_dpm_raw_data(query_inputs)
            df_incoming = raw_data["df_incoming"]
            df_target = raw_data["df_target"]
            df_side = raw_data["df_side"]
            df_l2 = raw_data["df_l2"]
            df_l3 = raw_data["df_l3"]
            incoming_preprocessed, target_preprocessed, calculation_result = run_preprocessing_and_calculation(
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

    elif form_state.update_btn:
        loaded_data = st.session_state["dpm_loaded_data"][selected_fab_type]
        if loaded_data is None:
            st.warning("먼저 Run all을 실행해 raw data를 불러와 주세요.")
        else:
            with st.spinner("Re-running preprocessing and DPM calculation..."):
                incoming_preprocessed, target_preprocessed, calculation_result = run_preprocessing_and_calculation(
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
        render_result_sections(selected_fab_type)
