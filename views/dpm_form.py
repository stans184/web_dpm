from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from models.dpm_models import CONTROL_SCOPE_OPTIONS, INCOMING_SUBITEM_OPTIONS, LOT_TYPE_OPTIONS
from services.step_mapping import (
    STEP_MAPPING_FILES,
    get_prc_step_candidates,
    get_side_metro_candidates,
    load_step_mapping_df_for_fab,
)
from views.dpm_shared import (
    build_query_inputs,
    compact_value,
    line_ids_to_text,
    merge_current_inputs,
    normalize_line_ids,
    normalize_process_ids,
    process_ids_to_text,
)


@dataclass(slots=True)
class DpmFormState:
    selected_fab_type: str
    query_inputs: dict
    run_all_btn: bool
    update_btn: bool
    step_mapping_df: pd.DataFrame


def render_input_section() -> DpmFormState:
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

        current_inputs = merge_current_inputs(selected_fab_type)
        with top_col_line:
            line_id = st.text_input(
                "Line ID",
                value=line_ids_to_text(current_inputs.get("line_id")),
                key=f"line_id_{selected_fab_type}",
                placeholder="Line ID ex) KFBN,P3DF",
                label_visibility="collapsed",
            )
        line_ids = normalize_line_ids(line_id)

        with top_col_process:
            process_id = st.text_input(
                "Process ID",
                value=process_ids_to_text(current_inputs.get("process_id")),
                key=f"process_id_{selected_fab_type}",
                placeholder="Process ID ex) PRC_01,PRC_02",
                label_visibility="collapsed",
            )
        process_ids = normalize_process_ids(process_id)
        step_mapping_df = load_step_mapping_df_for_fab(selected_fab_type)

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

        incoming_prc_candidates = get_prc_step_candidates(
            step_mapping_df,
            current_inputs["incoming_metro_step"],
            line_ids,
            process_ids,
        )
        if current_inputs["incoming_prc_step"] not in incoming_prc_candidates:
            current_inputs["incoming_prc_step"] = None

        target_prc_candidates = get_prc_step_candidates(
            step_mapping_df,
            current_inputs["target_metro_step"],
            line_ids,
            process_ids,
        )
        if current_inputs["target_prc_step"] not in target_prc_candidates:
            current_inputs["target_prc_step"] = None

        side_metro_candidates = get_side_metro_candidates(
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
                refreshed_incoming_candidates = get_prc_step_candidates(
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
                f"{compact_value(incoming_prc_step)} | metro: {compact_value(incoming_metro_step)} | "
                f"item: {compact_value(incoming_item)}"
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
                refreshed_target_candidates = get_prc_step_candidates(
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
                f"{compact_value(target_prc_step)} | metro: {compact_value(target_metro_step)} | "
                f"{compact_value(from_date)} ~ {compact_value(to_date)}"
            )

        with col_side:
            with st.popover("Side Effect", use_container_width=True):
                refreshed_side_candidates = get_side_metro_candidates(
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
            st.caption(f"metro: {compact_value(side_metro_step)} | item: {compact_value(side_metro_item)}")

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
                f"L2: {compact_value(l2_step_seq)} / {compact_value(l2_item_id)} | "
                f"L3: {compact_value(l3_step_seq)} / {compact_value(l3_item_id)}"
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
                dpm_summary = f"sigma: {compact_value(sigma)}"
            elif control_type == "spec":
                dpm_summary = f"spec: {compact_value(usl)} / {compact_value(lsl)}"
            else:
                dpm_summary = f"pct: {compact_value(percentile)}"
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

        query_inputs = build_query_inputs(
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

    return DpmFormState(
        selected_fab_type=selected_fab_type,
        query_inputs=query_inputs,
        run_all_btn=run_all_btn,
        update_btn=update_btn,
        step_mapping_df=step_mapping_df,
    )
