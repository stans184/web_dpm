from __future__ import annotations

import pandas as pd
import streamlit as st

from models.dpm_models import CONTROL_SCOPE_OPTIONS, DpmInputModel
from services.dpm_calculation import run_dpm_calculation
from services.preprocessing import preprocess_dpm_input_data


def default_inputs(fab_type: str) -> dict:
    return DpmInputModel(fab_type=fab_type).to_dict()


def compact_value(value) -> str:
    if value in (None, ""):
        return "-"
    return str(value)


def normalize_line_ids(raw_value: str | list[str] | None) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]
    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def line_ids_to_text(line_ids: str | list[str] | None) -> str:
    return ", ".join(normalize_line_ids(line_ids))


def normalize_process_ids(raw_value: str | list[str] | None) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token).strip() for token in raw_value if str(token).strip()]
    normalized = str(raw_value).replace("\n", ",").replace(";", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def process_ids_to_text(process_ids: str | list[str] | None) -> str:
    return ", ".join(normalize_process_ids(process_ids))


def merge_current_inputs(selected_fab_type: str) -> dict:
    merged_inputs = {
        **default_inputs(selected_fab_type),
        **st.session_state["dpm_inputs"][selected_fab_type],
    }
    st.session_state["dpm_inputs"][selected_fab_type] = merged_inputs
    return merged_inputs


def resolve_control_scope_mode(control_scope: str) -> str:
    if control_scope == CONTROL_SCOPE_OPTIONS[0]:
        return "mesh"
    if control_scope == CONTROL_SCOPE_OPTIONS[1]:
        return "small"
    return "large"


def build_query_inputs(
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
        control_scope_mode=resolve_control_scope_mode(control_scope),
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


def run_preprocessing_and_calculation(
    df_incoming: pd.DataFrame,
    df_target: pd.DataFrame,
    query_inputs: dict,
    df_side: pd.DataFrame | None = None,
    df_l2: pd.DataFrame | None = None,
    df_l3: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
