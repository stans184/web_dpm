from pathlib import Path

import pandas as pd
import streamlit as st


# session_manager는 앱 시작 시 필요한 session_state 구조를 한 번에 초기화한다.
# DPM Simulator에서 사용하는 입력값, raw/preprocessed 결과, step mapping dataframe이
# 모두 여기서 준비된다.


def _load_step_mapping_df() -> pd.DataFrame:
    # 세션 시작 시 metro step -> prc step 후보를 찾기 위해 사용하는
    # step mapping dataframe을 로드한다.
    data_path = Path(__file__).resolve().parents[1] / "dummy_semiconductor_data_step2.csv"
    try:
        return pd.read_csv(data_path)
    except Exception:
        return pd.DataFrame()


def init_user_session():
    """
    Streamlit session_state의 기본 구조를 초기화한다.
    """
    if "dpm_inputs" not in st.session_state:
        # memory / foundry를 분리해서 각각 독립적인 입력값을 유지한다.
        st.session_state["dpm_inputs"] = {
            "memory": {
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
            },
            "foundry": {
                "fab_type": "foundry",
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
            },
        }

    if "dpm_results" not in st.session_state:
        # dpm_results는 현재 단계에서는 dpm_calculation의 반환값을 저장한다.
        st.session_state["dpm_results"] = {
            "memory": None,
            "foundry": None,
        }

    if "dpm_loaded_data" not in st.session_state:
        # dpm_loaded_data는 raw data와 preprocessed data를 함께 보관한다.
        # Apply DPM setting only에서는 여기 저장된 raw data를 재사용한다.
        st.session_state["dpm_loaded_data"] = {
            "memory": None,
            "foundry": None,
        }

    if "current_fab_type" not in st.session_state:
        # 현재 화면에서 선택된 fab type을 기억해 rerun 후에도 유지한다.
        st.session_state["current_fab_type"] = "memory"

    if "dpm_step_mapping_df" not in st.session_state:
        # prc step 후보 selectbox를 만들기 위한 매핑 dataframe이다.
        st.session_state["dpm_step_mapping_df"] = _load_step_mapping_df()
