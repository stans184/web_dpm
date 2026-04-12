"""Streamlit 세션 상태를 초기화하는 도우미 모듈."""

from pathlib import Path

import pandas as pd
import streamlit as st

from models.dpm_models import DpmInputModel


def _load_step_mapping_df() -> pd.DataFrame:
    """앱 시작 시 사용할 기본 step mapping 데이터를 불러온다."""
    data_path = Path(__file__).resolve().parents[1] / "repositories" / "dummy_semiconductor_data_step2.csv"
    try:
        return pd.read_csv(data_path)
    except Exception:
        return pd.DataFrame()


def init_user_session():
    """시뮬레이터가 기대하는 세션 키를 처음 한 번만 만든다."""
    if "dpm_inputs" not in st.session_state:
        # Fab 유형별 입력 상태를 분리해 두면 화면을 전환해도 이전 입력값이 유지된다.
        st.session_state["dpm_inputs"] = {
            "memory": DpmInputModel(fab_type="memory").to_dict(),
            "foundry": DpmInputModel(fab_type="foundry").to_dict(),
        }

    if "dpm_results" not in st.session_state:
        # 계산 결과는 fab 유형별로 마지막 상태를 보관한다.
        st.session_state["dpm_results"] = {
            "memory": None,
            "foundry": None,
        }

    if "dpm_loaded_data" not in st.session_state:
        # 원본 데이터는 재계산 버튼이 다시 사용할 수 있도록 따로 저장한다.
        st.session_state["dpm_loaded_data"] = {
            "memory": None,
            "foundry": None,
        }

    if "current_fab_type" not in st.session_state:
        st.session_state["current_fab_type"] = "memory"

    if "dpm_step_mapping_df" not in st.session_state:
        # 기존 코드와의 호환을 위해 step mapping 데이터도 세션에 유지한다.
        st.session_state["dpm_step_mapping_df"] = _load_step_mapping_df()
