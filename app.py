"""DPM 시뮬레이터 웹 앱의 진입점 모듈."""

import streamlit as st

from states.session_manager import init_user_session
from views.dpm_simulator import render_dpm_simulator


st.set_page_config(
    page_title="MIDEA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_user_session()

st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """상위 탭 구조를 그리고 주요 화면을 연결한다."""
    tab_simulator, tab_other = st.tabs(["DPM Simulator", "Other Page"])

    with tab_simulator:
        render_dpm_simulator()

    with tab_other:
        st.markdown("### Other Page")
        st.info("추후 기능을 추가할 수 있는 영역입니다.")


if __name__ == "__main__":
    main()
