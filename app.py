import streamlit as st

from states.session_manager import init_user_session
from views.dpm_simulator import render_dpm_simulator


# app.py는 Streamlit 앱의 진입점이다.
# 페이지 공통 설정, session 초기화, 최상위 탭 구성을 담당한다.
# What's wrong


# 페이지 기본 설정을 먼저 지정한다.
st.set_page_config(
    page_title="MIDEA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 앱이 시작될 때 필요한 session_state를 준비한다.
init_user_session()

# Streamlit의 기본 사이드바 토글 버튼을 숨겨 화면을 더 깔끔하게 유지한다.
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # 현재는 DPM Simulator 탭과 placeholder 탭 2개로 구성되어 있다.
    tab_simulator, tab_other = st.tabs(["DPM Simulator", "Other Page"])

    with tab_simulator:
        render_dpm_simulator()

    with tab_other:
        st.markdown("### Other Page")
        st.info("추후 다른 기능이 들어올 수 있는 영역입니다.")


if __name__ == "__main__":
    main()
