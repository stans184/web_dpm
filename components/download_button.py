import pandas as pd
import streamlit as st

from utils.export_file import convert_df_to_csv


# DataFrame을 CSV로 내려받는 공통 버튼 컴포넌트다.
# Raw / Preprocessed 결과 테이블에서 동일한 UI를 재사용하기 위해 분리했다.


def render_download_button(
    data: pd.DataFrame,
    filename: str = "data_export.csv",
    label: str = "결과 다운로드",
):
    """
    DataFrame이 비어있지 않으면 Streamlit download_button을 렌더링한다.
    """
    if data is None or data.empty:
        st.warning("다운로드할 결과 데이터가 없습니다.")
        return

    csv_data = convert_df_to_csv(data)

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=f"btn_dl_{filename}",
    )
