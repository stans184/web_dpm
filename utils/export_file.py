"""내보내기 파일 관련 공통 유틸리티 모듈."""

import pandas as pd


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """데이터프레임을 UTF-8 BOM이 포함된 CSV 바이트로 변환한다."""
    return df.to_csv(index=False).encode("utf-8-sig")
