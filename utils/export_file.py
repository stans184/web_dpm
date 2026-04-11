import pandas as pd


# export_file 유틸은 DataFrame을 다운로드 가능한 파일 형태로 바꿔주는
# 가장 작은 단위의 공통 함수만 모아 둔다.


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    DataFrame을 UTF-8 BOM이 포함된 CSV bytes로 변환한다.
    Excel 환경에서 한글이 깨지지 않도록 utf-8-sig 인코딩을 사용한다.
    """
    return df.to_csv(index=False).encode("utf-8-sig")
