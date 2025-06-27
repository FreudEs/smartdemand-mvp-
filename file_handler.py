# file_handler.py
import streamlit as st
import pandas as pd
import chardet

# 다른 .py 파일에서 함수 임포트
from data_handler import create_sample_data
from feature_engineer import create_date_features

def load_raw_data_from_uploaded_file(uploaded_file):
    """
    (수정) 업로드된 파일을 읽어 가공 전의 원본 데이터프레임으로 반환합니다.
    """
    try:
        uploaded_file.seek(0)
        
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            # 인코딩 자동 감지
            raw_data = uploaded_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            st.info(f"파일 인코딩을 '{encoding}'으로 감지했습니다.")
            df = pd.read_csv(pd.io.common.BytesIO(raw_data), encoding=encoding)
        else:
            st.error("지원하지 않는 파일 형식입니다. (CSV, XLSX, XLS 파일만 가능)")
            return None
        
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None
