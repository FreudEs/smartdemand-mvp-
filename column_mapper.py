# column_mapper.py
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any

def auto_map_columns(df):
    """
    데이터프레임의 컬럼명을 분석하여 표준 컬럼명으로 자동 변경합니다.
    [수정] 날씨, 프로모션 등 다양한 외부 변수 컬럼을 인식하도록 기능 확장.
    """
    st.info("📊 업로드된 데이터의 컬럼을 분석하고 표준화합니다...")
    
    # 표준 컬럼명과, 실제 데이터에서 사용될 수 있는 다양한 이름들의 목록
    # 외부 변수('event', 'promotion', 'temperature', 'weather') 추가
    REQUIRED_MAP = {
        '날짜': ['date', 'day', '일자', '날짜', '시간', 'time', '연월일', '조사일'],
        '품목명': ['item', 'product', '품목', '상품', '이름', 'name', '품목명'],
        '판매량': ['sales', 'quantity', 'qty', '수량', '판매', '개수', 'volume', '가격', 'price'],
        '이벤트': ['event', '이벤트', '행사'],
        '프로모션': ['promotion', '프로모션', '할인'],
        '기온': ['temp', 'temperature', '기온'],
        '날씨': ['weather', '날씨']
    }

    original_columns = df.columns.tolist()
    rename_dict = {}
    mapped_messages = []

    # 각 표준 컬럼에 대해 매핑 시도
    for standard_col, keywords in REQUIRED_MAP.items():
        if standard_col in original_columns:
            continue

        found = False
        for col in original_columns:
            # 이미 다른 표준 컬럼으로 매핑된 컬럼은 제외
            if col not in rename_dict:
                for keyword in keywords:
                    if keyword in col.lower():
                        rename_dict[col] = standard_col
                        mapped_messages.append(f"'{col}' 컬럼을 **'{standard_col}'**(으)로 자동 인식했습니다.")
                        found = True
                        break
            if found:
                break
    
    # 컬럼 이름 변경 적용
    if rename_dict:
        df = df.rename(columns=rename_dict)
        st.success("✅ 컬럼 자동 인식 및 표준화 완료!")
        for msg in mapped_messages:
            st.info(f"ℹ️ {msg}")

    return df
