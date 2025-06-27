import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional

@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    판매 데이터를 자동으로 전처리하는 함수
    
    Args:
        df: 원본 데이터프레임
        
    Returns:
        전처리된 데이터프레임 (날짜, 수요 컬럼)
    """
    # 데이터프레임 복사본 생성
    df_processed = df.copy()
    
    # 1. 날짜 컬럼 자동 탐지
    date_column = detect_date_column(df_processed)
    if date_column is None:
        raise ValueError("날짜 컬럼을 찾을 수 없습니다.")
    
    # 2. 판매량/수요 컬럼 자동 탐지
    demand_column = detect_demand_column(df_processed, exclude_col=date_column)
    if demand_column is None:
        raise ValueError("판매량/수요 컬럼을 찾을 수 없습니다.")
    
    # 필요한 컬럼만 선택
    df_processed = df_processed[[date_column, demand_column]].copy()
    
    # 컬럼명 표준화
    df_processed.rename(columns={
        date_column: '날짜',
        demand_column: '수요'
    }, inplace=True)
    
    # 3. 날짜 컬럼을 datetime으로 변환
    df_processed['날짜'] = pd.to_datetime(df_processed['날짜'], errors='coerce')
    
    # 날짜 변환 실패한 행 제거
    df_processed = df_processed.dropna(subset=['날짜'])
    
    # 4. 수요 컬럼을 수치형으로 변환
    df_processed['수요'] = pd.to_numeric(df_processed['수요'], errors='coerce')
    
    # 5. 날짜 기준으로 정렬
    df_processed = df_processed.sort_values('날짜')
    
    # 6. 결측치 처리 (forward fill)
    df_processed['수요'] = df_processed['수요'].fillna(method='ffill')
    
    # 여전히 남은 결측치는 backward fill
    df_processed['수요'] = df_processed['수요'].fillna(method='bfill')
    
    # 7. 이상치 제거 (Z-score ±3)
    z_scores = np.abs((df_processed['수요'] - df_processed['수요'].mean()) / df_processed['수요'].std())
    df_processed = df_processed[z_scores < 3]
    
    # 8. 인덱스 리셋
    df_processed = df_processed.reset_index(drop=True)
    
    return df_processed


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    날짜 컬럼을 자동으로 탐지하는 함수
    
    Args:
        df: 데이터프레임
        
    Returns:
        날짜 컬럼명 또는 None
    """
    # 날짜 관련 키워드
    date_keywords = ['date', '날짜', '일자', 'day', 'time', '시간', '일시']
    
    # 컬럼명에서 날짜 키워드 검색 (대소문자 무시)
    for col in df.columns:
        col_lower = col.lower()
        for keyword in date_keywords:
            if keyword in col_lower:
                # 실제로 날짜로 변환 가능한지 확인
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    # 변환 성공률이 80% 이상이면 날짜 컬럼으로 판단
                    success_rate = df[col].notna().sum() / len(df)
                    if success_rate > 0.8:
                        return col
                except:
                    continue
    
    # 키워드로 찾지 못한 경우, 각 컬럼을 날짜로 변환 시도
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                # 변환 성공률이 80% 이상이면 날짜 컬럼으로 판단
                success_rate = converted.notna().sum() / len(df)
                if success_rate > 0.8:
                    return col
            except:
                continue
    
    return None


def detect_demand_column(df: pd.DataFrame, exclude_col: str = None) -> Optional[str]:
    """
    판매량/수요 컬럼을 자동으로 탐지하는 함수
    
    Args:
        df: 데이터프레임
        exclude_col: 제외할 컬럼명 (날짜 컬럼)
        
    Returns:
        수요 컬럼명 또는 None
    """
    # 판매량/수요 관련 키워드
    demand_keywords = ['sales', '판매', '수요', 'demand', 'sold', 'quantity', '수량', 
                      'amount', '금액', 'revenue', '매출', 'volume', '거래량']
    
    # 제외 컬럼 리스트 생성
    exclude_cols = [exclude_col] if exclude_col else []
    
    # 컬럼명에서 수요 키워드 검색 (대소문자 무시)
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        col_lower = col.lower()
        for keyword in demand_keywords:
            if keyword in col_lower:
                # 숫자형 데이터인지 확인
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    return col
                except:
                    continue
    
    # 키워드로 찾지 못한 경우, 숫자형 컬럼 중 첫 번째 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in exclude_cols:
            return col
    
    # 숫자로 변환 가능한 object 타입 컬럼 찾기
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == 'object':
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # 변환 성공률이 80% 이상이면 수요 컬럼으로 판단
                success_rate = converted.notna().sum() / len(df)
                if success_rate > 0.8:
                    return col
            except:
                continue
    
    return None


# 사용 예시 (Streamlit 앱에서)
"""
# Streamlit 앱에서 사용하는 방법:

import streamlit as st
import pandas as pd

st.title("SmartDemand - 데이터 전처리")

# 파일 업로드
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    # 파일 읽기
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("원본 데이터:")
    st.dataframe(df.head())
    
    try:
        # 데이터 전처리
        df_processed = preprocess_data(df)
        
        st.success("데이터 전처리 완료!")
        st.write("전처리된 데이터:")
        st.dataframe(df_processed.head())
        
        # 전처리 결과 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 데이터 수", f"{len(df_processed):,}개")
        with col2:
            st.metric("날짜 범위", f"{df_processed['날짜'].min().date()} ~ {df_processed['날짜'].max().date()}")
        with col3:
            st.metric("평균 수요", f"{df_processed['수요'].mean():.1f}")
            
    except ValueError as e:
        st.error(f"전처리 오류: {str(e)}")
    except Exception as e:
        st.error(f"예상치 못한 오류: {str(e)}")
"""