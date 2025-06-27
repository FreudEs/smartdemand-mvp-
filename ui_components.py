# ui_components.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from typing import Optional, Dict, List, Tuple

def apply_custom_css():
    """커스텀 CSS 적용 - 다크 테마"""
    st.markdown("""
    <style>
    /* 전역 애니메이션 정의 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.8), 0 0 30px rgba(59, 130, 246, 0.5); }
        100% { box-shadow: 0 0 5px rgba(59, 130, 246, 0.5); }
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Streamlit 다크 테마 설정 */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
        background-image: 
            radial-gradient(at 20% 80%, #1a1b2620 0px, transparent 50%),
            radial-gradient(at 80% 0%, #2a2b3d20 0px, transparent 50%),
            radial-gradient(at 40% 40%, #16171f20 0px, transparent 50%);
        animation: fadeIn 0.5s ease-out;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
        backdrop-filter: blur(10px);
    }
    
    /* 모든 섹션 배경 검은색으로 */
    section[data-testid="stSidebar"], 
    .main .block-container,
    [data-testid="stToolbar"] {
        background-color: #0e1117 !important;
    }
    
    /* 모든 텍스트 흰색으로 */
    .stApp, .stApp *, 
    .stMarkdown, .stMarkdown *, 
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #ffffff !important;
    }
    
    /* 강조색 정의 */
    .accent-blue { color: #3b82f6 !important; }
    .accent-green { color: #10b981 !important; }
    .accent-yellow { color: #f59e0b !important; }
    .accent-red { color: #ef4444 !important; }
    .accent-purple { color: #8b5cf6 !important; }
    
    /* 메트릭 카드 스타일 - 다크 테마 */
    .metric-card {
        background: linear-gradient(135deg, #1a1b26 0%, #16171f 100%) !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
        border-color: #3b82f6 !important;
    }
    .metric-card h3 {
        color: #94a3b8 !important;
        font-size: 0.9rem;
        margin-bottom: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card small {
        color: #64748b !important;
        font-size: 0.85rem;
    }
    
    /* 제품 카드 스타일 - 다크 테마 */
    .product-card {
        background: linear-gradient(135deg, #1e1f2e 0%, #1a1b26 100%) !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .product-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    }
    .product-card:hover::before {
        transform: translateX(0);
    }
    .product-card:hover {
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.15);
        transform: translateY(-3px);
        border-color: #3b82f6 !important;
    }
    .product-card h4 {
        color: #f1f5f9 !important;
        margin-bottom: 20px;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .product-card p {
        color: #cbd5e1 !important;
        line-height: 1.6;
    }
    .recommendation {
        background: linear-gradient(135deg, #2a2b3d 0%, #1e1f2e 100%) !important;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 0.9rem;
        color: #e2e8f0 !important;
        border-left: 3px solid #10b981;
    }
    
    /* 입력 필드 스타일 - 다크 테마 */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #1a1b26 !important;
        color: #ffffff !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* 파일 업로더 스타일 - 다크 테마 */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #1a1b26 0%, #16171f 100%) !important;
        border: 2px dashed #3b82f6 !important;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #10b981 !important;
        background: linear-gradient(135deg, #1e1f2e 0%, #1a1b26 100%) !important;
    }
    
    [data-testid="stFileUploadDropzone"] * {
        color: #e2e8f0 !important;
    }
    
    /* Expander 스타일 - 다크 테마 */
    [data-testid="stExpander"] {
        background-color: #1a1b26 !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .streamlit-expanderHeader {
        background-color: #1e1f2e !important;
        color: #f1f5f9 !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #252631 !important;
    }
    
    /* Info, Warning, Error 박스 스타일 - 다크 테마 */
    .stAlert {
        background-color: #1a1b26 !important;
        border-radius: 10px;
        border: 1px solid #2a2b3d !important;
    }
    
    /* Info 박스 */
    [data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2f4e 100%) !important;
        border-left: 4px solid #3b82f6 !important;
        color: #dbeafe !important;
    }
    
    /* Success 박스 */
    [data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
        background: linear-gradient(135deg, #14532d 0%, #15422a 100%) !important;
        border-left: 4px solid #10b981 !important;
        color: #d1fae5 !important;
    }
    
    /* Warning 박스 */
    [data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
        background: linear-gradient(135deg, #78350f 0%, #5a2a0c 100%) !important;
        border-left: 4px solid #f59e0b !important;
        color: #fef3c7 !important;
    }
    
    /* Error 박스 */
    [data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
        background: linear-gradient(135deg, #7f1d1d 0%, #5f1818 100%) !important;
        border-left: 4px solid #ef4444 !important;
        color: #fee2e2 !important;
    }
    
    /* 전처리 관련 스타일 - 다크 테마 */
    .preprocess-step {
        background-color: #1a1b26;
        border-left: 4px solid #2a2b3d;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .preprocess-step.active {
        border-left-color: #3b82f6;
        background: linear-gradient(135deg, #1e1f2e 0%, #1a1b26 100%);
    }
    .preprocess-step.complete {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #14532d20 0%, #15422a20 100%);
    }
    .preprocess-step.error {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #7f1d1d20 0%, #5f181820 100%);
    }
    
    /* Optuna 로딩 관련 스타일 - 다크 테마 */
    .optuna-container {
        background: linear-gradient(135deg, #1e1f2e 0%, #1a1b26 100%);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid #2a2b3d;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .optuna-title {
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .optuna-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    .trial-info {
        background: linear-gradient(135deg, #16171f 0%, #1a1b26 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        border: 1px solid #2a2b3d;
    }
    .trial-metric {
        display: inline-block;
        margin: 0 20px;
    }
    .trial-metric-label {
        color: #64748b;
        font-size: 0.9rem;
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .trial-metric-value {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 버튼 스타일 - 다크 테마 */
    .stButton > button {
        background: linear-gradient(135deg, #1e1f2e 0%, #1a1b26 100%) !important;
        color: #e2e8f0 !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(59, 130, 246, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #252631 0%, #1e1f2e 100%) !important;
        border-color: #3b82f6 !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 25px rgba(59, 130, 246, 0.3);
    }
    
    /* Primary 버튼 스타일 - 강조색 */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 5px 25px rgba(59, 130, 246, 0.5);
    }
    
    /* 메트릭 컴포넌트 스타일 - 다크 테마 */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1b26 0%, #16171f 100%) !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 12px;
        padding: 1.5rem !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: #3b82f6 !important;
        box-shadow: 0 5px 20px rgba(59, 130, 246, 0.2);
    }
    
    [data-testid="metric-container"] * {
        color: #f1f5f9 !important;
    }
    
    /* 진행률 바 스타일 - 다크 테마 */
    .stProgress > div > div {
        background-color: #1a1b26 !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%) !important;
        height: 8px;
        border-radius: 10px;
        animation: progressShine 2s ease-in-out infinite;
    }
    
    @keyframes progressShine {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* 사이드바 스타일 - 다크 테마 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16171f 0%, #0e1117 100%) !important;
        border-right: 1px solid #2a2b3d !important;
    }
    
    /* 차트 배경 - 다크 테마 */
    .js-plotly-plot .plotly {
        background-color: transparent !important;
    }
    
    /* 테이블 스타일 - 다크 테마 */
    .dataframe {
        background-color: #1a1b26 !important;
        color: #e2e8f0 !important;
    }
    
    .dataframe th {
        background-color: #1e1f2e !important;
        color: #f1f5f9 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    
    .dataframe td {
        border-bottom: 1px solid #2a2b3d !important;
    }
    
    /* 스크롤바 스타일링 */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1b26;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2b3d;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3b82f6;
    }
    
    /* 추가 시각 효과 */
    
    /* 네온 글로우 효과 */
    .neon-text {
        text-shadow: 
            0 0 10px rgba(59, 130, 246, 0.8),
            0 0 20px rgba(59, 130, 246, 0.6),
            0 0 30px rgba(59, 130, 246, 0.4),
            0 0 40px rgba(59, 130, 246, 0.2);
    }
    
    /* 카드에 홀로그램 효과 */
    .hologram {
        background: linear-gradient(45deg, 
            rgba(59, 130, 246, 0.1), 
            rgba(16, 185, 129, 0.1), 
            rgba(139, 92, 246, 0.1));
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    /* 링크 호버 효과 */
    a {
        color: #3b82f6 !important;
        text-decoration: none;
        transition: all 0.3s ease;
        position: relative;
    }
    
    a:hover {
        color: #10b981 !important;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }
    
    /* 인풋 포커스 효과 강화 */
    input:focus, textarea:focus, select:focus {
        outline: none !important;
        box-shadow: 
            0 0 0 3px rgba(59, 130, 246, 0.3),
            0 0 20px rgba(59, 130, 246, 0.2) !important;
        animation: glow 2s ease-in-out infinite;
    }
    
    /* 성공/에러 메시지 애니메이션 */
    .stSuccess, .stError, .stWarning, .stInfo {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 코드 블록 스타일 */
    .stCodeBlock {
        background-color: #1a1b26 !important;
        border: 1px solid #2a2b3d !important;
        border-radius: 10px;
    }
    
    /* 드롭다운 메뉴 스타일 */
    [data-baseweb="select"] {
        background-color: #1a1b26 !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #1a1b26 !important;
        border-color: #2a2b3d !important;
    }
    
    /* 체크박스와 라디오 버튼 스타일 */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p,
    .stRadio > label > div[data-testid="stMarkdownContainer"] > p {
        color: #e2e8f0 !important;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1b26;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background-color: transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* 모달/다이얼로그 스타일 */
    [data-baseweb="modal"] {
        background-color: #1a1b26 !important;
    }
    
    /* 사이드바 위젯 스타일 */
    .sidebar .stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    /* 툴팁 스타일 */
    [role="tooltip"] {
        background-color: #1e1f2e !important;
        border: 1px solid #3b82f6 !important;
        color: #e2e8f0 !important;
    }
    
    /* 페이지 전환 효과 */
    .main > div {
        animation: fadeIn 0.3s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """앱 헤더를 표시하는 UI 함수 - 다크 테마"""
    apply_custom_css()
    
    # 헤더 컨테이너
    header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
    
    with header_col2:
        # 헤더 타이틀
        st.markdown(
            """
            <div style="text-align: center;">
                <h1 style="background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
                           -webkit-background-clip: text;
                           -webkit-text-fill-color: transparent;
                           font-size: 3rem;
                           font-weight: 700;
                           margin-bottom: 0;">SmartDemand AI</h1>
                <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 0.5rem;">
                    AI 수요 예측 시스템 - v1.0 Beta
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 구분선 - 그라데이션
    st.markdown(
        """
        <hr style="margin: 30px 0; 
                   border: none; 
                   height: 2px;
                   background: linear-gradient(90deg, 
                                             transparent 0%, 
                                             #3b82f6 20%, 
                                             #10b981 50%, 
                                             #3b82f6 80%, 
                                             transparent 100%);">
        """,
        unsafe_allow_html=True
    )

def display_upload_ui():
    """초기 데이터 업로드 화면을 표시하는 UI 함수"""
    apply_custom_css()
    
    st.image("https://i.imgur.com/k2G9s2x.png", width=100)
    st.title("SmartDemand")
    st.subheader("AI 수요 예측 시스템 - 실제 데이터 업로드 버전")

    with st.container(border=True):
        st.markdown("##### 📊 데이터 업로드")
        st.write("CSV 또는 Excel 파일을 업로드하여 실제 데이터로 수요를 예측해보세요!")
        uploaded_file = st.file_uploader("파일 선택", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        
        upload_clicked = col1.button("🚀 업로드 및 분석", type="primary", disabled=(uploaded_file is None))
        sample_clicked = col2.button("🎮 샘플 데이터 사용")

        return uploaded_file, upload_clicked, sample_clicked

def display_upload_section():
    """메인 화면의 파일 업로드 섹션을 표시하는 UI 함수"""
    with st.container():
        st.markdown("### 📊 데이터 업로드")
        st.write("CSV 또는 Excel 파일을 업로드하여 실제 데이터로 수요를 예측해보세요!")
        
        uploaded_file = st.file_uploader(
            "파일 선택", 
            type=['csv', 'xlsx', 'xls'], 
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            upload_clicked = st.button(
                "🚀 업로드 및 분석", 
                type="primary", 
                disabled=(uploaded_file is None),
                use_container_width=True
            )
        
        with col2:
            sample_clicked = st.button(
                "🎮 샘플 데이터 사용",
                use_container_width=True
            )
        
        return uploaded_file, upload_clicked, sample_clicked

def display_metrics(df, models_data):
    """전체 메트릭을 표시하는 UI 함수"""
    available_products = [
        p for p, d in models_data.items() 
        if d.get('ensemble_forecast') is not None
    ]
    
    if not available_products:
        st.warning("분석 가능한 데이터가 없습니다.")
        return
    
    # 평균 정확도 계산
    valid_accuracies = [
        models_data[p]['ensemble_performance']['Accuracy(%)'] 
        for p in available_products 
        if 'Accuracy(%)' in models_data[p].get('ensemble_performance', {})
    ]
    avg_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0
    
    # 전체 예측량 계산
    total_predicted_sales_7day = sum([
        models_data[p]['ensemble_forecast']['ensemble_yhat'].sum()
        for p in available_products
    ])
    
    # 분석 기간 계산
    analysis_period_days = (df['날짜'].max() - df['날짜'].min()).days + 1
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>📈 평균 정확도</h3><p>{avg_accuracy:.1f}%</p></div>', 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>📦 분석 품목</h3><p>{len(available_products)}개</p></div>', 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f'<div class="metric-card"><h3>🎯 7일 예측</h3><p>{total_predicted_sales_7day:,.0f}개</p></div>', 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f'<div class="metric-card"><h3>📅 분석 기간</h3><p>{analysis_period_days}일</p></div>', 
            unsafe_allow_html=True
        )

def display_overall_chart(df, models_data):
    """전체 판매량 차트를 표시하는 UI 함수"""
    available_products = [
        p for p, d in models_data.items() 
        if d.get('ensemble_forecast') is not None
    ]
    
    if not available_products:
        return
    
    # 전체 예측 데이터 준비
    all_forecasts = []
    for p in available_products:
        forecast = models_data[p]['ensemble_forecast'].copy()
        forecast = forecast.rename(columns={
            forecast.columns[0]: '날짜', 
            'ensemble_yhat': '예측 판매량'
        })
        all_forecasts.append(forecast)
    
    overall_forecast_df = pd.concat(all_forecasts).groupby('날짜')['예측 판매량'].sum().reset_index()
    
    # 차트 생성
    fig = go.Figure()
    overall_sales_df = df.groupby('날짜')['판매량'].sum().reset_index()
    
    fig.add_trace(go.Scatter(
        x=overall_sales_df['날짜'], 
        y=overall_sales_df['판매량'], 
        mode='lines', 
        name='실제 판매량', 
        line=dict(color='#333', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=overall_forecast_df['날짜'], 
        y=overall_forecast_df['예측 판매량'], 
        mode='lines', 
        name='예측 판매량', 
        line=dict(color='#666', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title='전체 판매량 추이 및 예측',
        xaxis_title='날짜',
        yaxis_title='판매량',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#333'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    st.plotly_chart(fig, use_container_width=True)

def display_product_analysis(product, product_model_data, df):
    """개별 품목 분석을 표시하는 UI 함수"""
    with st.expander(f"**{product}** - 상세 분석 보기"):
        st.markdown("##### **모델 성능 평가**")
        
        col1, col2, col3 = st.columns(3)
        
        ens_acc = product_model_data['ensemble_performance'].get('Accuracy(%)', 0)
        ens_mae = product_model_data['ensemble_performance'].get('MAE', 0)
        
        col1.metric("🏆 최종 예측 정확도", f"{ens_acc:.1f}%")
        col2.metric("MAE (평균 절대 오차)", f"{ens_mae:.1f}개")
        
        forecast_sum = 0
        if product_model_data.get('ensemble_forecast') is not None:
            forecast_sum = product_model_data.get('ensemble_forecast')['ensemble_yhat'].sum()
        
        col3.metric("7일 예상 판매량", f"{forecast_sum:.0f}개" if forecast_sum > 0 else "N/A")
        
        # 인사이트 생성 및 표시
        commentary, insights = generate_insights_and_commentary(product, product_model_data, df)
        
        st.markdown("##### **AI 분석 및 제안**")
        st.info(commentary)
        
        for insight in insights:
            st.markdown(f"- {insight}")

def display_success_message(message: str):
    """성공 메시지를 표시하는 UI 함수"""
    st.success(f"✅ {message}")

def display_warning_message(message: str):
    """경고 메시지를 표시하는 UI 함수"""
    st.warning(f"⚠️ {message}")

def display_error_message(error_type: str, message: str):
    """에러 메시지를 표시하는 UI 함수"""
    if error_type == "warning":
        st.warning(f"⚠️ {message}")
    elif error_type == "error":
        st.error(f"❌ {message}")
    elif error_type == "info":
        st.info(f"ℹ️ {message}")
    else:
        st.write(message)

def display_loading_animation():
    """로딩 애니메이션을 표시하는 UI 함수 - 다크 테마"""
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem;">
            <div class="spinner">
                <div class="double-bounce1"></div>
                <div class="double-bounce2"></div>
            </div>
            <p style="margin-top: 2rem; 
                      background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
                      -webkit-background-clip: text;
                      -webkit-text-fill-color: transparent;
                      font-size: 1.2rem;
                      font-weight: 600;">AI 모델을 학습하고 있습니다...</p>
        </div>
        <style>
            .spinner {
                width: 60px;
                height: 60px;
                position: relative;
                margin: 0 auto;
            }
            .double-bounce1, .double-bounce2 {
                width: 100%;
                height: 100%;
                border-radius: 50%;
                background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
                opacity: 0.6;
                position: absolute;
                top: 0;
                left: 0;
                animation: sk-bounce 2.0s infinite ease-in-out;
            }
            .double-bounce2 {
                animation-delay: -1.0s;
                background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
            }
            @keyframes sk-bounce {
                0%, 100% { 
                    transform: scale(0.0);
                } 50% { 
                    transform: scale(1.0);
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def display_loading_with_progress(current: int, total: int, product_name: str):
    """진행 상황과 함께 로딩을 표시하는 UI 함수 - 다크 테마"""
    progress = current / total
    st.markdown(
        f"""
        <div style="text-align: center; padding: 1rem;">
            <h4 style="background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       font-size: 1.3rem;">🤖 AI 모델 학습 중...</h4>
            <p style="color: #94a3b8; margin: 0.5rem 0;">현재 처리 중: <strong style="color: #f1f5f9;">{product_name}</strong></p>
            <div style="background-color: #1a1b26; border-radius: 50px; overflow: hidden; margin: 1rem 0; height: 10px;">
                <div style="background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%); 
                            height: 100%; 
                            width: {progress*100}%; 
                            transition: width 0.3s ease;
                            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);">
                </div>
            </div>
            <p style="color: #64748b; font-size: 0.9rem;">
                <span style="color: #3b82f6; font-weight: bold;">{current}</span> / 
                <span style="color: #94a3b8;">{total}</span> 품목 완료
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_data_loading():
    """데이터 로딩 중 메시지를 표시하는 UI 함수 - 다크 테마"""
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0;">
            <div style="font-size: 4rem; 
                        animation: pulse 1.5s infinite;
                        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;">📊</div>
            <h3 style="background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       margin: 1rem 0;
                       font-size: 1.8rem;">데이터를 불러오는 중...</h3>
            <p style="color: #94a3b8;">잠시만 기다려주세요.</p>
        </div>
        <style>
            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.7; transform: scale(1.1); }
                100% { opacity: 1; transform: scale(1); }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def display_loading_message(message: str = "처리 중입니다..."):
    """로딩 메시지를 표시하는 UI 함수"""
    return st.spinner(message)

def display_preprocessing_progress(steps: List[Dict[str, any]]):
    """
    데이터 전처리 진행 상황을 표시하는 UI
    
    Args:
        steps: [{'name': '단계명', 'status': 'pending|active|complete|error', 'message': '설명'}]
    """
    st.markdown("### 🔄 데이터 전처리 진행 상황")
    
    for step in steps:
        status_class = step.get('status', 'pending')
        icon = {
            'pending': '⏳',
            'active': '🔄',
            'complete': '✅',
            'error': '❌'
        }.get(status_class, '⏳')
        
        st.markdown(
            f"""
            <div class="preprocess-step {status_class}">
                <strong>{icon} {step['name']}</strong><br>
                <span style="color: #666; font-size: 0.9rem;">{step.get('message', '')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_preprocessing_results(original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                processing_info: Dict[str, any]):
    """
    전처리 결과를 시각적으로 표시하는 UI
    
    Args:
        original_df: 원본 데이터프레임
        processed_df: 전처리된 데이터프레임
        processing_info: 전처리 정보 딕셔너리
    """
    st.markdown("### ✨ 데이터 전처리 완료")
    
    # 전처리 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        original_rows = len(original_df)
        processed_rows = len(processed_df)
        removed_rows = original_rows - processed_rows
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>🗑️ 제거된 데이터</h3>
                <p>{removed_rows:,}</p>
                <small style="color: #888;">({removed_rows/original_rows*100:.1f}%)</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>📅 날짜 범위</h3>
                <p style="font-size: 1.2rem;">{processing_info.get('date_range', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>🔢 결측치 처리</h3>
                <p>{processing_info.get('missing_filled', 0):,}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>⚡ 이상치 제거</h3>
                <p>{processing_info.get('outliers_removed', 0):,}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # 전처리 전후 비교
    with st.expander("📊 전처리 전후 데이터 비교", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**원본 데이터 (처음 10행)**")
            st.dataframe(original_df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("**전처리된 데이터 (처음 10행)**")
            st.dataframe(processed_df.head(10), use_container_width=True)

def display_optuna_progress(current_trial: int, total_trials: int, 
                          best_score: float, elapsed_time: float,
                          current_params: Optional[Dict] = None):
    """
    Optuna 하이퍼파라미터 튜닝 진행 상황을 표시하는 UI
    
    Args:
        current_trial: 현재 시도 횟수
        total_trials: 전체 시도 횟수
        best_score: 현재까지의 최고 점수
        elapsed_time: 경과 시간 (초)
        current_params: 현재 테스트 중인 파라미터
    """
    st.markdown(
        """
        <div class="optuna-container">
            <div class="optuna-title">🎯 AI 모델 최적화 진행 중</div>
            <div class="optuna-subtitle">Optuna를 사용하여 최적의 하이퍼파라미터를 찾고 있습니다</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 프로그레스 바
    progress = current_trial / total_trials
    st.progress(progress)
    
    # 진행 정보
    st.markdown(
        f"""
        <div class="trial-info">
            <div class="trial-metric">
                <span class="trial-metric-label">진행률</span>
                <span class="trial-metric-value">{current_trial}/{total_trials}</span>
            </div>
            <div class="trial-metric">
                <span class="trial-metric-label">최고 정확도</span>
                <span class="trial-metric-value">{best_score:.2f}%</span>
            </div>
            <div class="trial-metric">
                <span class="trial-metric-label">경과 시간</span>
                <span class="trial-metric-value">{elapsed_time:.1f}초</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 현재 테스트 중인 파라미터 (옵션)
    if current_params:
        with st.expander("🔍 현재 테스트 중인 하이퍼파라미터", expanded=False):
            params_df = pd.DataFrame([current_params]).T
            params_df.columns = ['값']
            st.dataframe(params_df, use_container_width=True)
    
    # 예상 남은 시간
    if current_trial > 0:
        avg_time_per_trial = elapsed_time / current_trial
        remaining_trials = total_trials - current_trial
        estimated_remaining = avg_time_per_trial * remaining_trials
        
        st.markdown(
            f"""
            <div style="text-align: center; color: #666; margin-top: 20px;">
                예상 남은 시간: <strong>{estimated_remaining:.0f}초</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

def display_dashboard(df, models_data, loading_time):
    """분석이 완료된 후 전체 결과 대시보드를 표시하는 UI 함수"""
    apply_custom_css()
    
    st.image("https://i.imgur.com/k2G9s2x.png", width=80)
    st.header("SmartDemand 예측 결과")
    
    st.success(f"🎉 데이터 분석 완료! AI 수요예측이 성공적으로 실행되었습니다! (소요 시간: {loading_time:.2f}초)")

    # 전체 요약 지표
    valid_accuracies = [data['ensemble_performance']['Accuracy(%)'] for data in models_data.values() 
                       if data.get('ensemble_performance', {}).get('Accuracy(%)', -1) > -1]
    avg_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0
    
    analyzed_products_count = len(valid_accuracies)
    
    total_predicted_sales_7day = sum(data['ensemble_forecast'].iloc[:, 1].sum() 
                                   for data in models_data.values() 
                                   if data.get('ensemble_forecast') is not None)
    
    analysis_period_days = (df['날짜'].max() - df['날짜'].min()).days + 1 if not df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>📈 평균 예측 정확도</h3><p>{avg_accuracy:.0f}%</p></div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>📦 분석 품목 수</h3><p>{analyzed_products_count}개</p></div>', 
                   unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>🗓️ 7일 총 예상</h3><p>{total_predicted_sales_7day:,.0f}개</p></div>', 
                   unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>⏳ 분석 기간</h3><p>{analysis_period_days}일</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 품목별 예측 카드
    st.subheader("품목별 7일 수요 예측")
    
    product_cols = st.columns(3)
    
    available_products = [p for p, d in models_data.items() 
                         if d.get('ensemble_performance', {}).get('Accuracy(%)', -1) > -1]

    for i, product in enumerate(available_products):
        product_data = models_data[product]
        trend_msg, recommendation, avg_daily, total_7day, accuracy = generate_insights(product, product_data)
        
        with product_cols[i % 3]:
            st.markdown(f"""
            <div class="product-card">
                <h4>{product}</h4>
                <p style="font-size: 1rem; color: #555; margin:0;">일 평균 예측: <b>{avg_daily:.0f}개/일</b></p>
                <p style="font-size: 1rem; color: #555; margin:0;">7일 총 예상: <b>{total_7day:,.0f}개</b></p>
                <p style="font-size: 1rem; color: #555; margin:0;">트렌드: <b>{trend_msg}</b></p>
                <p style="font-size: 1rem; color: #555; margin:0;">예측 정확도: <b>{accuracy:.0f}%</b></p>
                <p class="recommendation">💡 {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 새 데이터로 다시 분석하기"):
        st.session_state.clear()
        st.rerun()

def display_model_training_progress(model_name: str, progress: float, status: str = "training"):
    """모델 훈련 진행 상황을 표시하는 UI 함수"""
    status_emoji = {
        "preparing": "🔧",
        "training": "🏃",
        "validating": "🔍",
        "complete": "✅",
        "error": "❌"
    }
    
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <strong>{status_emoji.get(status, '🔄')} {model_name}</strong>
            <div style="margin-top: 10px;">
                <progress value="{int(progress * 100)}" max="100" style="width: 100%; height: 20px;"></progress>
                <span style="color: #666; font-size: 0.9rem;">{progress * 100:.1f}% 완료</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_sidebar_info():
    """사이드바에 추가 정보를 표시하는 UI 함수"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📌 사용 가이드")
        st.markdown(
            """
            1. **데이터 업로드**: CSV/Excel 파일 선택
            2. **전처리**: 자동으로 데이터 정제
            3. **모델 훈련**: AI가 최적 모델 생성
            4. **결과 확인**: 예측 결과 및 인사이트
            """
        )
        
        st.markdown("---")
        st.markdown("### 🔗 유용한 링크")
        st.markdown("- [사용자 매뉴얼]()")
        st.markdown("- [기술 문서]()")
        st.markdown("- [문의하기]()")

def display_main_dashboard(df, models_data):
    """메인 대시보드를 표시하는 UI 함수"""
    st.subheader("📊 전체 비즈니스 현황 요약")
    
    available_products_for_overall = [
        p for p, d in models_data.items() 
        if d.get('ensemble_forecast') is not None
    ]
    
    if available_products_for_overall:
        # 전체 예측 데이터 준비
        all_forecasts = []
        for p in available_products_for_overall:
            forecast = models_data[p]['ensemble_forecast'].copy()
            forecast = forecast.rename(columns={
                forecast.columns[0]: '날짜', 
                'ensemble_yhat': '예측 판매량'
            })
            all_forecasts.append(forecast)
        
        overall_forecast_df = pd.concat(all_forecasts).groupby('날짜')['예측 판매량'].sum().reset_index()
        
        # 정확도 계산
        valid_accuracies = [
            models_data[p]['ensemble_performance']['Accuracy(%)'] 
            for p in available_products_for_overall 
            if 'Accuracy(%)' in models_data[p].get('ensemble_performance', {})
        ]
        avg_accuracy = np.mean(valid_accuracies) if valid_accuracies else 0
        total_predicted_sales_7day = overall_forecast_df['예측 판매량'].sum()
        
        # 메트릭 표시
        col1, col2 = st.columns(2)
        col1.metric("전체 품목 평균 정확도", f"{avg_accuracy:.1f}%")
        col2.metric("전체 7일 예상 판매량", f"{total_predicted_sales_7day:.0f}개")
        
        # 차트 생성
        fig = go.Figure()
        overall_sales_df = df.groupby('날짜')['판매량'].sum().reset_index()
        
        fig.add_trace(go.Scatter(
            x=overall_sales_df['날짜'], 
            y=overall_sales_df['판매량'], 
            mode='lines', 
            name='총 실제 판매량', 
            line=dict(color='royalblue')
        ))
        
        fig.add_trace(go.Scatter(
            x=overall_forecast_df['날짜'], 
            y=overall_forecast_df['예측 판매량'], 
            mode='lines', 
            name='총 예측 판매량', 
            line=dict(color='limegreen', dash='dash')
        ))
        
        fig.update_layout(
            title='전체 판매량 및 미래 수요 예측',
            xaxis_title='날짜',
            yaxis_title='총 판매량',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("분석할 데이터가 충분한 품목이 없습니다.")

def display_data_preview(df: pd.DataFrame, title: str = "데이터 미리보기"):
    """데이터프레임 미리보기를 표시하는 UI 함수"""
    with st.expander(f"📊 {title}", expanded=True):
        # 데이터 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("행 수", f"{len(df):,}")
        with col2:
            st.metric("열 수", f"{len(df.columns):,}")
        with col3:
            st.metric("메모리 사용량", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # 데이터 표시
        st.dataframe(df.head(10), use_container_width=True)
        
        # 컬럼 정보
        if st.checkbox("컬럼 정보 보기"):
            info_df = pd.DataFrame({
                '컬럼명': df.columns,
                '데이터 타입': df.dtypes,
                '결측치': df.isnull().sum(),
                '고유값 수': df.nunique()
            })
            st.dataframe(info_df, use_container_width=True)

def display_debug_info(df, models_data, min_data_days=30):
    """디버깅 정보를 표시하는 UI 함수"""
    with st.expander("🕵️‍♂️ 디버깅 정보 보기"):
        st.write("#### 현재 처리된 데이터 정보")
        st.write(f"전체 데이터프레임 크기: `{df.shape}`")
        st.write(f"분석 대상 품목 목록: `{df['품목명'].unique().tolist()}`")
        
        st.write("#### 품목별 데이터 개수")
        product_counts = df.groupby('품목명').size().reset_index(name='데이터 개수')
        st.dataframe(product_counts)
        
        available_products = [
            p for p, d in models_data.items() 
            if d.get('ensemble_performance', {}).get('Accuracy(%)', -1) > -1
        ]
        st.write(f"분석 가능한 품목 (최소 {min_data_days}일 이상): `{available_products}`")

# generate_insights 함수 추가
def generate_insights(product_name, product_model_data):
    """UI 표시를 위한 간단한 인사이트 생성 로직"""
    if not product_model_data or product_model_data.get('ensemble_performance', {}).get('Accuracy(%)', -1) <= -1:
        return "데이터 부족", "데이터가 부족하여 분석할 수 없습니다.", 0, 0, 0

    accuracy = product_model_data['ensemble_performance']['Accuracy(%)']
    forecast_df = product_model_data.get('ensemble_forecast')
    
    if forecast_df is None or forecast_df.empty:
        return "예측 실패", "미래 예측값을 생성할 수 없었습니다.", 0, 0, 0
        
    forecast_7day_total = forecast_df.iloc[:, 1].sum()
    avg_daily_forecast = forecast_7day_total / 7

    trend_msg = "안정적인 추세"
    
    recommendation = "현재 수준 유지"
    if accuracy > 80:
        recommendation = "강한 성장세 - 재고 증량 권장"
    elif accuracy < 60:
        recommendation = "감소 추세 - 마케팅 강화 필요"

    return trend_msg, recommendation, avg_daily_forecast, forecast_7day_total, accuracy

# generate_insights_and_commentary 함수도 추가
def generate_insights_and_commentary(product_name, product_model_data, historical_df):
    """품목별 인사이트와 설명을 생성하는 함수"""
    commentary = "데이터가 부족하여 상세 분석을 생성할 수 없습니다."
    insights = []
    
    accuracy = product_model_data.get('ensemble_performance', {}).get('Accuracy(%)', 0)
    mae = product_model_data.get('ensemble_performance', {}).get('MAE', 0)
    forecast_df = product_model_data.get('ensemble_forecast')
    
    if forecast_df is None or forecast_df.empty:
        return commentary, insights
        
    forecast_7day_total = forecast_df['ensemble_yhat'].sum()
    avg_daily_forecast = forecast_7day_total / 7
    
    commentary = f"🤖 AI가 '{product_name}'의 판매 데이터를 분석했습니다. 향후 7일간 일 평균 **약 {avg_daily_forecast:.0f}개**의 판매가 예상됩니다.\n"
    commentary += f"모델의 평균 예측 오차는 **약 {mae:.1f}개** 수준이며, 정확도(MAPE 기반)는 **{accuracy:.1f}%** 입니다."
    
    insights.append(f"**재고 관리**: 향후 7일간 총 **{forecast_7day_total:.0f}개** 판매가 예상됩니다. 예측 오차(평균 {mae:.1f}개)를 감안하여 안전 재고를 설정하세요.")
    
    product_hist_df = historical_df[historical_df['품목명'] == product_name]
    if not product_hist_df.empty:
        weekday_sales = product_hist_df.groupby('요일')['판매량'].mean()
        best_day_index = weekday_sales.idxmax()
        day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        insights.append(f"**마케팅 제안**: 과거 데이터에 따르면 **{day_names[best_day_index]}**에 판매량이 가장 높은 경향을 보입니다. 해당 요일 프로모션을 고려해 보세요.")
    
    return commentary, insights

# --- [신규 추가] 이상 현상 분석 리포트 UI 함수 ---
def display_anomaly_report(anomaly_log_df: pd.DataFrame):
    """
    이상 현상 분석 리포트 UI를 표시합니다.

    Args:
        anomaly_log_df (pd.DataFrame): anomaly_analyzer.py에서 생성된 이상 현상 로그 데이터프레임.
    """
    st.subheader("🕵️‍♂️ 이상 현상 분석 리포트")
    if anomaly_log_df is not None and not anomaly_log_df.empty:
        with st.expander("자세한 분석 로그 보기", expanded=True):
            st.info("AI가 판매 데이터에서 통계적으로 유의미한 이상 현상 및 결측치를 감지하고, 그 원인을 자동 추정했습니다. 이 정보는 모델의 예측 정확도를 높이는 데 사용됩니다.")
            
            # 가독성을 위해 컬럼 순서 및 이름 변경
            display_df = anomaly_log_df.rename(columns={
                '날짜': '발생일',
                '품목명': '품목',
                '판매량': '실판매량',
                '이상_유형': '유형',
                '이벤트_정보': '관련 이벤트',
                '추정_원인': 'AI 추정 원인',
                'z_score': '이상 점수'
            })
            
            # z_score 포맷팅
            display_df['이상 점수'] = display_df['이상 점수'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

            # 최종적으로 보여줄 컬럼 목록
            display_columns = ['발생일', '품목', '실판매량', '유형', 'AI 추정 원인', '관련 이벤트', '이상 점수']
            
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True # 인덱스 숨기기
            )
    else:
        st.success("✅ 데이터에서 특별한 이상 현상이 발견되지 않았습니다.")
    st.markdown("---")