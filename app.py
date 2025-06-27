# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any

# --- 사용자 정의 모듈 임포트 ---
from data_handler import create_sample_data
from feature_engineer import create_date_features
from model_trainer import MIN_DATA_DAYS
from parallel_processor import run_parallel_processing
from column_mapper import auto_map_columns
from ui_components import (
    apply_custom_css, display_header, display_upload_section,
    display_metrics, display_overall_chart, display_product_analysis,
    display_loading_animation, display_debug_info, display_anomaly_report
)

try:
    from gpt_explainer import get_product_explanation, GPTExplainer
    from report_generator import create_downloadable_report
    ADDITIONAL_FEATURES_ENABLED = True
except ImportError:
    st.error("UI, LLM, PDF 관련 모듈 로딩 실패. 일부 기능이 비활성화됩니다.")
    ADDITIONAL_FEATURES_ENABLED = False


# --- 페이지 기본 설정 ---
st.set_page_config(page_title="SmartDemand AI", layout="wide", page_icon="🎯")
apply_custom_css()
display_header()

# --- Session State 초기화 ---
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'uploaded_file' not in st.session_state: st.session_state.uploaded_file = None
if 'use_sample' not in st.session_state: st.session_state.use_sample = False
if 'gpt_explanations' not in st.session_state: st.session_state.gpt_explanations = {}
if 'api_key' not in st.session_state: st.session_state.api_key = None
if 'font_file' not in st.session_state: st.session_state.font_file = None


# --- 사이드바 UI ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 AI 해설 설정")
    api_key_input = st.text_input(
        "Fireworks AI API Key", type="password",
        help="AI 해설 기능을 사용하려면 Fireworks AI API 키를 입력하세요."
    )
    if api_key_input: st.session_state.api_key = api_key_input
    
    st.subheader("📄 PDF 설정")
    font_file_uploader = st.file_uploader(
        "한글 폰트 파일 (선택)", type=['ttf'],
        help="PDF 리포트의 한글 표시를 위한 폰트 파일"
    )
    if font_file_uploader: st.session_state.font_file = font_file_uploader

    st.markdown("---")
    st.subheader("🔧 시스템 설정")
    if st.button("🔄 캐시 초기화", help="예측이 제대로 되지 않거나 오류 발생 시 사용하세요"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ 캐시가 초기화되었습니다!")
        st.rerun()

# --- 데이터 처리 및 모델링 함수 ---
def process_uploaded_file(uploaded_file):
    """
    [수정] Streamlit의 UploadedFile 객체를 안정적으로 처리하기 위해 BytesIO를 사용하고,
    다양한 엑셀 엔진 및 CSV 인코딩을 순차적으로 시도합니다.
    """
    try:
        file_name = uploaded_file.name
        # 파일을 메모리에서 직접 읽어 BytesIO로 변환하여 안정성 확보
        file_bytes = BytesIO(uploaded_file.getvalue())

        if file_name.endswith('.csv'):
            try:
                # UTF-8으로 먼저 시도
                file_bytes.seek(0)
                df = pd.read_csv(file_bytes, encoding='utf-8-sig')
                st.info("✅ CSV 파일을 UTF-8 형식으로 성공적으로 읽었습니다.")
            except UnicodeDecodeError:
                # 실패 시, UTF-16으로 다시 시도
                st.warning("⚠️ UTF-8로 파일을 읽는 데 실패했습니다. UTF-16(엑셀 CSV) 형식으로 다시 시도합니다.")
                file_bytes.seek(0)
                df = pd.read_csv(file_bytes, encoding='utf-16', sep='\t')
                st.info("✅ CSV 파일을 UTF-16 형식으로 성공적으로 읽었습니다.")
            return df

        elif file_name.endswith(('.xls', '.xlsx')):
            engines = ['openpyxl', 'xlrd']
            for engine in engines:
                try:
                    file_bytes.seek(0)
                    df = pd.read_excel(file_bytes, engine=engine)
                    st.info(f"✅ 엑셀 파일을 '{engine}' 엔진으로 성공적으로 읽었습니다.")
                    return df
                except Exception:
                    st.warning(f"⚠️ '{engine}' 엔진으로 엑셀 파일을 여는 데 실패했습니다. 다음 방법으로 다시 시도합니다.")

            # 엑셀 엔진으로 모두 실패 시, 텍스트 파일일 가능성을 염두
            st.warning("⚠️ 엑셀 형식으로 파일을 여는 데 실패했습니다. 텍스트 파일(CSV) 형식으로 마지막 시도를 합니다.")
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes, encoding='utf-16', sep='\t')
            st.info("✅ 파일을 텍스트(UTF-16) 형식으로 성공적으로 읽었습니다.")
            return df
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드해주세요.")
            return None
            
    except Exception as e:
        st.error(f"파일을 처리하는 중 예기치 못한 오류가 발생했습니다: {e}")
        st.info("엑셀 파일(.xls, .xlsx)을 정상적으로 읽기 위해 `xlrd`와 `openpyxl` 라이브러리가 모두 필요할 수 있습니다. (pip install xlrd openpyxl)")
        return None

@st.cache_data
def get_data(uploaded_file, use_sample=False):
    if uploaded_file and not use_sample:
        raw_df = process_uploaded_file(uploaded_file)
        if raw_df is None: return None
    else:
        st.info("샘플 데이터로 실행합니다...")
        raw_df = create_sample_data()
    
    mapped_df = auto_map_columns(raw_df)
    
    if '날짜' not in mapped_df.columns or '품목명' not in mapped_df.columns or '판매량' not in mapped_df.columns:
        st.error("자동 컬럼 인식에 실패했습니다. 파일에 '날짜', '품목명', '판매량(또는 가격)'에 해당하는 컬럼이 있는지 확인해주세요.")
        return None

    try:
        mapped_df['날짜'] = pd.to_datetime(mapped_df['날짜'], errors='coerce')
        if mapped_df['날짜'].isna().any():
            st.warning("⚠️ '날짜' 컬럼에 인식할 수 없는 값이 있어 해당 행을 제외합니다. 날짜 형식을 확인해주세요.")
            mapped_df.dropna(subset=['날짜'], inplace=True)
    except Exception as e:
        st.error(f"'날짜' 컬럼을 날짜 형식으로 변환하는 중 오류가 발생했습니다: {e}")
        return None

    return create_date_features(mapped_df)


@st.cache_data(ttl=3600)
def load_models_and_ensemble(_df):
    models_data = run_parallel_processing(_df)
    return models_data


# --- 메인 로직 ---
uploaded_file, upload_btn, sample_btn = display_upload_section()

if upload_btn and uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.use_sample = False
    st.session_state.data_loaded = True
    st.cache_data.clear()
    st.rerun()

if sample_btn:
    st.session_state.uploaded_file = None
    st.session_state.use_sample = True
    st.session_state.data_loaded = True
    st.cache_data.clear()
    st.rerun()

if st.session_state.get('data_loaded', False):
    df = get_data(st.session_state.uploaded_file, st.session_state.use_sample)

    if df is not None:
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            display_loading_animation()
        
        models_data = load_models_and_ensemble(df)
        
        loading_placeholder.empty() 
        
        anomaly_log_df = models_data.pop('total_anomaly_log', pd.DataFrame())
        
        available_products = [p for p, d in models_data.items() if d.get('ensemble_forecast') is not None]

        if available_products:
            display_anomaly_report(anomaly_log_df)

            st.subheader("📊 전체 비즈니스 현황 요약")
            display_metrics(df, models_data)
            display_overall_chart(df, models_data)
            st.markdown("---")

            st.subheader("📈 품목별 상세 분석 및 AI 인사이트")
            st.markdown("### 🎯 품목별 예측 결과")
            for i in range(0, len(available_products), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(available_products):
                        product = available_products[i + j]
                        with col:
                            display_product_analysis(product, models_data[product], df)

            if ADDITIONAL_FEATURES_ENABLED:
                st.markdown("---")
                st.subheader("📄 추가 기능")
                exp_col1, exp_col2 = st.columns(2)

                with exp_col1:
                    with st.expander("🤖 모든 품목 AI 해설 일괄 생성"):
                        if not st.session_state.api_key:
                            st.warning("AI 해설을 생성하려면 사이드바에서 Fireworks AI API 키를 입력하세요.")
                        else:
                            if st.button("🚀 모든 품목 해설 생성 실행"):
                                explainer = GPTExplainer(st.session_state.api_key)
                                batch_data = []

                                # [핵심 수정] AI 해설에 이상 현상 정보를 포함시키기 위한 로직
                                for p, md in models_data.items():
                                    if isinstance(md, dict) and p in available_products:
                                        fc = md.get('ensemble_forecast')
                                        
                                        # 해당 품목의 이상 현상 로그만 필터링하고 가공
                                        anomaly_info_list = []
                                        if not anomaly_log_df.empty:
                                            product_anomaly_log = anomaly_log_df[anomaly_log_df['품목명'] == p]
                                            if not product_anomaly_log.empty:
                                                # 너무 많으면 요약이 어려우므로 최근 5개만 사용
                                                for _, row in product_anomaly_log.head(5).iterrows():
                                                    log_str = (
                                                        f"{row['날짜'].strftime('%Y년 %m월 %d일')}: "
                                                        f"판매량이 비정상적으로 '{row['이상_유형']}'했습니다. "
                                                        f"(AI 추정 원인: {row['추정_원인']})"
                                                    )
                                                    anomaly_info_list.append(log_str)

                                        batch_data.append({
                                            'name': p, 'model': 'Ensemble',
                                            'accuracy': md['ensemble_performance'].get('Accuracy(%)', 0),
                                            'mae': md['ensemble_performance'].get('MAE', 0),
                                            'forecast_7day': fc['ensemble_yhat'].sum() if fc is not None else 0,
                                            'avg_daily': fc['ensemble_yhat'].mean() if fc is not None else 0,
                                            # 가공된 이상 현상 정보를 함께 전달
                                            'anomaly_info': anomaly_info_list if anomaly_info_list else None
                                        })

                                with st.spinner("AI가 모든 품목의 리포트를 분석 중입니다... (과거 데이터 특징 포함)"):
                                    explanations = explainer.generate_batch_explanations(batch_data)
                                    st.session_state.gpt_explanations.update(explanations)
                                st.success(f"{len(explanations)}개 품목의 해설이 생성되었습니다!")

                with exp_col2:
                    with st.expander("📥 PDF 리포트 다운로드"):
                        if st.button("📑 PDF 리포트 생성 및 다운로드"):
                            with st.spinner("PDF 리포트를 생성하는 중..."):
                                valid_model_data = [md for md in models_data.values() if isinstance(md, dict)]
                                
                                avg_acc = np.mean([md['ensemble_performance'].get('Accuracy(%)',0) for md in valid_model_data if md.get('ensemble_performance')])
                                total_fc = sum([md['ensemble_forecast']['ensemble_yhat'].sum() for md in valid_model_data if md.get('ensemble_forecast') is not None])
                                summary_data = {
                                    'period': f"{df['날짜'].min().strftime('%Y-%m-%d')} ~ {df['날짜'].max().strftime('%Y-%m-%d')}",
                                    'product_count': len(available_products), 'avg_accuracy': avg_acc, 'total_forecast': total_fc
                                }
                                products_report_data = {p: {'accuracy': md['ensemble_performance'].get('Accuracy(%)',0),
                                                            'mae': md['ensemble_performance'].get('MAE',0),
                                                            'forecast_7day': md['ensemble_forecast']['ensemble_yhat'].sum() if md.get('ensemble_forecast') is not None else 0,
                                                            'avg_daily': md['ensemble_forecast']['ensemble_yhat'].mean() if md.get('ensemble_forecast') is not None else 0}
                                                      for p, md in models_data.items() if isinstance(md, dict) and p in available_products}
                                
                                font_path = None
                                if st.session_state.font_file:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as tmp:
                                        tmp.write(st.session_state.font_file.getvalue())
                                        font_path = tmp.name
                                pdf_data = create_downloadable_report(summary_data, products_report_data, st.session_state.gpt_explanations, font_path)
                                st.download_button(
                                    label="📥 PDF 다운로드", data=pdf_data,
                                    file_name=f"smartdemand_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf"
                                )

            display_debug_info(df, models_data, MIN_DATA_DAYS)

        st.markdown("---")
        if st.button("🔄 새 데이터로 다시 분석하기", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.cache_data.clear()
            st.rerun()