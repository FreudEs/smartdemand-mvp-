# feature_engineer.py
import pandas as pd
import numpy as np

def create_date_features(df):
    """날짜 컬럼에서 시계열 특징을 생성합니다."""
    df['년도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    df['일'] = df['날짜'].dt.day
    df['요일'] = df['날짜'].dt.dayofweek
    df['주차'] = df['날짜'].dt.isocalendar().week.astype(int)
    df['분기'] = df['날짜'].dt.quarter
    return df

def create_xgboost_features(df, product_name):
    """
    XGBoost 모델을 위한 특징을 생성합니다.
    [수정] anomaly_analyzer에서 생성된 피처 및 외부 변수를 자동으로 처리합니다.
    """
    product_df = df[df['품목명'] == product_name].copy()
    
    # Lag 및 Rolling Window 특징 생성
    mean_sales = product_df['판매량'].mean()
    for lag in [1, 7, 14]:
        product_df[f'판매량_lag_{lag}'] = product_df['판매량'].shift(lag).fillna(mean_sales)
    
    for window in [7, 14]:
        product_df[f'판매량_roll_mean_{window}'] = product_df['판매량'].rolling(window=window, min_periods=1).mean()
        product_df[f'판매량_roll_std_{window}'] = product_df['판매량'].rolling(window=window, min_periods=1).std().fillna(0)
    
    product_df['요일별_평균_판매량'] = product_df.groupby('요일')['판매량'].transform('mean')
    
    # 학습에 사용할 특징 목록 정의 (분석에 불필요한 컬럼 제외)
    features_to_exclude = ['날짜', '품목명', '판매량']
    features = product_df.drop(columns=features_to_exclude, errors='ignore')
    
    # 범주형 피처(object 타입)를 원-핫 인코딩으로 자동 변환
    categorical_cols = features.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        dummies = pd.get_dummies(features[categorical_cols], prefix=categorical_cols, drop_first=True, dtype=int)
        features = features.drop(columns=categorical_cols).join(dummies)

    target = product_df['판매량']
    
    # XGBoost가 처리할 수 없는 컬럼명 문자 변환
    features.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in features.columns]
    
    # 결측치 처리 및 무한대 값 처리 (안전장치)
    features = features.bfill().fillna(0)
    features.replace([np.inf, -np.inf], 0, inplace=True)
    
    return features, target