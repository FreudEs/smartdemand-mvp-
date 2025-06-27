# model_trainer.py (최종 수정 버전)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from feature_engineer import create_date_features, create_xgboost_features
from optimizer import optimize_xgboost

MIN_DATA_DAYS = 30

def train_prophet_model(df, product_name):
    """
    Prophet 모델을 훈련하고 예측 성능을 평가합니다.
    """
    # 필요한 데이터만 추출
    product_df_full = df[df['품목명'] == product_name].copy()
    prophet_data = product_df_full[['날짜', '판매량']].rename(columns={'날짜': 'ds', '판매량': 'y'})
    
    # 데이터 검증
    if len(prophet_data) < MIN_DATA_DAYS:
        print(f"  ⚠️ {product_name}: 데이터 부족 ({len(prophet_data)}일 < {MIN_DATA_DAYS}일)")
        return None, None, None, {'Accuracy(%)': -1, 'MAE': -1}
    
    # 훈련/테스트 분할
    test_days = min(30, int(len(prophet_data) * 0.2))
    train_data = prophet_data.iloc[:-test_days]
    test_data = prophet_data.iloc[-test_days:]
    
    try:
        # Prophet 모델 생성
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.1,
            interval_width=0.95
        )
        
        # 휴일/이벤트 처리 (있는 경우)
        if '이벤트' in product_df_full.columns:
            holidays_df = product_df_full[product_df_full['이벤트'].notna()][['날짜', '이벤트']]
            if not holidays_df.empty:
                holidays_df = holidays_df.rename(columns={'날짜': 'ds', '이벤트': 'holiday'})
                model = Prophet(
                    holidays=holidays_df,
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.1
                )
        
        # 모델 학습
        model.fit(train_data)
        
        # 테스트 예측
        future = model.make_future_dataframe(periods=test_days, freq='D')
        forecast = model.predict(future)
        
        # 성능 평가
        y_true = test_data['y'].values
        y_pred_test = forecast.tail(test_days)['yhat'].values
        y_pred_test = np.maximum(y_pred_test, 0)  # 음수 제거
        
        mae = mean_absolute_error(y_true, y_pred_test)
        mape = np.mean(np.abs((y_true - y_pred_test) / np.maximum(y_true, 1))) * 100
        performance = {'Accuracy(%)': 100 - mape, 'MAE': mae}
        
        # 미래 7일 예측
        final_future = model.make_future_dataframe(periods=7, freq='D')
        final_forecast = model.predict(final_future)
        final_forecast['yhat'] = np.maximum(final_forecast['yhat'], 0)
        
        print(f"  ✅ {product_name}: Prophet 학습 완료 (정확도: {performance['Accuracy(%)']:.1f}%)")
        
        return model, final_forecast, y_pred_test, performance
        
    except Exception as e:
        print(f"  ❌ {product_name}: Prophet 학습 실패 - {str(e)}")
        return None, None, None, {'Accuracy(%)': -1, 'MAE': -1}


def train_xgboost_model(features, target):
    """
    XGBoost 모델을 훈련합니다.
    """
    if len(features) < MIN_DATA_DAYS + 30:
        print(f"  ⚠️ XGBoost: 데이터 부족 ({len(features)}행)")
        return None, None, {'Accuracy(%)': -1, 'MAE': -1}
    
    try:
        # 훈련/테스트 분할
        test_days = 30
        X_test, y_test = features.iloc[-test_days:], target.iloc[-test_days:]
        X_train, y_train = features.iloc[:-test_days], target.iloc[:-test_days]
        
        # Optuna 최적화
        print(f"  🔍 XGBoost 하이퍼파라미터 최적화 중...")
        best_params = optimize_xgboost(X_train, y_train)
        
        # 모델 학습
        final_model = xgb.XGBRegressor(**best_params, random_state=42)
        final_model.fit(X_train, y_train, verbose=False)
        
        # 예측 및 평가
        y_pred_test = np.maximum(final_model.predict(X_test), 0)
        mae = mean_absolute_error(y_test.values, y_pred_test)
        mape = np.mean(np.abs((y_test.values - y_pred_test) / np.maximum(y_test.values, 1))) * 100
        performance = {'Accuracy(%)': 100 - mape, 'MAE': mae}
        
        print(f"  ✅ XGBoost 학습 완료 (정확도: {performance['Accuracy(%)']:.1f}%)")
        
        return final_model, y_pred_test, performance
        
    except Exception as e:
        print(f"  ❌ XGBoost 학습 실패 - {str(e)}")
        return None, None, {'Accuracy(%)': -1, 'MAE': -1}


def predict_xgboost_future(df, product_name, xgb_model, days_to_predict=7):
    """
    XGBoost 모델로 미래를 예측합니다.
    특징 불일치 문제를 해결한 버전
    """
    if xgb_model is None:
        return pd.DataFrame()
    
    try:
        # 1. 학습 시 사용된 특징 이름 가져오기
        try:
            expected_features = xgb_model.get_booster().feature_names
            if expected_features is None:
                # feature_names가 None인 경우 n_features_in_를 사용
                n_features = xgb_model.n_features_in_
                expected_features = [f'f{i}' for i in range(n_features)]
                print(f"    경고: feature_names가 None입니다. 특징 수: {n_features}")
            else:
                print(f"    예상 특징 수: {len(expected_features)}")
        except:
            # 안전장치: 모든 특징 사용
            expected_features = None
            print("    경고: 특징 이름을 가져올 수 없습니다.")
        
        # 2. 과거 데이터 준비
        historical_df = df[df['품목명'] == product_name].copy()
        last_date = historical_df['날짜'].max()
        future_predictions = []
        
        # 3. 반복적으로 1일씩 예측
        for i in range(days_to_predict):
            next_date = last_date + pd.Timedelta(days=i + 1)
            
             # [핵심 수정] historical_df의 마지막 행을 복사하여 모든 피처 컬럼을 유지
            # 이렇게 하면 is_anomaly 등 모든 컬럼이 자동으로 포함됨
            next_day_row = historical_df.iloc[-1:].copy()
            next_day_row['날짜'] = next_date
            next_day_row['판매량'] = 0  # 예측을 위해 임시로 0을 설정
            
            # 기존 데이터와 결합
            temp_df = pd.concat([historical_df, next_day_row], ignore_index=True)
            
            # 날짜 특징 생성
            temp_df = create_date_features(temp_df)
            
            # anomaly 관련 컬럼이 없다면 기본값으로 추가
            if 'is_anomaly' not in temp_df.columns:
                temp_df['is_anomaly'] = False
            if 'anomaly_type' not in temp_df.columns:
                temp_df['anomaly_type'] = '정상'
            if 'anomaly_reason' not in temp_df.columns:
                temp_df['anomaly_reason'] = '정상'
            if 'anomaly_score' not in temp_df.columns:
                temp_df['anomaly_score'] = 0.0
            
            # 다른 외부 변수들도 forward fill
            external_vars = ['예측수요', '재고량', '기온', '강수량', '이벤트']
            for var in external_vars:
                if var in temp_df.columns:
                    temp_df[var] = temp_df[var].fillna(method='ffill').fillna(method='bfill')
            
            # 특징 생성
            features, _ = create_xgboost_features(temp_df, product_name)
            
            # 마지막 행(미래 날짜)의 특징만 추출
            last_features = features.tail(1)
            
            # 특징 순서 맞추기 (중요!)
            if expected_features:
                # 학습 시 사용된 특징만 선택하고 순서도 맞춤
                missing_features = set(expected_features) - set(last_features.columns)
                if missing_features:
                    print(f"    경고: 누락된 특징 {missing_features}")
                    # 누락된 특징은 0으로 채움
                    for feat in missing_features:
                        last_features[feat] = 0
                
                # 순서 맞추기
                last_features = last_features[expected_features]
            
            # 예측
            next_pred = max(0, xgb_model.predict(last_features)[0])
            
            # 예측값을 historical_df에 추가
            new_row = pd.DataFrame({
                '날짜': [next_date],
                '품목명': [product_name],
                '판매량': [next_pred],
                'is_anomaly': [False],  # 미래는 정상으로 가정
                'anomaly_type': ['정상'],
                'anomaly_reason': ['정상'],
                'anomaly_score': [0.0]
            })
            
            # 외부 변수도 추가 (있는 경우)
            for var in external_vars:
                if var in historical_df.columns:
                    new_row[var] = historical_df[var].iloc[-1]  # 마지막 값 사용
            
            historical_df = pd.concat([historical_df, new_row], ignore_index=True)
            
            future_predictions.append({'날짜': next_date, '판매량': next_pred})
        
        print(f"    ✅ XGBoost 미래 예측 성공")
        return pd.DataFrame(future_predictions)
        
    except Exception as e:
        print(f"  ❌ XGBoost 미래 예측 실패 - {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()