# parallel_processor.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# --- 기존 모듈 ---
from model_trainer import train_prophet_model, train_xgboost_model, predict_xgboost_future
from feature_engineer import create_xgboost_features
from anomaly_analyzer import SmartDemandPreprocessor

# --- [신규] 고급 기능 모듈 임포트 ---
from external_factor_detector import ExternalFactorDetector
from advanced_models import LSTMPredictor, LightGBMPredictor, VolatilityAwareEnsemble

def process_product(args):
    """
    [수정] 단일 품목에 대한 전체 분석 파이프라인.
    외부 요인 탐지, 이상 현상 분석, 4개 모델 학습 및 동적 앙상블을 수행합니다.
    """
    product_name, df_product_orig = args
    print(f"🔧 {product_name} 처리 시작...")
    
    try:
        # --- 1. 외부 요인 자동 탐지 및 변동성 분석 ---
        factor_detector = ExternalFactorDetector()
        df_enhanced = factor_detector.detect_and_create_factors(df_product_orig, product_name)
        volatility_info = factor_detector.detected_factors[product_name]['volatility']
        print(f"  ⚡️ {product_name}: 변동성({volatility_info['level']}) 분석 및 외부 요인 피처 생성 완료.")

        # --- 2. 이상 현상 원인 분석 ---
        anomaly_preprocessor = SmartDemandPreprocessor()
        df_processed, anomaly_log = anomaly_preprocessor.process(df_enhanced)
        if not anomaly_log.empty:
            print(f"  🕵️‍♂️ {product_name}: {len(anomaly_log)}개의 이상 현상 감지 및 원인 분석 완료.")
        
        # --- 3. 기본 모델 학습 (Prophet, XGBoost) ---
        features, target = create_xgboost_features(df_processed, product_name)
        prophet_model, prophet_fc, prophet_test_pred, prophet_perf = train_prophet_model(df_processed, product_name)
        xgb_model, xgb_test_pred, xgb_perf = train_xgboost_model(features, target)

        # --- 4. 고급 모델 학습 (LSTM, LightGBM) ---
        lstm_model = LSTMPredictor()
        lstm_result = lstm_model.train(df_processed, product_name)
        
        lgbm_model = LightGBMPredictor()
        lgbm_result = lgbm_model.train(features, target)

        # --- 5. 동적 앙상블 ---
        all_performances = {
            'prophet': prophet_perf.get('Accuracy(%)', 0),
            'xgboost': xgb_perf.get('Accuracy(%)', 0),
            'lstm': lstm_result.get('performance', {}).get('Accuracy(%)', 0),
            'lightgbm': lgbm_result.get('performance', {}).get('Accuracy(%)', 0),
        }
        
        # 앙상블에 사용할 모든 모델의 테스트 예측값 수집
        all_test_preds = {
            'prophet': prophet_test_pred if prophet_test_pred is not None else np.array([]),
            'xgboost': xgb_test_pred if xgb_test_pred is not None else np.array([]),
            'lstm': lstm_result.get('predictions', np.array([])),
            'lightgbm': lgbm_result.get('predictions', np.array([]))
        }

        # 예측값이 있는 모델만 필터링하고 길이를 통일
        valid_preds = {name: pred for name, pred in all_test_preds.items() if len(pred) > 0}
        
        if not valid_preds:
            raise ValueError(f"{product_name}: 예측 가능한 모델이 없습니다.")
            
        min_len = min(len(pred) for pred in valid_preds.values())
        
        # 길이가 통일된 예측 결과 딕셔너리를 최종적으로 생성
        all_predictions = {
            name: pred[-min_len:] for name, pred in valid_preds.items()
        }

        # all_predictions에 포함된 모델들의 성능만으로 가중치를 다시 계산
        performance_for_ensemble = {
            model_name: perf
            for model_name, perf in all_performances.items()
            if model_name in all_predictions
        }

        # 테스트 데이터셋에 대한 앙상블 예측 및 성능 평가
        y_true_test = target.iloc[-min_len:].values
        ensemble = VolatilityAwareEnsemble()
        
        ensemble_test_pred = ensemble.ensemble_predict(
            all_predictions,
            volatility_info['cv'],
            performance_for_ensemble
        )
        
        ens_mae = mean_absolute_error(y_true_test, ensemble_test_pred)
        ens_mape = np.mean(np.abs((y_true_test - ensemble_test_pred) / np.maximum(y_true_test, 1))) * 100
        ensemble_perf = {'Accuracy(%)': 100 - ens_mape, 'MAE': ens_mae, 'weights': ensemble.weights}
        
        print(f"  ✅ {product_name} 동적 앙상블 정확도: {ensemble_perf['Accuracy(%)']:.1f}%")

        # 미래 예측 (가장 성능 좋은 단일 모델 또는 앙상블 예측)
        xgb_future_pred_df = predict_xgboost_future(df_processed, product_name, xgb_model, 7)
        prophet_future_pred = prophet_fc.tail(7) if prophet_fc is not None else None
        
        ensemble_forecast = None
        if prophet_future_pred is not None and not xgb_future_pred_df.empty:
            prophet_weight = ensemble.weights.get('prophet', 0.5)
            xgb_weight = ensemble.weights.get('xgboost', 0.5)
            total_w = prophet_weight + xgb_weight
            prophet_weight /= total_w
            xgb_weight /= total_w

            prophet_future_pred['ds'] = pd.to_datetime(prophet_future_pred['ds']).dt.normalize()
            xgb_future_pred_df['날짜'] = pd.to_datetime(xgb_future_pred_df['날짜']).dt.normalize()
            prophet_future_pred = prophet_future_pred[['ds', 'yhat']].set_index('ds')
            xgb_future_pred = xgb_future_pred_df[['날짜', '판매량']].set_index('날짜').rename(columns={'판매량': 'yhat'})
            merged_preds = prophet_future_pred.join(xgb_future_pred, lsuffix='_prophet', rsuffix='_xgb', how='outer').fillna(0)
            merged_preds['ensemble_yhat'] = (merged_preds['yhat_prophet'] * prophet_weight) + (merged_preds['yhat_xgb'] * xgb_weight)
            ensemble_forecast = merged_preds[['ensemble_yhat']].reset_index().rename(columns={'index':'날짜'})


        print(f"  ✅ {product_name} 처리 완료!")
        
        return product_name, {
            'prophet_performance': prophet_perf, 
            'xgboost_performance': xgb_perf,
            'lstm_performance': lstm_result.get('performance', {}),
            'lightgbm_performance': lgbm_result.get('performance', {}),
            'ensemble_performance': ensemble_perf, 
            'ensemble_forecast': ensemble_forecast,
            'anomaly_log': anomaly_log
        }
    
    except Exception as e:
        print(f"  ❌ {product_name} 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return product_name, {}


def run_parallel_processing(_df):
    """
    모든 품목에 대한 모델 훈련을 병렬로 실행하고, 이상 현상 로그를 수집합니다.
    """
    models_data = {}
    all_anomaly_logs = []
    
    tasks = [(name, group) for name, group in _df.groupby('품목명')]
    
    print(f"🚀 총 {len(tasks)}개 품목에 대한 예측 모델 학습을 시작합니다...")
    
    # 병렬 처리 ( concurrent.futures 사용 )
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_product, task) for task in tasks]
    #     for future in as_completed(futures):
    #         name, data = future.result()
    #         if data:
    #             models_data[name] = data
    #             if 'anomaly_log' in data and not data['anomaly_log'].empty:
    #                 all_anomaly_logs.append(data['anomaly_log'])
    
    # 순차 처리 (디버깅용)
    for task in tasks:
        name, data = process_product(task)
        if data:
            models_data[name] = data
            if 'anomaly_log' in data and data['anomaly_log'] is not None and not data['anomaly_log'].empty:
                all_anomaly_logs.append(data['anomaly_log'])

    if all_anomaly_logs:
        models_data['total_anomaly_log'] = pd.concat(all_anomaly_logs, ignore_index=True)
    else:
        models_data['total_anomaly_log'] = pd.DataFrame()
    
    print(f"\n✅ 모든 품목 처리 완료! 성공: {len([v for n,v in models_data.items() if n != 'total_anomaly_log' and isinstance(v, dict) and v.get('ensemble_forecast') is not None])}개")
                
    return models_data