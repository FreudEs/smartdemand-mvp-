# advanced_models.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 변동성이 큰 데이터에 강한 추가 모델들

class LSTMPredictor:
    """
    LSTM (Long Short-Term Memory) 모델
    장점: 장기 의존성을 학습하여 복잡한 시계열 패턴 포착
    특히 변동성이 크고 비선형적인 패턴에 강함
    """
    
    def __init__(self, lookback_days: int = 14, forecast_days: int = 7):
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 데이터를 LSTM 입력 형태로 변환"""
        X, y = [], []
        for i in range(self.lookback_days, len(data) - self.forecast_days + 1):
            X.append(data[i-self.lookback_days:i])
            y.append(data[i:i+self.forecast_days])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """LSTM 모델 구조 정의"""
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(25, activation='relu'),
            Dropout(0.2),
            Dense(self.forecast_days)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, df: pd.DataFrame, product_name: str) -> Dict:
        """LSTM 모델 학습"""
        product_df = df[df['품목명'] == product_name].copy()
        
        if len(product_df) < self.lookback_days + self.forecast_days + 30:
            return {'trained': False, 'message': '데이터 부족'}
        
        # 데이터 정규화
        sales_data = product_df['판매량'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(sales_data)
        
        # 시퀀스 생성
        X, y = self.prepare_sequences(scaled_data.flatten())
        
        # 훈련/테스트 분할
        test_size = min(30, int(len(X) * 0.2))
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # 모델 구축 및 학습
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        self.model = self.build_model((X_train.shape[1], 1))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 성능 평가
        y_pred = self.model.predict(X_test, verbose=0)
        
        # 역정규화
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / np.maximum(y_test_original, 1))) * 100
        
        return {
            'trained': True,
            'performance': {
                'Accuracy(%)': 100 - mape,
                'MAE': mae
            },
            'history': history.history,
            'predictions': y_pred_original
        }
    
    def predict_future(self, df: pd.DataFrame, product_name: str) -> pd.DataFrame:
        """미래 7일 예측"""
        if self.model is None:
            return pd.DataFrame()
        
        product_df = df[df['품목명'] == product_name].copy()
        
        # 마지막 lookback_days 데이터로 예측
        last_sequence = product_df['판매량'].values[-self.lookback_days:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        # 예측
        X_future = last_sequence_scaled.reshape((1, self.lookback_days, 1))
        future_pred_scaled = self.model.predict(X_future, verbose=0)
        future_pred = self.scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
        
        # 음수 제거
        future_pred = np.maximum(future_pred, 0)
        
        # 데이터프레임 생성
        last_date = product_df['날짜'].max()
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(self.forecast_days)]
        
        return pd.DataFrame({
            '날짜': future_dates,
            '판매량': future_pred[:self.forecast_days]
        })


class LightGBMPredictor:
    """
    LightGBM 모델
    장점: XGBoost보다 빠르고 메모리 효율적, 범주형 변수 자동 처리
    변동성이 큰 데이터에서도 안정적인 성능
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        
    def train(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """LightGBM 모델 학습"""
        if len(features) < 60:
            return {'trained': False, 'message': '데이터 부족'}
        
        # 훈련/테스트 분할
        test_size = min(30, int(len(features) * 0.2))
        X_train, X_test = features[:-test_size], features[-test_size:]
        y_train, y_test = target[:-test_size], target[-test_size:]
        
        # 하이퍼파라미터 (변동성 대응 강화)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.1,
            'verbose': -1
        }
        
        # 학습
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # 예측 및 평가
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred = np.maximum(y_pred, 0)  # 음수 제거
        
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        
        # 특성 중요도 저장
        self.feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return {
            'trained': True,
            'performance': {
                'Accuracy(%)': 100 - mape,
                'MAE': mae
            },
            'best_iteration': self.model.best_iteration,
            'feature_importance': self.feature_importance,
            'predictions': y_pred
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if self.model is None:
            return np.array([])
        
        predictions = self.model.predict(features, num_iteration=self.model.best_iteration)
        return np.maximum(predictions, 0)


class VolatilityAwareEnsemble:
    """
    변동성을 고려한 동적 앙상블 모델
    변동성이 높을 때는 LSTM의 가중치를 높이고,
    안정적일 때는 전통적 모델의 가중치를 높입니다.
    """
    
    def __init__(self):
        self.weights = {}
        
    def calculate_dynamic_weights(
        self, 
        volatility_score: float,
        model_performances: Dict[str, float]
    ) -> Dict[str, float]:
        """
        변동성과 모델 성능에 따라 동적 가중치 계산
        
        Args:
            volatility_score: 0~1 사이의 변동성 점수 (1에 가까울수록 변동성 높음)
            model_performances: 각 모델의 정확도
        """
        if not model_performances:
            return {}
            
        # 기본 가중치: 성능 기반
        total_performance = sum(model_performances.values())
        if total_performance == 0:
             # 모든 모델의 성능이 0일 경우, 균등하게 가중치 배분
            num_models = len(model_performances)
            return {model: 1.0 / num_models for model in model_performances}

        base_weights = {
            model: perf / total_performance 
            for model, perf in model_performances.items()
        }
        
        # 변동성 조정
        # 변동성이 높을수록 LSTM/LightGBM에 더 많은 가중치
        volatility_adjustment = {
            'prophet': 1 - volatility_score * 0.3,  # 변동성 높을수록 감소
            'xgboost': 1 - volatility_score * 0.2,
            'lightgbm': 1 + volatility_score * 0.2,  # 변동성 높을수록 증가
            'lstm': 1 + volatility_score * 0.3
        }
        
        # 조정된 가중치 계산
        adjusted_weights = {}
        for model in base_weights:
            if model in volatility_adjustment:
                adjusted_weights[model] = base_weights[model] * volatility_adjustment[model]
            else:
                adjusted_weights[model] = base_weights[model]
        
        # 정규화
        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            num_models = len(adjusted_weights)
            return {model: 1.0 / num_models for model in adjusted_weights}
            
        normalized_weights = {
            model: weight / total_weight 
            for model, weight in adjusted_weights.items()
        }
        
        self.weights = normalized_weights
        return normalized_weights
    
    def ensemble_predict(
        self, 
        predictions: Dict[str, np.ndarray],
        volatility_score: float,
        model_performances: Dict[str, float]
    ) -> np.ndarray:
        """
        여러 모델의 예측을 앙상블
        """
        weights = self.calculate_dynamic_weights(volatility_score, model_performances)
        
        # 가중 평균
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model, pred in predictions.items():
            if model in weights:
                ensemble_pred += pred * weights[model]
        
        return ensemble_pred


# 통합 사용 예시
def train_advanced_models(df: pd.DataFrame, product_name: str, features: pd.DataFrame, target: pd.Series):
    """
    추가 모델들을 학습하고 결과를 반환합니다.
    
    Returns:
        Dict: 각 모델의 학습 결과 및 성능
    """
    results = {}
    
    # 1. LSTM 학습
    print(f"  🧠 {product_name}: LSTM 모델 학습 중...")
    lstm_model = LSTMPredictor(lookback_days=14, forecast_days=7)
    lstm_result = lstm_model.train(df, product_name)
    results['lstm'] = {
        'model': lstm_model,
        'result': lstm_result
    }
    
    # 2. LightGBM 학습
    print(f"  💡 {product_name}: LightGBM 모델 학습 중...")
    lgb_model = LightGBMPredictor()
    lgb_result = lgb_model.train(features, target)
    results['lightgbm'] = {
        'model': lgb_model,
        'result': lgb_result
    }
    
    return results