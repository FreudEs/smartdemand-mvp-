# optimizer.py
# 버전: v1.8 (호환성 최종 수정)
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Optuna 로그 레벨 설정 (너무 많은 로그 방지)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_xgboost(X, y):
    """
    시계열 교차 검증 및 Optuna를 사용하여 XGBoost의 최적 하이퍼파라미터를 찾습니다.
    수정: 오래된 라이브러리 버전과의 호환성을 위해 조기 종료 기능을 제거합니다.
    """
    def objective(trial):
        # Optuna가 제안하는 하이퍼파라미터
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'random_state': 42,
            'n_jobs': -1
        }

        # 시계열 교차 검증 설정
        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = XGBRegressor(**param)
            
            # [핵심 수정] 호환성 문제를 해결하기 위해 조기 종료 관련 인자(callbacks, early_stopping_rounds)를 모두 제거합니다.
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # 전체 모델로 예측 수행
            preds = np.maximum(model.predict(X_val), 0)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)
        
        return np.mean(rmses)

    # Optuna 스터디 생성 및 최적화 실행
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        objective, 
        n_trials=20,
        timeout=60,
        show_progress_bar=False
    )

    print(f"  ✅ 최적화 완료! 최적 RMSE: {study.best_value:.4f}")
    
    # 최적화된 하이퍼파라미터 반환
    return study.best_params
