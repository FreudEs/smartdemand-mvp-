# anomaly_analyzer.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class SmartDemandPreprocessor:
    """
    SmartDemand 시스템을 위한 고급 데이터 전처리 클래스.
    이상치 및 결측치를 감지하고, 원인을 추정하며, 모델 학습을 위한 피처를 생성합니다.
    """

    def __init__(self, z_score_threshold: float = 2.5):
        """
        전처리기 초기화

        Args:
            z_score_threshold (float): 이상치를 판단하기 위한 Z-score 임계값.
        """
        self.z_score_threshold = z_score_threshold
        self.anomaly_log = None

    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        전체 데이터 전처리 파이프라인을 실행합니다.

        Args:
            df (pd.DataFrame): 원본 데이터프레임. 
                               필수 컬럼: ['날짜', '품목명', '판매량']
                               선택 컬럼: ['이벤트', '프로모션', '날씨', '기온', '재고']

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - feature가 추가된 전처리 완료 데이터프레임
                - 이상 현상 로그 데이터프레임
        """
        if not all(col in df.columns for col in ['날짜', '품목명', '판매량']):
            raise ValueError("데이터에 '날짜', '품목명', '판매량' 컬럼이 반드시 포함되어야 합니다.")

        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values(by=['품목명', '날짜']).reset_index(drop=True)

        all_anomalies = []
        for product_name, group in df.groupby('품목명'):
            anomalies = self._detect_anomalies_and_missing(group.copy())
            all_anomalies.append(anomalies)

        if all_anomalies:
            self.anomaly_log = pd.concat(all_anomalies, ignore_index=True)
        else:
            self.anomaly_log = pd.DataFrame()

        processed_df = self._create_features_from_log(df)

        return processed_df, self.anomaly_log

    def _detect_anomalies_and_missing(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """단일 품목 그룹에 대해 이상치와 결측치를 감지하고 원인을 추정합니다."""
        group_df['z_score'] = self._calculate_z_score(group_df['판매량'])
        
        anomaly_mask = (group_df['z_score'].abs() > self.z_score_threshold) | (group_df['판매량'].isna())
        anomalies_df = group_df[anomaly_mask].copy()

        if anomalies_df.empty:
            return pd.DataFrame()

        anomalies_df['이상_유형'] = anomalies_df.apply(self._define_anomaly_type, axis=1)
        anomalies_df['추정_원인'] = anomalies_df.apply(self._infer_reason, axis=1)
        anomalies_df['이벤트_정보'] = anomalies_df.get('이벤트', pd.Series(index=anomalies_df.index, dtype=str)).fillna('없음')

        log_columns = ['날짜', '품목명', '판매량', '이상_유형', '이벤트_정보', '추정_원인', 'z_score']
        return anomalies_df.reindex(columns=log_columns)
    
    @staticmethod
    def _calculate_z_score(series: pd.Series) -> pd.Series:
        """판매량 시리즈에 대한 Z-score를 계산합니다."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std

    def _define_anomaly_type(self, row: pd.Series) -> str:
        """이상 현상의 유형(급등, 급감, 결측)을 정의합니다."""
        if pd.isna(row['판매량']):
            return '결측'
        elif row['z_score'] > self.z_score_threshold:
            return '급등'
        elif row['z_score'] < -self.z_score_threshold:
            return '급감'
        return '정상'

    @staticmethod
    def _infer_reason(row: pd.Series) -> str:
        """규칙 기반으로 이상 현상의 원인을 추정합니다."""
        event = row.get('이벤트', None)
        promotion = row.get('프로모션', None)
        weather = row.get('날씨', None)
        anomaly_type = row['이상_유형']

        if anomaly_type == '결측':
            return '품절 또는 집계 누락'

        if (promotion == 1 or (isinstance(event, str) and '할인' in event)) and anomaly_type == '급등':
            return '프로모션 영향'
        
        if isinstance(weather, str):
            if any(w in weather for w in ['비', '눈', '폭우', '폭설']) and anomaly_type == '급감':
                return '기상 악화 영향'
            if any(w in weather for w in ['맑음', '화창']) and row.get('기온', 15) > 25 and anomaly_type == '급등':
                return '더운 날씨 특수'

        if pd.notna(event) and event != '없음':
            if any(h in event for h in ['명절', '연휴', '월드컵']) and anomaly_type == '급등':
                return '특별 이벤트 특수'
            return f"'{event}' 이벤트 영향"
        
        return '통계적 이상치 (원인 불명)'

    def _create_features_from_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """분석된 이상 현상 로그를 바탕으로 원본 데이터에 피처를 추가합니다."""
        if self.anomaly_log is None or self.anomaly_log.empty:
            df['is_anomaly'] = False
            df['anomaly_type'] = '정상'
            df['anomaly_reason'] = '정상'
            df['anomaly_score'] = 0.0
            return df

        anomaly_features = self.anomaly_log[['날짜', '품목명', '이상_유형', '추정_원인', 'z_score']].copy()
        anomaly_features.rename(columns={
            '이상_유형': 'anomaly_type',
            '추정_원인': 'anomaly_reason',
            'z_score': 'anomaly_score'
        }, inplace=True)
        anomaly_features['is_anomaly'] = True

        merged_df = pd.merge(df, anomaly_features, on=['날짜', '품목명'], how='left')

        merged_df['is_anomaly'].fillna(False, inplace=True)
        merged_df['anomaly_type'].fillna('정상', inplace=True)
        merged_df['anomaly_reason'].fillna('정상', inplace=True)
        merged_df['anomaly_score'].fillna(0.0, inplace=True)
        return merged_df
