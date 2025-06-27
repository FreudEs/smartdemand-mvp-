# external_factor_detector.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ExternalFactorDetector:
    """
    판매량 변동성을 분석하여 외부 변수를 자동으로 생성하는 클래스
    변동성이 큰 패턴을 감지하고 이를 설명할 수 있는 가상의 외부 변수를 생성합니다.
    """
    
    def __init__(self, volatility_threshold: float = 0.5):
        """
        Args:
            volatility_threshold: 변동성을 판단하는 임계값 (변동계수 기준)
        """
        self.volatility_threshold = volatility_threshold
        self.detected_factors = {}
        
    def detect_and_create_factors(self, df: pd.DataFrame, product_name: str) -> pd.DataFrame:
        """
        품목별 데이터를 분석하여 외부 변수를 자동 생성합니다.
        
        Returns:
            외부 변수가 추가된 데이터프레임
        """
        product_df = df[df['품목명'] == product_name].copy()
        
        # 1. 변동성 분석
        volatility_info = self._analyze_volatility(product_df)
        
        # 2. 계절성 패턴 감지
        product_df = self._detect_seasonality_patterns(product_df)
        
        # 3. 급격한 변화점 감지 (Change Point Detection)
        product_df = self._detect_change_points(product_df)
        
        # 4. 주기적 패턴 감지
        product_df = self._detect_cyclic_patterns(product_df)
        
        # 5. 트렌드 변화 감지
        product_df = self._detect_trend_shifts(product_df)
        
        # 6. 요일별 특수성 강화
        product_df = self._enhance_weekday_effects(product_df)
        
        # 7. 변동성 기반 신뢰도 가중치 생성
        product_df = self._create_volatility_weights(product_df, volatility_info)
        
        # 감지된 패턴 저장
        self.detected_factors[product_name] = {
            'volatility': volatility_info,
            'patterns': self._summarize_patterns(product_df)
        }
        
        return product_df
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """판매량의 변동성을 분석합니다."""
        sales = df['판매량'].values
        
        # 변동계수 (CV: Coefficient of Variation)
        cv = np.std(sales) / np.mean(sales) if np.mean(sales) > 0 else 0
        
        # 이동 표준편차
        rolling_std = df['판매량'].rolling(window=7, min_periods=1).std()
        
        # 변동성 레벨 분류
        if cv < 0.2:
            volatility_level = "매우 안정"
        elif cv < 0.5:
            volatility_level = "안정"
        elif cv < 1.0:
            volatility_level = "변동성 있음"
        else:
            volatility_level = "매우 변동적"
        
        return {
            'cv': cv,
            'level': volatility_level,
            'mean_rolling_std': rolling_std.mean(),
            'max_rolling_std': rolling_std.max()
        }
    
    def _detect_seasonality_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """계절성 패턴을 감지하고 관련 변수를 생성합니다."""
        df = df.copy()
        
        # 월별 평균과의 편차
        monthly_avg = df.groupby('월')['판매량'].transform('mean')
        df['월별_편차'] = df['판매량'] / monthly_avg - 1
        
        # 주차별 패턴
        weekly_avg = df.groupby('주차')['판매량'].transform('mean')
        df['주차별_편차'] = df['판매량'] / weekly_avg - 1
        
        # 계절별 특수 이벤트 가능성 점수
        # 여름(6-8월), 겨울(12-2월)에 특수 패턴이 있는지 확인
        df['여름_특수'] = ((df['월'].isin([6, 7, 8])) & 
                          (df['판매량'] > df['판매량'].mean() * 1.2)).astype(int)
        df['겨울_특수'] = ((df['월'].isin([12, 1, 2])) & 
                          (df['판매량'] > df['판매량'].mean() * 1.2)).astype(int)
        
        return df
    
    def _detect_change_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """급격한 변화점을 감지합니다."""
        df = df.copy()
        
        # 전일 대비 변화율
        df['일간_변화율'] = df['판매량'].pct_change().fillna(0)
        
        # 급등/급락 감지 (2 표준편차 이상)
        change_std = df['일간_변화율'].std()
        change_mean = df['일간_변화율'].mean()
        
        df['급등_신호'] = (df['일간_변화율'] > change_mean + 2*change_std).astype(int)
        df['급락_신호'] = (df['일간_변화율'] < change_mean - 2*change_std).astype(int)
        
        # 변화점 강도 (얼마나 급격한지)
        df['변화_강도'] = np.abs(df['일간_변화율']) / change_std if change_std > 0 else 0
        
        return df
    
    def _detect_cyclic_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """주기적 패턴을 감지합니다."""
        df = df.copy()
        
        # 7일 주기 (주간 패턴)
        df['주간_사이클'] = np.sin(2 * np.pi * df.index / 7)
        df['주간_사이클_cos'] = np.cos(2 * np.pi * df.index / 7)
        
        # 14일 주기 (격주 패턴)
        df['격주_사이클'] = np.sin(2 * np.pi * df.index / 14)
        
        # 30일 주기 (월간 패턴)
        df['월간_사이클'] = np.sin(2 * np.pi * df['일'] / 30)
        
        return df
    
    def _detect_trend_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """트렌드 변화를 감지합니다."""
        df = df.copy()
        
        # 이동평균 기반 트렌드
        df['MA7'] = df['판매량'].rolling(window=7, min_periods=1).mean()
        df['MA14'] = df['판매량'].rolling(window=14, min_periods=1).mean()
        
        # 트렌드 방향 (상승/하락)
        df['트렌드_방향'] = (df['MA7'] > df['MA14']).astype(int)
        
        # 트렌드 강도
        df['트렌드_강도'] = (df['MA7'] - df['MA14']) / df['MA14'].replace(0, 1)
        
        # 트렌드 전환점
        df['트렌드_전환'] = df['트렌드_방향'].diff().fillna(0).abs()
        
        return df
    
    def _enhance_weekday_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """요일별 효과를 강화합니다."""
        df = df.copy()
        
        # 요일별 평균 대비 비율
        weekday_avg = df.groupby('요일')['판매량'].transform('mean')
        overall_avg = df['판매량'].mean()
        
        df['요일_효과_비율'] = weekday_avg / overall_avg
        
        # 주말 효과 (금토일)
        df['주말_효과'] = df['요일'].isin([4, 5, 6]).astype(int)
        
        # 월요일 효과
        df['월요일_효과'] = (df['요일'] == 0).astype(int)
        
        # 금요일 효과
        df['금요일_효과'] = (df['요일'] == 4).astype(int)
        
        return df
    
    def _create_volatility_weights(self, df: pd.DataFrame, volatility_info: Dict) -> pd.DataFrame:
        """변동성 기반 신뢰도 가중치를 생성합니다."""
        df = df.copy()
        
        # 기본 신뢰도 (변동성이 클수록 낮은 신뢰도)
        base_confidence = 1.0 - min(volatility_info['cv'], 1.0)
        
        # 이동 변동성 기반 동적 신뢰도
        rolling_cv = df['판매량'].rolling(window=7, min_periods=1).apply(
            lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0
        )
        
        df['예측_신뢰도'] = base_confidence * (1 - rolling_cv.fillna(0).clip(0, 1))
        
        # 데이터 품질 점수
        df['데이터_품질_점수'] = df['예측_신뢰도'] * (1 - df['변화_강도'].clip(0, 1))
        
        return df
    
    def _summarize_patterns(self, df: pd.DataFrame) -> Dict:
        """감지된 패턴을 요약합니다."""
        return {
            '급등_횟수': df['급등_신호'].sum(),
            '급락_횟수': df['급락_신호'].sum(),
            '트렌드_전환_횟수': df['트렌드_전환'].sum(),
            '주말_효과_강도': df[df['주말_효과'] == 1]['판매량'].mean() / df['판매량'].mean() if df['판매량'].mean() > 0 else 1,
            '평균_신뢰도': df['예측_신뢰도'].mean()
        }
    
    def get_factor_importance(self, product_name: str) -> pd.DataFrame:
        """생성된 외부 변수들의 중요도를 반환합니다."""
        if product_name not in self.detected_factors:
            return pd.DataFrame()
        
        patterns = self.detected_factors[product_name]['patterns']
        volatility = self.detected_factors[product_name]['volatility']
        
        importance_data = {
            '변수명': ['변동성 수준', '급등 패턴', '급락 패턴', '트렌드 전환', '주말 효과'],
            '중요도': [
                volatility['cv'],
                patterns['급등_횟수'] / 100,  # 정규화
                patterns['급락_횟수'] / 100,
                patterns['트렌드_전환_횟수'] / 50,
                abs(patterns['주말_효과_강도'] - 1)
            ],
            '설명': [
                volatility['level'],
                f"{patterns['급등_횟수']}회 발생",
                f"{patterns['급락_횟수']}회 발생",
                f"{patterns['트렌드_전환_횟수']}회 전환",
                f"평균 대비 {patterns['주말_효과_강도']:.2f}배"
            ]
        }
        
        return pd.DataFrame(importance_data).sort_values('중요도', ascending=False)


# 사용 예시
def enhance_data_with_external_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 품목에 대해 외부 변수를 자동 생성합니다.
    
    사용법:
    enhanced_df = enhance_data_with_external_factors(df)
    """
    detector = ExternalFactorDetector(volatility_threshold=0.5)
    
    enhanced_dfs = []
    for product in df['품목명'].unique():
        product_enhanced = detector.detect_and_create_factors(df, product)
        enhanced_dfs.append(product_enhanced)
        
        # 중요도 출력 (디버깅용)
        print(f"\n{product} 외부 변수 중요도:")
        importance_df = detector.get_factor_importance(product)
        if not importance_df.empty:
            print(importance_df.to_string(index=False))
    
    return pd.concat(enhanced_dfs, ignore_index=True)