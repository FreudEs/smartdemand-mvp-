# data_handler.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

def create_sample_data():
    """(수정) 품목별로 다른 패턴을 가진 샘플 데이터를 생성합니다."""
    print("현실적인 샘플 데이터를 생성 중입니다...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    
    sample_data = []
    products_info = {
        '아메리카노': {'base': 120, 'weekday_effect': [1.0, 0.95, 1.05, 1.1, 1.4, 1.3, 1.15]}, # 금요일 강세
        '카페라떼': {'base': 90, 'weekday_effect': [1.2, 1.1, 1.0, 0.9, 1.0, 1.1, 1.15]},    # 월요일 강세
        '베이글': {'base': 70, 'weekday_effect': [1.0, 1.0, 1.0, 1.2, 1.3, 1.5, 1.45]}       # 토요일 강세
    }
    
    for product, info in products_info.items():
        base_demand = info['base']
        weekday_effect_pattern = info['weekday_effect']
        month_seasonality = [0.9, 0.85, 1.0, 1.1, 1.15, 1.2, 1.25, 1.2, 1.1, 1.0, 0.95, 0.9]
        
        for date in date_range:
            weekday_effect = weekday_effect_pattern[date.weekday()]
            month_effect = month_seasonality[date.month - 1]
            trend_effect = 1.0 + 0.0005 * (date - start_date).days
            noise = np.random.normal(1.0, 0.1)
            
            sales = int(base_demand * weekday_effect * month_effect * trend_effect * noise)
            
            sample_data.append({
                '날짜': date, '품목명': product, '판매량': max(0, sales)
            })
            
    df = pd.DataFrame(sample_data)
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜').reset_index(drop=True)
    print("샘플 데이터 생성이 완료되었습니다.")
    return df