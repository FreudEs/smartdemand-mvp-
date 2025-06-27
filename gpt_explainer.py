# gpt_explainer.py
import os
from openai import OpenAI
from typing import Dict, List, Optional
import json
import streamlit as st

class GPTExplainer:
    """
    LLM을 사용하여 수요 예측 결과를 비전문가도 이해할 수 있는 
    자연어로 해설하는 클래스.
    [수정] 이상 현상 분석 결과를 프롬프트에 포함하여 더 깊이 있는 분석을 제공.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Explainer 초기화
        """
        self.api_key = api_key or os.getenv('FIREWORKS_API_KEY')
        if not self.api_key:
            raise ValueError("Fireworks AI API 키가 필요합니다.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
        
    def generate_explanation(
        self,
        product_name: str,
        model_name: str,
        accuracy: float,
        mae: float,
        forecast_7day: float,
        avg_daily: float,
        anomaly_info: Optional[List[str]] = None  # [신규] 이상 현상 정보 인자 추가
    ) -> Dict[str, str]:
        """
        예측 결과에 대한 심층적인 경영 분석 및 조언을 생성합니다.
        """
        
        # [신규] anomaly_info가 있을 경우 프롬프트에 추가할 섹션 생성
        anomaly_prompt_section = ""
        if anomaly_info:
            anomaly_details = "\n".join([f"* {info}" for info in anomaly_info])
            anomaly_prompt_section = f"""
        **주요 과거 데이터 특징 (AI 자동 분석):**
{anomaly_details}
        """

        prompt = f"""
        당신은 대한민국 중소기업 및 소상공인 사장님들을 위한 데이터 기반 경영 컨설턴트입니다. 아래의 수요 예측 분석 결과를 바탕으로, 사장님이 즉시 실행에 옮길 수 있는 구체적이고 깊이 있는 경영 조언을 제공해주세요.

        **분석 데이터:**
        - **분석 대상:** {product_name} (제품/서비스)
        - **예측 정확도:** {accuracy:.1f}%
        - **평균 예측 오차 (MAE):** 약 {mae:.1f}개
        - **향후 7일 총 수요 예측:** {forecast_7day:.0f}개
        - **일 평균 수요 예측:** {avg_daily:.0f}개
{anomaly_prompt_section}

        **응답 형식 (아래 4가지 항목을 반드시 포함하여 마크다운 형식으로 작성):**

        ### 1. 예측 신뢰도 및 리스크 분석
        * **정확도 해석:** 예측 정확도 {accuracy:.1f}%는 우리 비즈니스에서 어떤 의미인가요? 이 예측을 얼마나 믿고 의사결정에 활용해야 할까요?
        * **오차 리스크 분석:** 평균 {mae:.1f}개의 오차는 재고 및 인력 운영에 어떤 리스크를 의미하나요? '좋은 시나리오'와 '나쁜 시나리오'를 구체적인 숫자로 제시해주세요. (만약 '과거 데이터 특징'이 있다면, 특정 이벤트가 미래에도 반복될 가능성을 언급하며 리스크를 설명해주세요.)
        * **이상 현상 분석:** 과거 데이터에서 발견된 이상 현상이나 패턴은 무엇인가요? (예: 특정 요일에 수요가 급증하거나 감소하는 경향 등) 이 정보가 예측에 어떤 영향을 미쳤나요?
        * **오차 기반 재고 관리:** 예측 오차를 고려하여 재고를 어떻게 관리해야 할까요? (예: '안전 재고를 최소 X개에서 최대 Y개 사이로 유지하고, 수요가 몰리는 날에는 추가 인력 Z명을 배치하는 것을 고려해야 합니다.' 와 같이 '~' 기호 대신 '에서', '사이' 등의 단어를 사용해주세요.)
        * **과거 데이터 특징:** 과거 데이터에서 어떤 특징이 있었나요? (예: 특정 시즌에 수요가 급증, 특정 이벤트가 수요에 큰 영향을 미침 등) 이 특징들이 현재 예측에 어떻게 반영되었나요?

        
        ### 2. 단기 실행 계획 (향후 7일)
        * **재고/자원 관리 전략:** 예측된 수요와 오차 범위를 고려한 구체적인 일일 발주 전략을 제안해주세요. (예: '핵심 재고는 최소 X개에서 최대 Y개 사이로 유지하고, 수요가 몰리는 날에는 추가 인력 Z명을 배치하는 것을 고려해야 합니다.' 와 같이 '~' 기호 대신 '에서', '사이' 등의 단어를 사용해주세요.)
        * **매출 증대 전략:** 예측된 수요에 기반한 마케팅 또는 판매 전략을 제안해주세요. ('과거 데이터 특징'에서 언급된 프로모션이나 이벤트가 성공적이었다면, 유사한 전략을 다시 제안하고 그 효과를 극대화할 방안을 제시해주세요.)

        ### 3. 중장기 전략 제언
        * **수익성 개선 방안:** 이 예측 데이터를 장기적으로 어떻게 활용하면 불필요한 비용(재고, 인건비 등)을 줄이고, 판매 기회를 극대화하여 우리 회사의 수익성을 높일 수 있을까요? ('과거 데이터 특징'의 원인들을 어떻게 활용할지 포함해주세요.)
        * **비즈니스 성장 전략:** 이 예측 데이터를 바탕으로 향후 6개월에서 1년간의 비즈니스 성장 전략을 제안해주세요. (예: '과거 데이터 특징'에서 발견된 특정 시즌이나 이벤트를 활용한 장기적인 마케팅 계획 등)

        ### 4. 추가 고려사항
        * **기타 고려사항:** 이 예측 결과를 바탕으로 사장님이 반드시 알아야 할 추가적인 고려사항이나 주의점이 있다면 무엇인가요? (예: '과거 데이터 특징'에서 발견된 특정 이벤트가 미래에도 반복될 가능성에 대한 경고 등)
        
        **주의사항:** 이 해설은 사장님이 쉽게 이해하고 바로 행동할 수 있도록 작성되어야 합니다.
        - 전문 용어는 피하고, 필요한 경우 간단한 설명을 추가해주세요.
        - 각 항목은 명확하고 간결하게 작성해주세요.
        - 예측 결과를 바탕으로 실질적인 조언을 제공해주세요.

        사장님이 쉽게 이해하고 바로 행동할 수 있도록, 친절하고 전문적인 컨설턴트의 어조로 설명해주세요.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="accounts/fireworks/models/llama4-scout-instruct-basic", # 함부로 바꾸지 말 것
                messages=[
                    {"role": "system", "content": "당신은 대한민국 중소기업 및 소상공인 사장님들을 위한 데이터 기반 경영 컨설턴트입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=4096,
                top_p=1
            )
            
            gpt_response = response.choices[0].message.content
            
            sections = gpt_response.split('###')
            parsed_explanation = {}
            for section in sections:
                if "예측 신뢰도 및 리스크 분석" in section:
                    parsed_explanation['confidence'] = "###" + section
                elif "단기 실행 계획" in section:
                    parsed_explanation['short_term_plan'] = "###" + section
                elif "중장기 전략 제언" in section:
                    parsed_explanation['long_term_plan'] = "###" + section

            full_explanation = (
                parsed_explanation.get('confidence', '') + "\n\n" +
                parsed_explanation.get('short_term_plan', '') + "\n\n" +
                parsed_explanation.get('long_term_plan', '')
            ).strip()

            summary = gpt_response.split('\n\n')[0].replace('###', '').strip()

            return {
                'summary': summary,
                'detail': full_explanation,
                'recommendation': parsed_explanation.get('short_term_plan', '')
            }
            
        except Exception as e:
            st.error(f"LLM 해설 생성 중 오류 발생: {str(e)}")
            return self._generate_fallback_explanation(
                product_name, accuracy, mae, forecast_7day, avg_daily
            )

    def _generate_fallback_explanation(self, product_name, accuracy, mae, forecast_7day, avg_daily):
        return {
            'summary': f"{product_name}의 7일 예상 수요는 총 {forecast_7day:.0f}개입니다.",
            'detail': f"예측 정확도는 {accuracy:.1f}%, 평균 예측 오차는 약 {mae:.1f}개입니다. 이 데이터를 바탕으로 재고 및 운영 전략을 수립하세요.",
            'recommendation': "재고/자원 관리: 예측된 수요와 오차를 고려하여 안전 재고 및 필요 자원을 확보하세요.\n마케팅: 수요가 높을 것으로 예상되는 날에 프로모션을 집중하세요."
        }
    
    def generate_batch_explanations(self, products_data: List[Dict]) -> Dict[str, Dict[str, str]]:
        explanations = {}
        for product in products_data:
            try:
                explanation = self.generate_explanation(
                    product_name=product['name'], model_name=product.get('model', 'Ensemble'),
                    accuracy=product['accuracy'], mae=product['mae'],
                    forecast_7day=product['forecast_7day'], avg_daily=product['avg_daily'],
                    anomaly_info=product.get('anomaly_info') # [신규] 배치 처리 시에도 anomaly_info 전달
                )
                explanations[product['name']] = explanation
            except Exception as e:
                st.warning(f"{product['name']} 해설 생성 중 오류: {str(e)}")
                continue
        return explanations

@st.cache_data(ttl=3600)
def get_product_explanation(
    product_name: str, model_name: str, accuracy: float, mae: float,
    forecast_7day: float, avg_daily: float, api_key: Optional[str] = None,
    anomaly_info: Optional[List[str]] = None # [신규] anomaly_info 인자 추가
) -> Dict[str, str]:
    try:
        explainer = GPTExplainer(api_key)
        return explainer.generate_explanation(
            product_name, model_name, accuracy, mae, 
            forecast_7day, avg_daily,
            anomaly_info=anomaly_info # [신규] explainer에 anomaly_info 전달
        )
    except Exception as e:
        st.error(f"LLM 해설 생성 실패: {str(e)}")
        return {
            'summary': f"{product_name}의 7일 예상: {forecast_7day:.0f}개",
            'detail': f"예측 정확도: {accuracy:.1f}%, 평균 오차: {mae:.1f}개",
            'recommendation': "예측 결과를 참고하여 재고 관리에 활용하세요."
        }