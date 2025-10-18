# 📊 SmartDemand - 중소기업을 위한 AI 수요예측 시스템

SmartDemand는 중소상공인을 위한 **AI 기반 수요예측 웹 서비스**입니다.  
코딩 없이도 판매 데이터를 업로드하면, 미래 수요를 예측하고 PDF 리포트와 자연어 해설까지 제공합니다.

---

## 🚀 주요 기능

| 기능 | 설명 |
|------|------|
| 🔍 **수요예측 모델** | Prophet, XGBoost, LSTM, LightGBM을 조합한 동적 앙상블 |
| 📊 **파일 기반 예측** | Excel/CSV 판매 데이터를 업로드하면 자동 분석 수행 |
| 🧠 **GPT 해설 제공** | 예측 결과를 LLM 기반 자연어 해설로 출력 |
| 📈 **오차 분석 리포트** | 예측 결과의 오차율을 자동 계산하고 리포트 PDF로 저장 |
| ⚙️ **컬럼 매핑/이상치 제거** | 일관성 없는 데이터도 자동 정제 및 처리 가능 |

---

## 🧪 실험 구조

- **데이터 생성**: GPT 기반으로 실제 편의점 품목(예: 삼각김밥, 핫바 등)의 판매 데이터를 1년치 생성
- **정답셋 정의**: 향후 7일간의 실제 수요를 임의로 설정하여 정확도 비교 가능
- **예측 성능 평가**: 단순 지수 평활법 등 전통 모델 대비 오차율 3~5% 수준의 성능 입증

---
SmartDemand 다운로드 및 설치 방법
윈도우 환경

# Git 설치: https://git-scm.com/download/win 에서 설치 후

# 명령 프롬프트 또는 PowerShell 실행
git clone https://github.com/DALB1T/smartdemand.git
cd smartdemand
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
우분투 (Linux) 환경

sudo apt update
sudo apt install -y git python3-venv python3-pip

git clone https://github.com/DALB1T/smartdemand.git
cd smartdemand
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py

## 💻 사용 방법

1. 웹사이트 접속 (Streamlit Cloud 배포 링크 예정)
2. 샘플 데이터를 업로드하거나, 본인 POS 데이터를 사용
3. 예측 결과 확인 및 PDF 다운로드
4. GPT 기반 해설로 경영 인사이트 확보

---

## 🛠 기술 스택

- **프레임워크**: Streamlit
- **모델**: Prophet, XGBoost, LSTM, LightGBM
- **해설**: Fireworks AI / OpenAI GPT
- **데이터 처리**: Pandas, Scikit-learn
- **리포트 생성**: ReportLab, Matplotlib

---

## 📄 리포트 예시

> [샘플 PDF 리포트 보기](file:///C:/Users/user/Downloads/smartdemand_report_20250625%20(2).pdf)


---

## 📬 문의

FreudEs (Project Owner)  
📧 stelle1811@freudEs.com
📜 블로그: https://freudes.tistory.com

---

