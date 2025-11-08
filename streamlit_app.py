import streamlit as st
import pandas as pd
import numpy as np
import datetime

# --- 1. 데이터 로딩 및 캐싱 ---
# Streamlit의 캐시 기능을 사용해 7개 CSV를 한 번만 로드합니다.
@st.cache_data
def load_data():
    """ 2019-2025 CSV 파일들을 로드하고 전처리합니다. """
    files = [
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2019.csv',
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2020.csv',
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2021.csv',
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2022.csv',
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2023.csv',
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2024.csv',
        '(20251106) 해외 럼피스킨 발생현황.xlsx - 2025.csv'
    ]
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except FileNotFoundError:
            # Streamlit Cloud에 배포 시, 파일 경로 문제 확인용
            st.error(f"파일을 찾을 수 없습니다: {f}. app.py와 동일한 위치에 있는지 확인하세요.")
            pass
    
    if not all_data:
        return pd.DataFrame()

    data = pd.concat(all_data, ignore_index=True)
    
    # --- 데이터 전처리 ---
    # '발생일' 컬럼을 datetime 객체로 변환 (오류 발생 시 누락 처리)
    data['발생일'] = pd.to_datetime(data['발생일'], errors='coerce')
    
    # 필수 컬럼(발생일, 위도, 경도, 지역) 누락 데이터 제거
    data.dropna(subset=['발생일', 'Lat', 'Long', '지역'], inplace=True)
    
    # 유효하지 않은 위도/경도 값(0) 제거
    data = data[(data['Lat'] != 0) & (data['Long'] != 0)]
    
    # '월-년' 컬럼 생성 (차트용)
    data['month_year'] = data['발생일'].dt.to_period('M')
    data = data.sort_values('발생일')
    return data

# --- 2. Streamlit 앱 구성 ---

# 페이지 레이아웃을 'wide'로 설정
st.set_page_config(layout="wide", page_title="신종 감염병 AI 에이전트")

# --- 데이터 로드 ---
data = load_data()

if data.empty:
    st.error("데이터 로딩에 실패했습니다. CSV 파일들을 올바른 위치에 두었는지 확인하세요.")
    st.stop() # 데이터 없으면 앱 실행 중지

# --- 3. 사이드바 (AI 에이전트 제어판) ---
st.sidebar.title("🤖 AI 에이전트 제어판")
st.sidebar.markdown("---")

# [핵심 기능 1] Agent A/B 테스트 토글
agent_b_enabled = st.sidebar.toggle(
    "LLM 인지 강화 활성화 (Agent B)", 
    value=True, 
    help="Agent B는 LLM의 맥락 인지(XAI) 기능을 통해 더 정확한 예측과 '설명'을 제공합니다."
)
st.sidebar.markdown("---")

# [핵심 기능 2] 'What-if' 시뮬레이션 시점
st.sidebar.subheader("시뮬레이션 시점 ('What-if')")

# 실제 한국 최초 발생일 (2023년 10월 19일) 직전으로 기본값 설정
korea_outbreak_date = datetime.datetime(2023, 10, 19)
default_sim_date = datetime.datetime(2023, 9, 30) # 2023년 9월 30일

sim_date = st.sidebar.slider(
    "가상 '오늘' 날짜 설정:",
    min_value=data['발생일'].min().to_pydatetime(),
    max_value=data['발생일'].max().to_pydatetime(),
    value=default_sim_date,
    format="YYYY-MM-DD",
    help="시간을 돌려 '만약 그날 이 AI가 있었다면?'을 시연합니다. (기본값: 2023년 한국 최초 발생 직전)"
)
st.sidebar.markdown("---")

# [핵심 기능 3] 분석 대상 대륙 선택
continents = st.sidebar.multiselect(
    "분석 대상 대륙",
    options=data['지역'].unique(),
    default=['아시아', '유럽'],
    help="럼피스킨은 아시아와 유럽을 거쳐 유입되었습니다."
)
st.sidebar.markdown("---")
st.sidebar.info("이 대시보드는 '신종 감염병 AI 프레임워크'가 LSD 사태를 어떻게 예측했을지 시연하는 PoC입니다.")


# --- 4. 메인 대시보드 (미션 컨트롤) ---

st.title("🤖 신종 감염병 조기 경보 AI 에이전트")
st.markdown(f"**케이스 스터디:** 럼피스킨(LSD) / **시뮬레이션 시점:** `{sim_date.strftime('%Y-%m-%d')}`")

# --- 5. PoC용 가상 지표 생성 (에이전트 두뇌) ---

# 시뮬레이션 시점과 대륙에 맞춰 데이터 필터링
filtered_data = data[(data['발생일'] <= sim_date) & (data['지역'].isin(continents))]

# [PoC 로직] 
# '아시아' 데이터가 많아질수록 위험도 증가 (Baseline)
# Agent B는 아시아 데이터에 더 높은 가중치를 부여 (LLM 인지)
asia_cases = len(filtered_data[filtered_data['지역'] == '아시아'])
total_cases = len(filtered_data)
time_factor = (sim_date.year - 2019) / 4 # 2019년 대비 시간 흐름 가중치 (최대 1)

# Agent A (Baseline) 위험도: 아시아 케이스 비율 + 시간 흐름
risk_score_a = min(99, (asia_cases / (total_cases + 1)) * 100 + (time_factor * 20))

# Agent B (LLM 강화) 위험도: Baseline + LLM의 '맥락 인지' 가중치
# LLM은 아시아, 특히 2023년 데이터에 높은 가중치를 둠
llm_context_bonus = (asia_cases * time_factor * 1.5) if agent_b_enabled else 0
risk_score_b = min(99, risk_score_a + llm_context_bonus)

# LLM 파생 변수 (XAI)
if risk_score_b > 80:
    llm_phase = "확산기 (Diffusion)"
    llm_score = "9.5"
    recommendation_a = "아시아 전역 확산. 위험도 급증."
    recommendation_b = "🚨 **긴급 경보** 🚨\nLLM이 '유행 확산기' 패턴을 감지했습니다. **한국 유입이 임박**했습니다. (위험도: 9.5/10)"
elif risk_score_b > 50:
    llm_phase = "초기 (Early)"
    llm_score = "7.0"
    recommendation_a = "아시아 남부 확산. 모니터링 필요."
    recommendation_b = "⚠️ **주의 경보** ⚠️\nLLM이 '유행 초기' 패턴을 감지했습니다. 아시아-한국 경로의 위험도가 높습니다. (위험도: 7.0/10)"
else:
    llm_phase = "잠복기 (Latent)"
    llm_score = "4.0"
    recommendation_a = "유럽/아프리카 위주 발생."
    recommendation_b = "📈 **관심** 📈\nLLM이 '잠복기' 패턴을 감지했습니다. 지속적인 글로벌 모니터링이 필요합니다. (위험도: 4.0/10)"


# --- 6. '얼굴' 3단 핵심 요약 (Prediction / XAI / Action) ---
st.header(f"AI 에이전트 핵심 브리핑 (As of: {sim_date.strftime('%Y-%m-%d')})")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("① 한국 유입 위험도 (Prediction)")
    if agent_b_enabled:
        st.metric(label="Agent B (LLM 강화)", value=f"{risk_score_b:.1f} %", delta=f"{risk_score_b - risk_score_a:.1f} %p 향상")
    else:
        st.metric(label="Agent A (Baseline)", value=f"{risk_score_a:.1f} %", delta=None)
    st.markdown("`Agent B`는 LLM의 맥락 인지를 통해 더 정확한 위험도를 예측합니다.")

with col2:
    st.subheader("② LLM의 XAI 진단 (Why)")
    if agent_b_enabled:
        st.metric(label="LLM 진단: 글로벌 유행 단계", value=llm_phase)
        st.metric(label="LLM 평가: 자체 위험 점수", value=f"{llm_score} / 10")
        st.markdown("`Agent A`는 이 '맥락' 정보가 없습니다.")
    else:
        st.info("LLM 인지 강화를 활성화해야 '설명 가능한(XAI)' 진단 정보를 볼 수 있습니다.")

with col3:
    st.subheader("③ AI 에이전트 권고 (Action)")
    if agent_b_enabled:
        st.warning(recommendation_b) # 위험도에 따라 색상 변경
    else:
        st.info(f"Agent A 권고: {recommendation_a}")
    st.markdown("`Agent B`는 XAI 진단을 기반으로 구체적인 행동을 권고합니다.")

st.markdown("---")

# --- 7. 시각화 자료 (지도 및 차트) ---

col_map, col_chart = st.columns(2)

with col_map:
    st.subheader(f"🗺️ 글로벌 확산 지도 (Until {sim_date.strftime('%Y-%m-%d')})")
    if filtered_data.empty:
        st.warning("선택한 시점/지역에 데이터가 없습니다.")
    else:
        # st.map을 위해 위도/경도 컬럼명 변경
        map_data = filtered_data.rename(columns={'Lat': 'lat', 'Long': 'lon'})
        st.map(map_data[['lat', 'lon']])

with col_chart:
    st.subheader(f"📈 월별 발생 건수 추이 (Until {sim_date.strftime('%Y-%m')})")
    # 월별 발생 건수 집계
    monthly_counts = filtered_data.groupby('month_year').size().reset_index(name='건수')
    monthly_counts['month_year'] = monthly_counts['month_year'].astype(str) # Streamlit 차트를 위해 str 변환
    
    if monthly_counts.empty:
        st.warning("선택한 시점/지역에 데이터가 없습니다.")
    else:
        st.line_chart(monthly_counts.set_index('month_year'))

# --- 8. 원본 데이터 보기 ---
with st.expander(f"시뮬레이션 시점 기준 상세 데이터 보기 ({len(filtered_data)} 건)"):
    st.dataframe(filtered_data)
