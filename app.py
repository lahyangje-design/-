import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# 페이지 기본 설정
st.set_page_config(
    page_title="K-FedWatch: 금통위 기준금리 예측 대시보드",
    page_icon="🏛️",
    layout="wide"
)

# ----------------- 1. 모델 번들 로드 -----------------
@st.cache_resource
def load_bundle():
    return joblib.load("kfed_model_bundle.pkl")

try:
    bundle = load_bundle()
    rf_model = bundle["rf_model"]
    scaler = bundle["scaler"]
    tone_model = bundle["tone_model"]
    tfidf = bundle["tfidf"]
    final_features = bundle["final_features"]
    classes = bundle["classes"]
    default_base_rate = bundle["current_base_rate"]
except Exception as e:
    st.error(f"모델 파일을 불러오지 못했습니다: {e}")
    st.stop()

# ----------------- 2. 가우시안 시장 내재 확률 역산 함수 -----------------
def calc_dynamic_market_probs(net_shift_bp):
    sigma = 12.0
    score_cut  = np.exp(-((net_shift_bp - (-25.0)) ** 2) / (2 * (sigma ** 2)))
    score_hold = np.exp(-((net_shift_bp - 0.0) ** 2) / (2 * (sigma ** 2)))
    score_hike = np.exp(-((net_shift_bp - 25.0) ** 2) / (2 * (sigma ** 2)))
    total = score_cut + score_hold + score_hike
    return (
        round(score_cut / total * 100, 1),
        round(score_hold / total * 100, 1),
        round(score_hike / total * 100, 1)
    )

# ----------------- 3. 대시보드 헤더 -----------------
st.title("🏛️ K-FedWatch: 차기 금통위 기준금리 확률 추정 시스템")
st.caption("CME FedWatch 벤치마크 하이브리드 엔진 (채권 시장 내재 확률 60% + 거시/NLP AI 앙상블 40%)")
st.markdown("---")

# ----------------- 4. 사이드바: 실시간 시장 및 거시 변수 입력 -----------------
st.sidebar.header("⚙️ 시나리오 변수 시뮬레이터")

st.sidebar.subheader("1. 채권 및 시장 금리")
base_rate = st.sidebar.number_input("현재 한국은행 기준금리 (%)", value=default_base_rate, step=0.25)
msb_91d = st.sidebar.number_input("통안채 91일물 금리 (%)", value=3.500, step=0.005, format="%.3f")
ktb_3y = st.sidebar.number_input("국고채 3년물 금리 (%)", value=2.932, step=0.005, format="%.3f")
call_rate = st.sidebar.number_input("콜금리 (%)", value=3.000, step=0.01, format="%.3f")
net_shift_bp = st.sidebar.slider("통안채 60일 이동평균 대비 이탈도 (bp)", min_value=-50.0, max_value=50.0, value=16.7, step=0.1)

st.sidebar.subheader("2. 거시경제 충격 지표")
cpi_yoy = st.sidebar.slider("소비자물가상승률 CPI (YoY %)", min_value=0.0, max_value=6.0, value=3.09, step=0.01)
oil_change = st.sidebar.slider("최근 30일 국제유가(WTI) 변동률 (%)", min_value=-30.0, max_value=50.0, value=19.34, step=0.1)
fx_change = st.sidebar.slider("최근 30일 원/달러 환율 변동률 (%)", min_value=-20.0, max_value=20.0, value=-3.54, step=0.1)

st.sidebar.subheader("3. 금통위 의사록 어조 (NLP)")
tone_score = st.sidebar.slider("의사록 Tone Score (음수: 비둘기파 / 양수: 매파)", min_value=-0.5, max_value=0.5, value=-0.1701, step=0.001)

# ----------------- 5. 확률 계산 및 블렌딩 연산 -----------------
# 1) 시장 내재 확률 도출
mkt_p_cut, mkt_p_hold, mkt_p_hike = calc_dynamic_market_probs(net_shift_bp)

# 2) 파생 스프레드 계산
msb_spread_bp = (msb_91d - base_rate) * 100
curve_slope_bp = (ktb_3y - msb_91d) * 100
call_spread_bp = (call_rate - base_rate) * 100

# 3) AI 모델 예측
input_dict = {
    "tone_score": tone_score,
    "msb_spread_bp": msb_spread_bp,
    "net_shift_bp": net_shift_bp,
    "curve_slope_bp": curve_slope_bp,
    "call_spread_bp": call_spread_bp,
    "시장_인하확률(%)": mkt_p_cut,
    "시장_인상확률(%)": mkt_p_hike,
    "cpi_yoy": cpi_yoy,
    "oil_change_pct": oil_change,
    "fx_change_pct": fx_change
}

df_input = pd.DataFrame([input_dict])[final_features]
X_scaled = scaler.transform(df_input)
ai_probs = rf_model.predict_proba(X_scaled)[0]

cut_i  = classes.index(-1) if -1 in classes else None
hold_i = classes.index(0)  if 0  in classes else None
hike_i = classes.index(1)  if 1  in classes else None

ai_p_cut  = ai_probs[cut_i] * 100.0 if cut_i is not None else 0.0
ai_p_hold = ai_probs[hold_i] * 100.0 if hold_i is not None else 0.0
ai_p_hike = ai_probs[hike_i] * 100.0 if hike_i is not None else 0.0

# 4) 시장 60% : AI 40% 블렌딩
w_mkt, w_ai = 0.60, 0.40
final_cut  = w_mkt * mkt_p_cut  + w_ai * ai_p_cut
final_hold = w_mkt * mkt_p_hold + w_ai * ai_p_hold
final_hike = w_mkt * mkt_p_hike + w_ai * ai_p_hike

tot = final_cut + final_hold + final_hike
final_cut  = round(final_cut / tot * 100, 1)
final_hold = round(final_hold / tot * 100, 1)
final_hike = round(100.0 - final_cut - final_hold, 1)

# 메인 판정 도출
decision_map = {
    final_cut: ("인하 (-25bp)", "🔵", "#1e88e5"),
    final_hold: ("동결 (0bp)", "⚪", "#757575"),
    final_hike: ("인상 (+25bp)", "🔴", "#e53935")
}
max_prob = max(final_cut, final_hold, final_hike)
target_decision, status_icon, target_color = decision_map[max_prob]

# ----------------- 6. 상단 요약 메트릭 -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("현행 기준금리", f"{base_rate:.2f}%")
col2.metric("🔵 인하 (-25bp) 확률", f"{final_cut:.1f}%")
col3.metric("⚪ 동결 (0bp) 확률", f"{final_hold:.1f}%")
col4.metric("🔴 인상 (+25bp) 확률", f"{final_hike:.1f}%")

st.markdown(f"""
<div style="padding:15px; border-radius:10px; background-color:#f0f4f8; border-left:6px solid {target_color}; margin: 15px 0;">
    <h3 style="margin:0; color:#1a1a1a;">최종 예측 결과: {status_icon} <b>{target_decision}</b> (유력 확률: {max_prob:.1f}%)</h3>
</div>
""", unsafe_allow_html=True)

# ----------------- 7. 시각화 (도넛 게이지 & 기여도 바 차트) -----------------
tab1, tab2 = st.tabs(["📊 CME 스타일 확률 분포", "🔍 시장 vs AI 블렌딩 상세"])

with tab1:
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        fig_donut = go.Figure(data=[go.Pie(
            labels=["인하 (-25bp)", "동결 (0bp)", "인상 (+25bp)"],
            values=[final_cut, final_hold, final_hike],
            hole=0.55,
            marker_colors=["#5bc0de", "#d6d8db", "#d9534f"],
            textinfo="label+percent",
            insidetextorientation="radial"
        )])
        fig_donut.update_layout(
            title="차기 금통위 기준금리 결정 내재 확률",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_table:
        st.subheader("지표 진단 현황")
        st.table(pd.DataFrame({
            "핵심 지표": ["장단기 커브 (국고3년 - 통안91일)", "통안채 스프레드", "소비자물가(CPI)", "국제유가 변동률", "원/달러 환율 변동률"],
            "수치": [f"{curve_slope_bp:+.1f} bp", f"{msb_spread_bp:+.1f} bp", f"{cpi_yoy:.2f}%", f"{oil_change:+.2f}%", f"{fx_change:+.2f}%"]
        }))

with tab2:
    st.subheader("시장 호가(60%) vs AI 모델(40%) 기여도 분해")
    categories = ["인하 (-25bp)", "동결 (0bp)", "인상 (+25bp)"]
    
    fig_bar = go.Figure(data=[
        go.Bar(name="채권시장 내재 (60% 가중)", x=categories, y=[mkt_p_cut, mkt_p_hold, mkt_p_hike], marker_color="#f0ad4e"),
        go.Bar(name="AI 앙상블 (40% 가중)", x=categories, y=[ai_p_cut, ai_p_hold, ai_p_hike], marker_color="#337ab7"),
        go.Bar(name="최종 종합 확률", x=categories, y=[final_cut, final_hold, final_hike], marker_color="#5cb85c")
    ])
    fig_bar.update_layout(
        barmode="group",
        template="plotly_white",
        yaxis_title="확률 (%)",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_bar, use_container_width=True)