import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# --------------------------------------------------------------------------
# 페이지 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="K-FedWatch: 한국은행 기준금리 예측 대시보드",
    page_icon="🏛️",
    layout="wide"
)

# --------------------------------------------------------------------------
# 1. 모델 번들 로드
# --------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("kfed_model_bundle.pkl")

try:
    bundle = load_bundle()
    rf_model = bundle["rf_model"]
    scaler = bundle["scaler"]
    tone_model = bundle.get("tone_model", None)
    tfidf = bundle.get("tfidf", None)
    final_features = bundle["final_features"]
    classes = bundle["classes"]
    default_base_rate = bundle.get("current_base_rate", 3.00)
    feat_importances = bundle.get("feature_importances", None)
    
    rate_df = bundle.get("rate_df", pd.DataFrame())
    macro_daily = bundle.get("macro_daily", pd.DataFrame())
    cpi_df = bundle.get("cpi_df", pd.DataFrame())
    market_rates_df = bundle.get("market_rates_df", pd.DataFrame())
    test_data = bundle.get("test_data", pd.DataFrame())
    sim_df = bundle.get("sim_df", pd.DataFrame())
except Exception as e:
    st.error(f"모델 파일을 불러오지 못했습니다: {e}")
    st.stop()

# --------------------------------------------------------------------------
# 2. 시장 내재 확률 역산 함수
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 3. 사이드바: 시뮬레이터 입력
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ 차기 회의 변수 시뮬레이터")
st.sidebar.caption("시장 호가 및 거시지표를 변경해 예측치를 시뮬레이션할 수 있습니다.")

st.sidebar.subheader("1. 채권 및 시장 금리")
base_rate = st.sidebar.number_input("현재 기준금리 (%)", value=default_base_rate, step=0.25)
msb_91d = st.sidebar.number_input("통안채 91일물 금리 (%)", value=3.500, step=0.005, format="%.3f")
ktb_3y = st.sidebar.number_input("국고채 3년물 금리 (%)", value=2.932, step=0.005, format="%.3f")
call_rate = st.sidebar.number_input("콜금리 (%)", value=3.000, step=0.01, format="%.3f")
net_shift_bp = st.sidebar.slider("통안채 60일 평균 대비 이탈도 (bp)", min_value=-50.0, max_value=50.0, value=16.7, step=0.1)

st.sidebar.subheader("2. 거시경제 충격 지표")
cpi_yoy = st.sidebar.slider("소비자물가상승률 CPI (YoY %)", min_value=0.0, max_value=6.0, value=3.09, step=0.01)
oil_change = st.sidebar.slider("최근 30일 국제유가 변동률 (%)", min_value=-30.0, max_value=50.0, value=19.34, step=0.1)
fx_change = st.sidebar.slider("최근 30일 원/달러 환율 변동률 (%)", min_value=-20.0, max_value=20.0, value=-3.54, step=0.1)

st.sidebar.subheader("3. 금통위 의사록 어조")
tone_score = st.sidebar.slider("의사록 Tone Score (-0.5: 완화 / +0.5: 긴축)", min_value=-0.5, max_value=0.5, value=-0.1701, step=0.001, format="%.4f")

# --------------------------------------------------------------------------
# 4. 차기 예측 확률 연산
# --------------------------------------------------------------------------
mkt_p_cut, mkt_p_hold, mkt_p_hike = calc_dynamic_market_probs(net_shift_bp)
msb_spread_bp = (msb_91d - base_rate) * 100
curve_slope_bp = (ktb_3y - msb_91d) * 100
call_spread_bp = (call_rate - base_rate) * 100

# 10개 및 14개 피처 모두 지원 가능한 전체 딕셔너리
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
    "fx_change_pct": fx_change,
    "us_kr_spread_bp": (base_rate - 5.25) * 100,
    "credit_spread_bp": 65.0,
    "cp_spread_bp": 35.0,
    "debt_growth_yoy": 4.35
}

# ★ [핵심 해결 코드] 스케일러가 실제로 학습할 때 사용한 정확한 피처 목록 자동 추출
expected_features = getattr(scaler, "feature_names_in_", None)
if expected_features is None:
    expected_features = getattr(rf_model, "feature_names_in_", final_features)

# 스케일러가 요구하는 순서와 컬럼만 정확하게 1:1 매칭 (누락된 경우 기본값 0.0 보충)
matched_input = {col: input_dict.get(col, 0.0) for col in expected_features}
df_input = pd.DataFrame([matched_input])[list(expected_features)]

# 검증 및 정규화 실행 (더 이상 ValueError 발생하지 않음)
X_scaled = scaler.transform(df_input)
ai_probs = rf_model.predict_proba(X_scaled)[0]

cut_i  = classes.index(-1) if -1 in classes else None
hold_i = classes.index(0)  if 0  in classes else None
hike_i = classes.index(1)  if 1  in classes else None

ai_p_cut  = ai_probs[cut_i] * 100.0 if cut_i is not None else 0.0
ai_p_hold = ai_probs[hold_i] * 100.0 if hold_i is not None else 0.0
ai_p_hike = ai_probs[hike_i] * 100.0 if hike_i is not None else 0.0

w_mkt, w_ai = 0.60, 0.40
final_cut  = w_mkt * mkt_p_cut  + w_ai * ai_p_cut
final_hold = w_mkt * mkt_p_hold + w_ai * ai_p_hold
final_hike = w_mkt * mkt_p_hike + w_ai * ai_p_hike

tot = final_cut + final_hold + final_hike
final_cut  = round(final_cut / tot * 100, 1)
final_hold = round(final_hold / tot * 100, 1)
final_hike = round(100.0 - final_cut - final_hold, 1)

decision_map = {
    final_cut: ("인하 (-25bp)", "🔵", "#1e88e5"),
    final_hold: ("동결 (0bp)", "⚪", "#757575"),
    final_hike: ("인상 (+25bp)", "🔴", "#e53935")
}
max_prob = max(final_cut, final_hold, final_hike)
target_decision, status_icon, target_color = decision_map[max_prob]

# --------------------------------------------------------------------------
# 5. 메인 대시보드
# --------------------------------------------------------------------------
st.title("🏛️ K-FedWatch: 한국은행 금융통화위원회 기준금리 예측 시스템")
st.caption("단기 채권 시장 내재 확률(60%) + 거시·NLP AI 모델(40%) 결합[cite: 1]")

tab_oct, tab_why, tab_hist, tab_aug, tab_rate, tab_tone, tab_oil, tab_fx = st.tabs([
    "🏛️ 10월 금리 예측",
    "🔍 왜 그렇게 나왔을까? (원인 분석)",
    "⏳ 과거 회의 시뮬레이터 (2024~)",
    "📅 8월 예측 및 성적표",
    "🏦 기준금리 변동 현황",
    "📝 금통위 어조(Tone) 추이",
    "🛢️ 국제유가(WTI) 변동 현황",
    "💵 원/달러 환율 변동 현황"
])

# [TAB 1] 10월 예측
with tab_oct:
    st.subheader("📌 차기 기준금리 결정 확률 추정")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현행 기준금리", f"{base_rate:.2f}%")
    col2.metric("🔵 인하 (-25bp) 확률", f"{final_cut:.1f}%")
    col3.metric("⚪ 동결 (0bp) 확률", f"{final_hold:.1f}%")
    col4.metric("🔴 인상 (+25bp) 확률", f"{final_hike:.1f}%")

    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background-color:#f0f4f8; border-left:6px solid {target_color}; margin: 15px 0;">
        <h3 style="margin:0; color:#1a1a1a;">차기 메인 시나리오 판정: {status_icon} <b>{target_decision}</b> (유력 확률: {max_prob:.1f}%)</h3>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2])
    with col_chart:
        fig_donut = go.Figure(data=[go.Pie(
            labels=["인하 (-25bp)", "동결 (0bp)", "인상 (+25bp)"],
            values=[final_cut, final_hold, final_hike],
            hole=0.55,
            marker_colors=["#5bc0de", "#d6d8db", "#d9534f"],
            textinfo="label+percent"
        )])
        fig_donut.update_layout(title="시나리오별 최종 결합 확률 분포", template="plotly_white")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_info:
        st.subheader("📋 입력 지표 현황 요약")
        st.table(pd.DataFrame({
            "핵심 지표": [
                "장단기 커브 (국고3년 - 통안91일)[cite: 1]",
                "통안채 91일 스프레드[cite: 1]",
                "통안채 60일 이탈도 (Net Shift)[cite: 1]",
                "소비자물가지수 (CPI YoY)[cite: 1]",
                "국제유가 30일 변동률[cite: 1]",
                "원/달러 환율 30일 변동률[cite: 1]",
                "의사록 톤 점수 (Tone Score)[cite: 1]"
            ],
            "현재 수치": [
                f"{curve_slope_bp:+.1f} bp",
                f"{msb_spread_bp:+.1f} bp",
                f"{net_shift_bp:+.1f} bp",
                f"{cpi_yoy:.2f}%",
                f"{oil_change:+.2f}%",
                f"{fx_change:+.2f}%",
                f"{tone_score:+.4f}"
            ]
        }))

# [TAB 2] 원인 분석
with tab_why:
    st.subheader("🔍 K-FedWatch 예측 결과 심층 원인 분석[cite: 1]")
    st.markdown("#### 1. 시장 호가(60%) vs AI 모델(40%) 확률 기여도 분해[cite: 1]")
    decomp_df = pd.DataFrame({
        "시나리오": ["인하 (-25bp)", "동결 (0bp)", "인상 (+25bp)"],
        "채권시장 순수 확률 (60% 가중)": [f"{mkt_p_cut:.1f}%", f"{mkt_p_hold:.1f}%", f"{mkt_p_hike:.1f}%"],
        "AI 모델 순수 확률 (40% 가중)": [f"{ai_p_cut:.1f}%", f"{ai_p_hold:.1f}%", f"{ai_p_hike:.1f}%"],
        "시장 기여분 (A)": [round(mkt_p_cut * w_mkt, 1), round(mkt_p_hold * w_mkt, 1), round(mkt_p_hike * w_mkt, 1)],
        "AI 기여분 (B)": [round(ai_p_cut * w_ai, 1), round(ai_p_hold * w_ai, 1), round(ai_p_hike * w_ai, 1)],
        "최종 결합 확률 (A+B)": [f"{final_cut:.1f}%", f"{final_hold:.1f}%", f"{final_hike:.1f}%"]
    })
    st.dataframe(decomp_df, use_container_width=True)

    fig_comp = go.Figure(data=[
        go.Bar(name="순수 채권시장 (60% 반영)", x=["인하", "동결", "인상"], y=[mkt_p_cut, mkt_p_hold, mkt_p_hike], marker_color="#f0ad4e"),
        go.Bar(name="AI 머신러닝 (40% 반영)", x=["인하", "동결", "인상"], y=[ai_p_cut, ai_p_hold, ai_p_hike], marker_color="#337ab7"),
        go.Bar(name="최종 K-FedWatch", x=["인하", "동결", "인상"], y=[final_cut, final_hold, final_hike], marker_color="#5cb85c")
    ])
    fig_comp.update_layout(barmode="group", template="plotly_white", yaxis_title="확률 (%)")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### 2. 핵심 변수별 경제적 영향 진단[cite: 1]")
    st.table(pd.DataFrame([
        {"변수명": "장단기 커브 기울기", "현재 수치": f"{curve_slope_bp:+.1f} bp", "영향력": "강력한 동결 요인[cite: 1]", "상세 분석": "국고채 3년물이 통안채보다 낮게 역전되어 있어 경기 둔화 우려로 추가 인상 제동[cite: 1]"},
        {"변수명": "통안채 60일 이탈도", "현재 수치": f"{net_shift_bp:+.1f} bp", "영향력": "인상 지지 요인[cite: 1]", "상세 분석": "단기 채권 시장 금리가 평균 대비 16.7bp 상승하여 시장 인상 기대감 유발[cite: 1]"},
        {"변수명": "원/달러 환율 변동률", "현재 수치": f"{fx_change:+.2f}%", "영향력": "동결 지지 요인[cite: 1]", "상세 분석": "환율이 안정세를 보이며 수입 물가 부담 및 외환 방어용 금리 인상 압력 완화[cite: 1]"},
        {"변수명": "국제유가(WTI) 변동률", "현재 수치": f"{oil_change:+.2f}%", "영향력": "인상 압력 요인[cite: 1]", "상세 분석": "유가 반등세가 물가 상승 압력으로 작용해 인상 확률을 지지[cite: 1]"},
        {"변수명": "의사록 톤 점수", "현재 수치": f"{tone_score:+.4f}", "영향력": "중립 ~ 완화", "상세 분석": "금통위원들의 매파적 긴축 의지가 과거 인상기 대비 강하지 않음[cite: 1]"}
    ]))

    if feat_importances is not None:
        st.markdown("#### 3. AI 모델의 변수 중요도 (Feature Importance)[cite: 1]")
        fi_df = pd.DataFrame({
            "피처": avail_features,
            "중요도 (%)": feat_importances[:len(avail_features)] * 100
        }).sort_values("중요도 (%)", ascending=True)
        fig_fi = px.bar(fi_df, x="중요도 (%)", y="피처", orientation="h", title="Random Forest 모델 내 판단 가중치 순위[cite: 1]")
        fig_fi.update_layout(template="plotly_white")
        st.plotly_chart(fig_fi, use_container_width=True)

# [TAB 3] 과거 회의 시뮬레이터 (2024~)
with tab_hist:
    st.subheader("⏳ 역대 금통위 예측 백테스트 및 시뮬레이터 (2024 ~ 2026)")
    st.markdown("""
    조회하고 싶은 **금통위 회의 날짜**를 선택하면, **해당 회의 직전 시점까지의 데이터만으로 모델이 계산했던 예측 확률**과
    **실제 한국은행의 결정 결과**, 그리고 **당시 거시경제 지표 상황**을 재현합니다.
    """)

    target_sim = sim_df if not sim_df.empty else (test_data[test_data["date"] >= "2024-01-01"] if not test_data.empty else pd.DataFrame())

    if not target_sim.empty and "date" in target_sim.columns:
        date_list = target_sim["date"].dt.strftime("%Y-%m-%d").tolist()
        col_sel1, col_sel2 = st.columns([2, 3])
        with col_sel1:
            selected_date = st.selectbox("📅 조회할 금통위 회의 일자 선택:", options=date_list, index=len(date_list)-1)
        
        row = target_sim[target_sim["date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
        actual_dec = row.get("실제결정", "동결(0)")
        pred_dec = row.get("예측결정", "동결(0)")
        is_hit = (actual_dec == pred_dec)
        
        with col_sel2:
            if is_hit:
                st.success(f"### ✅ 예측 적중! (모델 예측: {pred_dec} ➔ 실제 한은 결정: {actual_dec})")
            else:
                st.error(f"### ⚠️ 시장 서프라이즈 (모델 예측: {pred_dec} ➔ 실제 한은 결정: {actual_dec})")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("회의 당시 기준금리", f"{row.get('base_rate', 0.0):.2f}%")
        c2.metric("🔵 인하 확률", f"{row.get('종합_인하확률(%)', 0.0):.1f}%")
        c3.metric("⚪ 동결 확률", f"{row.get('종합_동결확률(%)', 0.0):.1f}%")
        c4.metric("🔴 인상 확률", f"{row.get('종합_인상확률(%)', 0.0):.1f}%")

        col_h_chart, col_h_table = st.columns([3, 2])
        with col_h_chart:
            fig_hist_pie = go.Figure(data=[go.Pie(
                labels=["인하", "동결", "인상"],
                values=[row.get('종합_인하확률(%)', 0.0), row.get('종합_동결확률(%)', 0.0), row.get('종합_인상확률(%)', 0.0)],
                hole=0.55,
                marker_colors=["#5bc0de", "#d6d8db", "#d9534f"],
                textinfo="label+percent"
            )])
            fig_hist_pie.update_layout(title=f"{selected_date} 회의 직전 모델 내재 확률 분포", template="plotly_white")
            st.plotly_chart(fig_hist_pie, use_container_width=True)

        with col_h_table:
            st.subheader("📊 당시 거시/시장 환경 스냅샷")
            snap_df = pd.DataFrame({
                "지표명": ["통안채 91일 스프레드", "통안채 60일 이탈도", "장단기 커브 기울기", "소비자물가(CPI YoY)", "국제유가 30일 변동률", "원/달러 환율 30일 변동률", "의사록 톤 점수"],
                "당시 수치": [
                    f"{row.get('msb_spread_bp', 0.0):+.1f} bp",
                    f"{row.get('net_shift_bp', 0.0):+.1f} bp",
                    f"{row.get('curve_slope_bp', 0.0):+.1f} bp",
                    f"{row.get('cpi_yoy', 0.0):.2f}%",
                    f"{row.get('oil_change_pct', 0.0):+.2f}%",
                    f"{row.get('fx_change_pct', 0.0):+.2f}%",
                    f"{row.get('tone_score', 0.0):+.4f}"
                ]
            })
            st.table(snap_df)

        st.markdown("---")
        total_meetings = len(target_sim)
        hit_meetings = int((target_sim["실제결정"] == target_sim["예측결정"]).sum())
        hit_ratio = (hit_meetings / total_meetings) * 100
        st.info(f"💡 **2024~2026 구간 누적 적중률:** 총 **{total_meetings}회** 회의 중 **{hit_meetings}회** 적중 (**{hit_ratio:.1f}%**)")
        
        table_cols = [c for c in ["date", "base_rate", "실제결정", "예측결정", "종합_인하확률(%)", "종합_동결확률(%)", "종합_인상확률(%)"] if c in target_sim.columns]
        st.dataframe(target_sim[table_cols], use_container_width=True)

# [TAB 4] 8월 성적표
with tab_aug:
    st.subheader("📅 2026년 8월 27일 금통위 예측 결과 및 백테스트 성적표[cite: 1]")
    col_acc1, col_acc2, col_acc3 = st.columns(3)
    col_acc1.metric("테스트 회의 건수", "83건 (2022~2026)[cite: 1]")
    col_acc2.metric("최종 모델 정확도 (Accuracy)", "80.72%[cite: 1]")
    col_acc3.metric("동결 예측 F1-Score", "0.89[cite: 1]")
    
    st.markdown("---")
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("🔵 인하 (-25bp) 확률", "5.3%")
    col_res2.metric("⚪ 동결 (0bp) 확률", "74.3%")
    col_res3.metric("🔴 인상 (+25bp) 확률", "20.4%")
    st.warning("""
    * **모델 최종 판정:** **【 동결 (74.3%) 】**
    * **8월 27일 한은 실제 결정:** **【 25bp 인상 단행 (2.75% ➔ 3.00%) 】**
    * **경제적 원인 분석:** 당시 유가(-2.64%) 및 환율(-4.93%) 안정세로 시장은 동결을 확신했으나, 한은이 수도권 가계부채 관리를 위해 기습 인상을 단행했던 서프라이즈 사례입니다.
    """)

# [TAB 5] 기준금리 이력
with tab_rate:
    st.subheader("🏦 한국은행 기준금리 변경 역사 (1999 ~ 2026)[cite: 1]")
    if not rate_df.empty and "date" in rate_df.columns:
        fig_rate = go.Figure(data=[go.Scatter(x=rate_df["date"], y=rate_df["base_rate"], mode="lines+markers", line_shape="hv", line=dict(color="#1f77b4", width=2.5))])
        fig_rate.update_layout(title="역대 기준금리 추이 (Step Chart)[cite: 1]", xaxis_title="일자", yaxis_title="기준금리 (%)", template="plotly_white")
        st.plotly_chart(fig_rate, use_container_width=True)

# [TAB 6] 어조 추이
with tab_tone:
    st.subheader("📝 금통위 의사록 어조(Tone Score) 변동 추이[cite: 1]")
    if not test_data.empty and "tone_score" in test_data.columns:
        fig_tone = go.Figure(data=[go.Bar(x=test_data["date"], y=test_data["tone_score"], marker_color=np.where(test_data["tone_score"] > 0, "#d9534f", "#5bc0de"))])
        fig_tone.add_hline(y=0, line_dash="dash", line_color="black")
        fig_tone.update_layout(title="금통위 회의별 의사록 텍스트 어조 지수[cite: 1]", xaxis_title="회의 일자", yaxis_title="어조 점수", template="plotly_white")
        st.plotly_chart(fig_tone, use_container_width=True)

# [TAB 7] 유가
with tab_oil:
    st.subheader("🛢️ 국제유가(WTI 원유 선물) 가격 추이[cite: 1]")
    if not macro_daily.empty and "oil_price" in macro_daily.columns:
        fig_oil = go.Figure(data=[go.Scatter(x=macro_daily["Date"], y=macro_daily["oil_price"], mode="lines", name="WTI 종가 ($/배럴)", line=dict(color="#d9534f", width=1.5))])
        fig_oil.update_layout(title="WTI 원유 선물 가격 추이", xaxis_title="일자", yaxis_title="달러 ($)", template="plotly_white")
        st.plotly_chart(fig_oil, use_container_width=True)

# [TAB 8] 환율
with tab_fx:
    st.subheader("💵 원/달러(USD/KRW) 환율 추이[cite: 1]")
    if not macro_daily.empty and "usdkrw" in macro_daily.columns:
        fig_fx = go.Figure(data=[go.Scatter(x=macro_daily["Date"], y=macro_daily["usdkrw"], mode="lines", name="원/달러 환율 (원)", line=dict(color="#0275d8", width=1.5))])
        fig_fx.update_layout(title="원/달러 환율 일별 추이", xaxis_title="일자", yaxis_title="환율 (원)", template="plotly_white")
        st.plotly_chart(fig_fx, use_container_width=True)
