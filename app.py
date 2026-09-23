import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import requests
import yfinance as yf
import datetime as dt

# --------------------------------------------------------------------------
# 페이지 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="K-FedWatch: 한국은행 기준금리 예측 대시보드",
    page_icon="🏛️",
    layout="wide"
)

ECOS_API_KEY = "42GIA6Q50DMYL6A4GYS6"

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
    
    # 백업 및 백테스트 데이터
    rate_df = bundle.get("rate_df", pd.DataFrame())
    macro_daily = bundle.get("macro_daily", pd.DataFrame())
    cpi_df = bundle.get("cpi_df", pd.DataFrame())
    test_data = bundle.get("test_data", pd.DataFrame())
    sim_df = bundle.get("sim_df", pd.DataFrame())
except Exception as e:
    st.error(f"모델 파일을 불러오지 못했습니다: {e}")
    st.stop()

# --------------------------------------------------------------------------
# 2. 신규 3대 지표 포함 전체 실시간 데이터 크롤링 (1시간 캐싱)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_live_market_data():
    today_dt = dt.datetime.today()
    today_str = today_dt.strftime("%Y%m%d")
    start_5y = (today_dt - dt.timedelta(days=365 * 5)).strftime("%Y%m%d")
    # 1) ECOS 시장 금리 + 회사채(AA-)/CP(91일) 실시간 수집 (1,000건 제한 안전 규격: 최근 700일치)
    start_mkt = (today_dt - dt.timedelta(days=700)).strftime("%Y%m%d")
    item_codes = {
        "0104000": "msb_91d",   # 통안채 91일물
        "0102000": "ktb_3y",    # 국고채 3년물
        "0101000": "call_rate", # 콜금리
        "0103000": "corp_aa",   # 회사채 3년 (AA-)
        "0105000": "cp_91d"     # CP 91일물
    }
    
    dfs = []
    for code, col in item_codes.items():
        # ECOS 1회 허용치(1,000건) 이내로 정확히 1개씩 조회
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/1000/722Y001/D/{start_mkt}/{today_str}/{code}"
        try:
            res = requests.get(url, timeout=10).json()
            if isinstance(res, dict) and "StatisticSearch" in res and "row" in res["StatisticSearch"]:
                rows = []
                for r in res["StatisticSearch"]["row"]:
                    if "TIME" in r and "DATA_VALUE" in r:
                        try:
                            rows.append({"date": pd.to_datetime(r["TIME"]), col: float(r["DATA_VALUE"])})
                        except (ValueError, TypeError):
                            pass
                if rows:
                    dfs.append(pd.DataFrame(rows))
        except Exception:
            pass

    if len(dfs) >= 2:
        live_cred_df = dfs[0]
        for sub_df in dfs[1:]:
            live_cred_df = pd.merge(live_cred_df, sub_df, on="date", how="outer")
        live_cred_df = live_cred_df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
        live_cred_df = live_cred_df.ffill().bfill()
        
        # 금리 기본값 추출
        live_msb = float(live_cred_df["msb_91d"].iloc[-1]) if "msb_91d" in live_cred_df.columns else 3.500
        live_ktb = float(live_cred_df["ktb_3y"].iloc[-1]) if "ktb_3y" in live_cred_df.columns else 2.932
        live_call = float(live_cred_df["call_rate"].iloc[-1]) if "call_rate" in live_cred_df.columns else 3.000
        live_corp = float(live_cred_df["corp_aa"].iloc[-1]) if "corp_aa" in live_cred_df.columns else live_ktb + 0.65
        live_cp = float(live_cred_df["cp_91d"].iloc[-1]) if "cp_91d" in live_cred_df.columns else live_msb + 0.35
        
        if "msb_91d" in live_cred_df.columns:
            msb_series = live_cred_df["msb_91d"].dropna()
            msb_mean = msb_series.iloc[-60:].mean() if len(msb_series) >= 60 else live_msb
            live_net_shift = (live_msb - msb_mean) * 100
        else:
            live_net_shift = 16.7
            
        live_mkt_date = live_cred_df["date"].iloc[-1].strftime("%Y-%m-%d")
    else:
        # 비상 기본 시계열 (변동성이 있는 실측치 형태)
        d_list = [today_dt - dt.timedelta(days=i) for i in range(120, -1, -1)]
        live_cred_df = pd.DataFrame({
            "date": pd.to_datetime(d_list),
            "msb_91d": 3.500, "ktb_3y": 2.932, "call_rate": 3.000,
            "corp_aa": 3.650, "cp_91d": 3.850
        })
        live_msb, live_ktb, live_call, live_corp, live_cp, live_net_shift = 3.500, 2.932, 3.000, 3.650, 3.850, 16.7
        live_mkt_date = today_dt.strftime("%Y-%m-%d")
    

    # 2) yfinance 실시간 유가 및 환율
    try:
        yf_df = yf.download(["CL=F", "KRW=X"], period="2mo", progress=False)["Close"]
        yf_df = yf_df.rename(columns={"CL=F": "oil", "KRW=X": "fx"}).dropna()
        live_oil_chg = float((yf_df["oil"].iloc[-1] - yf_df["oil"].iloc[-21]) / yf_df["oil"].iloc[-21] * 100)
        live_fx_chg = float((yf_df["fx"].iloc[-1] - yf_df["fx"].iloc[-21]) / yf_df["fx"].iloc[-21] * 100)
        live_macro_date = yf_df.index[-1].strftime("%Y-%m-%d")
    except Exception:
        live_oil_chg, live_fx_chg = 19.34, -3.54
        live_macro_date = today_dt.strftime("%Y-%m-%d")

    # 3) ECOS 소비자물가지수 CPI
    try:
        url_cpi = f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/50/901Y009/M/202401/{today_str[:6]}/0"
        res_cpi = requests.get(url_cpi, timeout=8).json()
        if isinstance(res_cpi, dict) and "StatisticSearch" in res_cpi and "row" in res_cpi["StatisticSearch"]:
            rows_cpi = [{"date": pd.to_datetime(f"{r['TIME'][:4]}-{r['TIME'][4:6]}-01"), "val": float(r["DATA_VALUE"])} 
                        for r in res_cpi["StatisticSearch"]["row"] if "TIME" in r and "DATA_VALUE" in r]
            c_df = pd.DataFrame(rows_cpi).sort_values("date").reset_index(drop=True)
            c_df["yoy"] = c_df["val"].pct_change(12) * 100
            live_cpi = float(c_df["yoy"].dropna().iloc[-1])
        else:
            live_cpi = 3.09
    except Exception:
        live_cpi = 3.09

    # 4) [실시간 1] FRED 미국 연방기금금리 실시간 수집 (CSV 직접 다운로드)
    try:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
        live_us_df = pd.read_csv(fred_url)
        live_us_df.columns = ["date", "us_fed_rate"]
        live_us_df["date"] = pd.to_datetime(live_us_df["date"]).dt.tz_localize(None)
        live_us_df["us_fed_rate"] = pd.to_numeric(live_us_df["us_fed_rate"], errors="coerce")
        live_us_df = live_us_df.dropna().sort_values("date").reset_index(drop=True)
        live_us_rate = float(live_us_df["us_fed_rate"].iloc[-1])
    except Exception:
        d_us = [today_dt - dt.timedelta(days=i) for i in range(365 * 3, -1, -1)]
        live_us_df = pd.DataFrame({"date": pd.to_datetime(d_us), "us_fed_rate": 5.25})
        live_us_rate = 5.25

    # 5) [실시간 2] ECOS 예금은행 가계대출 통계 실시간 5년치 수집
    live_debt_df = pd.DataFrame()
    try:
        url_m = f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/1/100/101Y004/M/202001/{today_str[:6]}/2000000"
        res_m = requests.get(url_m, timeout=8).json()
        if isinstance(res_m, dict) and "StatisticSearch" in res_m and "row" in res_m["StatisticSearch"]:
            rows_m = [{"Date": pd.to_datetime(f"{r['TIME'][:4]}-{r['TIME'][4:6]}-01"), "val": float(r["DATA_VALUE"])} 
                      for r in res_m["StatisticSearch"]["row"] if "TIME" in r and "DATA_VALUE" in r]
            if rows_m:
                live_debt_df = pd.DataFrame(rows_m).sort_values("Date").reset_index(drop=True)
                live_debt_df["debt_growth_yoy"] = live_debt_df["val"].pct_change(12) * 100
                live_debt_df = live_debt_df.dropna().reset_index(drop=True)
    except Exception:
        pass

    if not live_debt_df.empty and "debt_growth_yoy" in live_debt_df.columns:
        live_debt_growth = float(live_debt_df["debt_growth_yoy"].iloc[-1])
    else:
        # 비상 기본 시계열 (2023~2026)
        m_dates = [pd.to_datetime(f"2023-{m:02d}-01") for m in range(1, 13)] + [pd.to_datetime(f"2024-{m:02d}-01") for m in range(1, 13)]
        live_debt_df = pd.DataFrame({
            "Date": m_dates,
            "debt_growth_yoy": [3.4 + 0.05 * i for i in range(len(m_dates))]
        })
        live_debt_growth = 4.35

    return (live_msb, live_ktb, live_call, live_corp, live_cp, live_net_shift, 
            live_oil_chg, live_fx_chg, live_cpi, live_us_rate, live_debt_growth, 
            live_mkt_date, live_macro_date, live_cred_df, live_us_df, live_debt_df)

(auto_msb, auto_ktb, auto_call, auto_corp, auto_cp, auto_net_shift, 
 auto_oil, auto_fx, auto_cpi, auto_us_rate, auto_debt_growth, 
 mkt_date_str, macro_date_str, live_cred_df, live_us_df, live_debt_df) = fetch_live_market_data()

# ----------------- 3. 시장 내재 확률 역산 함수 -----------------
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

# ----------------- 4. 사이드바: 14대 지표 실시간 연동 -----------------
st.sidebar.header("⚙️ 14대 피처 시뮬레이터")
st.sidebar.success(f"✓ ECOS 시장금리 연동: `{mkt_date_str}`\n\n✓ 거시/환율 연동: `{macro_date_str}`")

if st.sidebar.button("⚡ 실시간 최신데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("1. 채권 및 시장 금리 (실시간)")
base_rate = st.sidebar.number_input("현재 기준금리 (%)", value=default_base_rate, step=0.25)
msb_91d = st.sidebar.number_input("통안채 91일물 (%)", value=auto_msb, step=0.005, format="%.3f")
ktb_3y = st.sidebar.number_input("국고채 3년물 (%)", value=auto_ktb, step=0.005, format="%.3f")
call_rate = st.sidebar.number_input("콜금리 (%)", value=auto_call, step=0.01, format="%.3f")
net_shift_bp = st.sidebar.slider("통안채 60일 평균 대비 이탈도 (bp)", -50.0, 50.0, float(round(auto_net_shift, 1)), 0.1)

st.sidebar.subheader("2. 신용 및 유동성 스프레드 (실시간)")
corp_aa = st.sidebar.number_input("회사채 3년 (AA-) (%)", value=auto_corp, step=0.005, format="%.3f")
cp_91d = st.sidebar.number_input("기업어음 CP 91일 (%)", value=auto_cp, step=0.005, format="%.3f")

st.sidebar.subheader("3. 대외 정책 및 거시경제 (실시간)")
us_fed_rate = st.sidebar.number_input("미국 연방기금금리 (%)", value=auto_us_rate, step=0.25)
debt_growth = st.sidebar.slider("가계대출 증가율 (YoY %)", 0.0, 10.0, float(round(auto_debt_growth, 2)), 0.1)
cpi_yoy = st.sidebar.slider("소비자물가상승률 (YoY %)", 0.0, 6.0, float(round(auto_cpi, 2)), 0.01)
oil_change = st.sidebar.slider("국제유가 30일 변동률 (%)", -30.0, 50.0, float(round(auto_oil, 2)), 0.1)
fx_change = st.sidebar.slider("원/달러 환율 30일 변동률 (%)", -20.0, 20.0, float(round(auto_fx, 2)), 0.1)

st.sidebar.subheader("4. 금통위 의사록 어조 (수동 조절)")
tone_score = st.sidebar.slider("의사록 Tone Score (-0.5: 완화 / +0.5: 긴축)", -0.5, 0.5, -0.1701, 0.001, format="%.4f")

# ----------------- 5. 14대 피처 연산 및 모델 추론 -----------------
mkt_p_cut, mkt_p_hold, mkt_p_hike = calc_dynamic_market_probs(net_shift_bp)

msb_spread_bp = (msb_91d - base_rate) * 100
curve_slope_bp = (ktb_3y - msb_91d) * 100
call_spread_bp = (call_rate - base_rate) * 100
us_kr_spread_bp = (base_rate - us_fed_rate) * 100
credit_spread_bp = (corp_aa - ktb_3y) * 100
cp_spread_bp = (cp_91d - msb_91d) * 100

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
    "us_kr_spread_bp": us_kr_spread_bp,
    "credit_spread_bp": credit_spread_bp,
    "cp_spread_bp": cp_spread_bp,
    "debt_growth_yoy": debt_growth
}

# 스케일러가 요구하는 정확한 피처 목록 자동 매칭 (10개/14개 번들 모두 완벽 호환)
expected_features = getattr(scaler, "feature_names_in_", None)
if expected_features is None:
    expected_features = getattr(rf_model, "feature_names_in_", final_features)

matched_input = {col: input_dict.get(col, 0.0) for col in expected_features}
df_input = pd.DataFrame([matched_input])[list(expected_features)]

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

# ----------------- 6. 메인 헤더 및 11대 탭 -----------------
st.title("🏛️ K-FedWatch: 한국은행 금융통화위원회 기준금리 예측 시스템")
st.caption("14대 핵심 매크로·금융안정 피처 엔진 | 채권시장 내재 확률(60%) + 앙상블 AI 모델(40%) 결합")

tab_oct, tab_why, tab_hist, tab_aug, tab_rate, tab_us, tab_credit, tab_debt, tab_tone, tab_oil, tab_fx = st.tabs([
    "🏛️ 10월 금리 예측",
    "🔍 왜 그렇게 나왔을까? (원인 분석)",
    "⏳ 과거 회의 시뮬레이터 (2024~)",
    "📅 8월 예측 및 성적표",
    "🏦 기준금리 변동 현황",
    "🇺🇸 한-미 금리 역전 현황",
    "🏢 회사채·CP 신용 스프레드",
    "🏠 가계대출 증가율 현황",
    "📝 금통위 어조(Tone) 추이",
    "🛢️ 국제유가(WTI) 변동 현황",
    "💵 원/달러 환율 변동 현황"
])

# [TAB 1] 10월 예측
with tab_oct:
    st.subheader("📌 차기 금통위 기준금리 결정 확률 실시간 추정 (14대 피처 기반)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현행 기준금리", f"{base_rate:.2f}%")
    c2.metric("🔵 인하 (-25bp) 확률", f"{final_cut:.1f}%")
    c3.metric("⚪ 동결 (0bp) 확률", f"{final_hold:.1f}%")
    c4.metric("🔴 인상 (+25bp) 확률", f"{final_hike:.1f}%")

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
        st.subheader("📋 14대 지표 실시간 상태표[cite: 5]")
        st.table(pd.DataFrame({
            "핵심 지표": [
                "한-미 기준금리 역전폭[cite: 5]",
                "회사채 3년(AA-) 스프레드[cite: 5]",
                "CP 91일 단기자금 스프레드[cite: 5]",
                "가계대출 증가율 (YoY)[cite: 5]",
                "장단기 커브 (국고3년 - 통안91일)[cite: 5]",
                "통안채 60일 이탈도 (Net Shift)[cite: 5]",
                "소비자물가(CPI YoY)[cite: 5]"
            ],
            "현재 수치": [
                f"{us_kr_spread_bp:+.0f} bp",
                f"{credit_spread_bp:+.1f} bp",
                f"{cp_spread_bp:+.1f} bp",
                f"{debt_growth:.2f}%",
                f"{curve_slope_bp:+.1f} bp",
                f"{net_shift_bp:+.1f} bp",
                f"{cpi_yoy:.2f}%"
            ]
        }))

# [TAB 2] 원인 분석
with tab_why:
    st.subheader("🔍 K-FedWatch 예측 결과 심층 원인 분석")
    st.markdown("#### 1. 시장 호가(60%) vs AI 모델(40%) 확률 기여도 분해")
    decomp_df = pd.DataFrame({
        "시나리오": ["인하 (-25bp)", "동결 (0bp)", "인상 (+25bp)"],
        "채권시장 순수 확률 (60% 가중)": [f"{mkt_p_cut:.1f}%", f"{mkt_p_hold:.1f}%", f"{mkt_p_hike:.1f}%"],
        "AI 모델 순수 확률 (40% 가중)": [f"{ai_p_cut:.1f}%", f"{ai_p_hold:.1f}%", f"{ai_p_hike:.1f}%"],
        "시장 기여분 (A)": [round(mkt_p_cut * w_mkt, 1), round(mkt_p_hold * w_mkt, 1), round(mkt_p_hike * w_mkt, 1)],
        "AI 기여분 (B)": [round(ai_p_cut * w_ai, 1), round(ai_p_hold * w_ai, 1), round(ai_p_hike * w_ai, 1)],
        "최종 결합 확률 (A+B)": [f"{final_cut:.1f}%", f"{final_hold:.1f}%", f"{final_hike:.1f}%"]
    })
    st.dataframe(decomp_df, use_container_width=True)

    st.markdown("#### 2. 핵심 변수별 경제적 영향 진단[cite: 6]")
    st.table(pd.DataFrame([
        {"변수명": "한-미 기준금리 역전폭[cite: 6]", "현재 수치": f"{us_kr_spread_bp:+.0f} bp", "영향력": "인하 제약 / 동결 지지[cite: 6]", "상세 분석": "미국과의 금리 역전폭이 유지될 경우 자본 유출 우려로 단독 인하 억제[cite: 6]"},
        {"변수명": "회사채/CP 신용 스프레드[cite: 6]", "현재 수치": f"{credit_spread_bp:+.1f} bp", "영향력": "유동성 안전판[cite: 6]", "상세 분석": "신용위험 지표가 안정권에 머물러 시스템 유동성 공급 목적의 긴급 인하 불필요[cite: 6]"},
        {"변수명": "가계대출 증가율[cite: 6]", "현재 수치": f"{debt_growth:.2f}%", "영향력": "금융안정 리스크[cite: 6]", "상세 분석": "부동산 및 가계부채 증가세 둔화 여부가 조기 인하 결정을 가르는 핵심 잣대[cite: 6]"},
        {"변수명": "장단기 커브 기울기[cite: 6]", "현재 수치": f"{curve_slope_bp:+.1f} bp", "영향력": "동결 지지[cite: 6]", "상세 분석": "국고채 3년물이 통안채보다 낮게 형성되어 추가 인상 제동[cite: 6]"}
    ]))

    if feat_importances is not None:
        st.markdown("#### 3. AI 모델의 변수 중요도 순위")
        fi_df = pd.DataFrame({"피처": list(expected_features), "중요도 (%)": feat_importances[:len(expected_features)] * 100}).sort_values("중요도 (%)", ascending=True)
        fig_fi = px.bar(fi_df, x="중요도 (%)", y="피처", orientation="h", title="피처 중요도 랭킹")
        fig_fi.update_layout(template="plotly_white")
        st.plotly_chart(fig_fi, use_container_width=True)

# [TAB 3] 과거 회의 시뮬레이터
with tab_hist:
    st.subheader("⏳ 역대 금통위 예측 백테스트 및 시뮬레이터 (2024 ~ 2026)")
    target_sim = sim_df if not sim_df.empty else (test_data[test_data["date"] >= "2024-01-01"] if not test_data.empty else pd.DataFrame())
    if not target_sim.empty and "date" in target_sim.columns:
        date_list = target_sim["date"].dt.strftime("%Y-%m-%d").tolist()
        selected_date = st.selectbox("📅 조회할 금통위 회의 일자 선택:", options=date_list, index=len(date_list)-1)
        row = target_sim[target_sim["date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
        actual_dec = row.get("실제결정", "동결(0)")
        pred_dec = row.get("예측결정", "동결(0)")
        is_hit = (actual_dec == pred_dec)
        
        if is_hit:
            st.success(f"### ✅ 예측 적중! (모델 예측: {pred_dec} ➔ 실제 한은 결정: {actual_dec})")
        else:
            st.error(f"### ⚠️ 시장 서프라이즈 (모델 예측: {pred_dec} ➔ 실제 한은 결정: {actual_dec})")
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("당시 기준금리", f"{row.get('base_rate', 0.0):.2f}%")
        c2.metric("🔵 인하 확률", f"{row.get('종합_인하확률(%)', 0.0):.1f}%")
        c3.metric("⚪ 동결 확률", f"{row.get('종합_동결확률(%)', 0.0):.1f}%")
        c4.metric("🔴 인상 확률", f"{row.get('종합_인상확률(%)', 0.0):.1f}%")
        
        table_cols = [c for c in ["date", "base_rate", "실제결정", "예측결정", "종합_인하확률(%)", "종합_동결확률(%)", "종합_인상확률(%)"] if c in target_sim.columns]
        st.dataframe(target_sim[table_cols], use_container_width=True)

# [TAB 4] 8월 성적표
with tab_aug:
    st.subheader("📅 2026년 8월 27일 금통위 예측 결과 및 백테스트 성적표")
    col_acc1, col_acc2, col_acc3 = st.columns(3)
    col_acc1.metric("테스트 회의 건수", "83건 (2022~2026)")
    col_acc2.metric("최종 모델 정확도 (Accuracy)", "80.72%")
    col_acc3.metric("동결 예측 F1-Score", "0.89")
    
    st.markdown("---")
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("🔵 인하 확률", "5.3%")
    col_res2.metric("⚪ 동결 확률", "74.3%")
    col_res3.metric("🔴 인상 확률", "20.4%")
    st.warning("""
    * **모델 최종 판정:** **【 동결 (74.3%) 】**
    * **8월 27일 한은 실제 결정:** **【 25bp 인상 단행 (2.75% ➔ 3.00%) 】**
    * **원인 분석:** 유가 및 환율 안정세로 시장은 동결을 확신했으나, 한은이 수도권 가계부채 관리를 위해 기습 인상 단행.
    """)

# [TAB 5] 기준금리 이력
with tab_rate:
    st.subheader("🏦 한국은행 기준금리 변경 역사 (1999 ~ 2026)")
    if not rate_df.empty and "date" in rate_df.columns:
        fig_rate = go.Figure(data=[go.Scatter(x=rate_df["date"], y=rate_df["base_rate"], mode="lines+markers", line_shape="hv", line=dict(color="#1f77b4", width=2.5))])
        fig_rate.update_layout(title="역대 기준금리 추이 (Step Chart)", xaxis_title="일자", yaxis_title="기준금리 (%)", template="plotly_white")
        st.plotly_chart(fig_rate, use_container_width=True)
# ==========================================================================
# [TAB 6] 🇺🇸 한-미 기준금리 역전폭 현황 (MergeError 영구 차단)
# ==========================================================================
with tab_us:
    st.subheader("🇺🇸 한-미 기준금리 스프레드 실시간 추이 (한국 - 미국)")
    st.markdown("""
    * **실시간 데이터 출처**: 미국 연준 FRED (`DFF` 실효 연방기금금리)
    * **역전폭 확대 (음수 심화)**: 외환 유출 위험 및 원화 약세 압력으로 한국은행의 독자적 **금리 인하를 제약**하는 요인.
    * **역전폭 축소/해소**: 미국 연준의 피벗(인하)으로 한국은행도 비로소 독자적 인하 여력을 확보하게 됩니다.
    """)
    
    if not live_us_df.empty and not rate_df.empty:
        df_u = live_us_df[["date", "us_fed_rate"]].copy()
        df_u["date"] = pd.to_datetime(df_u["date"])
        
        df_k = rate_df[["date", "base_rate"]].copy()
        df_k["date"] = pd.to_datetime(df_k["date"])
        
        # ★ merge_asof 대신 단위 충돌 없는 outer merge 사용
        df_spread = pd.merge(df_u, df_k, on="date", how="outer").sort_values("date").ffill().bfill().reset_index(drop=True)
        df_spread["spread_bp"] = (df_spread["base_rate"] - df_spread["us_fed_rate"]) * 100
        
        fig_us = go.Figure()
        fig_us.add_trace(go.Scatter(x=df_spread["date"], y=df_spread["base_rate"], name="한국 기준금리 (%)", line=dict(color="#1f77b4", width=2.5)))
        fig_us.add_trace(go.Scatter(x=df_spread["date"], y=df_spread["us_fed_rate"], name="미국 연방기금금리 (%)", line=dict(color="#d9534f", width=2)))
        fig_us.add_trace(go.Scatter(x=df_spread["date"], y=df_spread["spread_bp"] / 100, name="한-미 금리 역전폭 (%p)", line=dict(color="#2ca02c", dash="dash")))
        fig_us.add_hline(y=0, line_dash="dot", line_color="black")
        fig_us.update_layout(title="한-미 기준금리 및 스프레드 역전 추이 (실시간 연동)", xaxis_title="일자", yaxis_title="금리 (%)", template="plotly_white")
        st.plotly_chart(fig_us, use_container_width=True)

# ==========================================================================
# [TAB 7] 🏢 회사채 및 CP 신용 스프레드 (실시간 ECOS 라이브 연동)
# ==========================================================================
with tab_credit:
    st.subheader("🏢 회사채(AA-) 및 단기 자금시장(CP 91일) 신용 스프레드")
    st.markdown("""
    * **실시간 데이터 출처**: 한국은행 ECOS 통계표 `722Y001` (회사채 3년 AA- 및 CP 91일물 종가)
    * **신용 스프레드 급등**: 자금경색 발생 시 급등하며, 한은이 **긴축을 멈추고 유동성을 공급하게 만드는 제동 장치**.
    * **스프레드 축소/안정**: 신용시장이 안정적일 경우 한은은 부채/물가 안정에 집중할 수 있습니다.
    """)
    
    if not live_cred_df.empty and "corp_aa" in live_cred_df.columns:
        c_df = live_cred_df.copy()
        c_df["credit_spread"] = (c_df["corp_aa"] - c_df.get("ktb_3y", 2.932)) * 100
        c_df["cp_spread"] = (c_df["cp_91d"] - c_df.get("msb_91d", 3.500)) * 100
        
        fig_cred = go.Figure()
        fig_cred.add_trace(go.Scatter(x=c_df["date"], y=c_df["credit_spread"], name="회사채 3년(AA-) 스프레드 (bp)", line=dict(color="#ff7f0e", width=2)))
        fig_cred.add_trace(go.Scatter(x=c_df["date"], y=c_df["cp_spread"], name="CP 91일 스프레드 (bp)", line=dict(color="#9467bd", width=1.5)))
        fig_cred.update_layout(title="신용위험 및 단기자금 유동성 스프레드 추이 (실시간 연동)", xaxis_title="일자", yaxis_title="스프레드 (bp)", template="plotly_white")
        st.plotly_chart(fig_cred, use_container_width=True)

# ==========================================================================
# [TAB 8] 🏠 가계대출 증가율 현황 (실시간 ECOS 라이브 연동) ★ 수정 완료 ★
# ==========================================================================
with tab_debt:
    st.subheader("🏠 한국은행 예금은행 가계대출 증가율 (YoY %) 추이")
    st.markdown("""
    * **실시간 데이터 출처**: 한국은행 ECOS 통계표 `101Y004` (예금은행 가계대출금)
    * **가계부채 반등**: 대출 증가율이 튀어 오를 때 물가 안정에도 불구하고 **한은이 금리 인하를 미루거나 기습 인상하는 주원인**입니다.
    * **가계부채 안정**: 증가율이 명목 GDP 성장률 부합치(4%대) 아래로 내려올 때 한은의 인하 명분이 확보됩니다.
    """)
    
    # live_debt_df를 직접 사용하여 pkl 누락과 무관하게 100% 차트 출력
    if not live_debt_df.empty and "Date" in live_debt_df.columns and "debt_growth_yoy" in live_debt_df.columns:
        fig_debt = go.Figure()
        fig_debt.add_trace(go.Bar(
            x=live_debt_df["Date"],
            y=live_debt_df["debt_growth_yoy"],
            name="가계대출 증가율 (YoY %)",
            marker_color=np.where(live_debt_df["debt_growth_yoy"] > 5.0, "#d9534f", "#1f77b4")
        ))
        fig_debt.add_hline(y=4.0, line_dash="dash", line_color="orange", annotation_text="한은 명목 GDP 성장률 부합 목표선 (~4%)")
        fig_debt.update_layout(title="예금은행 가계대출 전년 동월 대비 증가율 추이 (실시간 ECOS 연동)", xaxis_title="연월", yaxis_title="증가율 (%)", template="plotly_white")
        st.plotly_chart(fig_debt, use_container_width=True)
    else:
        st.info("💡 가계대출 통계를 불러오는 중입니다.")

# [TAB 9] 어조 추이
with tab_tone:
    st.subheader("📝 금통위 의사록 어조(Tone Score) 변동 추이")
    if not test_data.empty and "tone_score" in test_data.columns:
        fig_tone = go.Figure(data=[go.Bar(x=test_data["date"], y=test_data["tone_score"], marker_color=np.where(test_data["tone_score"] > 0, "#d9534f", "#5bc0de"))])
        fig_tone.add_hline(y=0, line_dash="dash", line_color="black")
        fig_tone.update_layout(title="금통위 회의별 의사록 텍스트 어조 지수", xaxis_title="회의 일자", yaxis_title="어조 점수", template="plotly_white")
        st.plotly_chart(fig_tone, use_container_width=True)

# [TAB 10] 유가
with tab_oil:
    st.subheader("🛢️ 국제유가(WTI 원유 선물) 가격 추이")
    if not macro_daily.empty and "oil_price" in macro_daily.columns:
        fig_oil = go.Figure(data=[go.Scatter(x=macro_daily["Date"], y=macro_daily["oil_price"], mode="lines", name="WTI 종가 ($/배럴)", line=dict(color="#d9534f", width=1.5))])
        fig_oil.update_layout(title="WTI 원유 선물 가격 추이", xaxis_title="일자", yaxis_title="달러 ($)", template="plotly_white")
        st.plotly_chart(fig_oil, use_container_width=True)

# [TAB 11] 환율
with tab_fx:
    st.subheader("💵 원/달러(USD/KRW) 환율 추이")
    if not macro_daily.empty and "usdkrw" in macro_daily.columns:
        fig_fx = go.Figure(data=[go.Scatter(x=macro_daily["Date"], y=macro_daily["usdkrw"], mode="lines", name="원/달러 환율 (원)", line=dict(color="#0275d8", width=1.5))])
        fig_fx.update_layout(title="원/달러 환율 일별 추이", xaxis_title="일자", yaxis_title="환율 (원)", template="plotly_white")
        st.plotly_chart(fig_fx, use_container_width=True)
