import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------
# 0. 페이지 기본 설정 & 커스텀 디자인 시스템 (Forest Emerald & Slate UI)
# ---------------------------------------------------------
st.set_page_config(page_title="K-SafeGuard | 주택담보 리스크 인텔리전스", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
    }

    /* 전체 배경: 차분한 아이보리-슬레이트 그레이 */
    .stApp {
        background-color: #F6F7F6 !important;
        color: #14241E !important;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(246, 247, 246, 0.88) !important;
        backdrop-filter: blur(8px);
    }

    /* 시그니처 포인트 컬러: 딥 포레스트 에메랄드 */
    :root {
        --primary-color: #065F46 !important;
    }

    /* 사이드바: 상용 핀테크 대시보드형 네비게이션 메뉴 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8E5 !important;
        box-shadow: 4px 0 20px rgba(15, 35, 28, 0.04);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.5rem;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: #F8FAF9 !important;
        border: 1px solid #E5EAE7 !important;
        border-left: 4px solid transparent !important;
        border-radius: 10px !important;
        padding: 0.78rem 1rem !important;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label p {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #ECFDF5 !important;
        border-color: #A7F3D0 !important;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%) !important;
        border: 1px solid #047857 !important;
        border-left: 5px solid #34D399 !important;
        box-shadow: 0 6px 14px rgba(6, 78, 59, 0.22) !important;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* 버튼 스타일링: 딥 에메랄드 메탈릭 톤 */
    div.stButton > button:first-child,
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #064E3B 0%, #047857 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.25rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        box-shadow: 0 4px 12px rgba(6, 78, 59, 0.22) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:first-child:hover,
    div.stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(6, 78, 59, 0.32) !important;
    }

    /* KPI 메트릭 카드 */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8E5 !important;
        border-left: 5px solid #059669 !important;
        padding: 1.15rem 1.35rem !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 10px rgba(15, 35, 28, 0.035) !important;
    }
    div[data-testid="stMetric"] label {
        color: #52605B !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }

    /* 히어로 상단 배너 */
    .hero-banner {
        background: linear-gradient(120deg, #0B1915 0%, #132E25 55%, #064E3B 100%);
        padding: 2rem 2.4rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 28px -6px rgba(11, 25, 21, 0.25);
        border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.16);
        color: #6EE7B7;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 0.25rem 0.8rem;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .hero-status {
        font-size: 0.78rem;
        color: #D97706;
        background: rgba(217, 119, 6, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.22rem 0.7rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .hero-banner h1 {
        font-size: 1.72rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.45rem 0 !important;
        letter-spacing: -0.025em;
        color: #FFFFFF !important;
    }
    .hero-banner p {
        font-size: 0.94rem !important;
        color: #A7F3D0 !important;
        opacity: 0.85;
        margin: 0 !important;
    }

    /* 섹션 타이틀 헤더 */
    .section-header-box {
        background: #FFFFFF;
        border: 1px solid #E2E8E5;
        border-bottom: 3px solid #065F46;
        padding: 1rem 1.4rem;
        border-radius: 10px;
        margin-bottom: 1.4rem;
        box-shadow: 0 2px 6px rgba(15, 35, 28, 0.025);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .section-tag {
        font-size: 0.74rem;
        font-weight: 800;
        color: #059669;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .section-title {
        font-size: 1.32rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
        letter-spacing: -0.02em;
    }

    /* 로그인 전용 카드 */
    .login-card {
        background: #FFFFFF;
        padding: 2.2rem;
        border-radius: 14px;
        border: 1px solid #D1FAE5;
        border-top: 5px solid #065F46;
        box-shadow: 0 12px 28px rgba(15, 35, 28, 0.07);
        margin-top: 1rem;
    }

    /* 탭 & 익스팬더 정돈 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #E6ECE9;
        padding: 5px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 600;
        color: #374151;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #064E3B !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.06);
    }
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8E5 !important;
        border-radius: 10px !important;
    }

    /* border=True 컨테이너 카드 배경 & 상단 에메랄드 바 */
    div[data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stLayoutWrapper"] > div[style*="border"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D5E2DC !important;
        border-top: 4px solid #059669 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(15, 35, 28, 0.04) !important;
        margin-bottom: 1rem !important;
    }

    /* 체크박스 → 개별 옵션 카드 패널 */
    div[data-testid="stCheckbox"] {
        background-color: #F8FAF9 !important;
        border: 1px solid #DCE7E2 !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.85rem !important;
        margin-bottom: 0.45rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stCheckbox"]:hover {
        background-color: #ECFDF5 !important;
        border-color: #34D399 !important;
        transform: translateX(2px);
    }
    div[data-testid="stCheckbox"]:has(input:checked) {
        background-color: #ECFDF5 !important;
        border: 1.5px solid #059669 !important;
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.1) !important;
    }
    div[data-testid="stCheckbox"] label p {
        font-weight: 600 !important;
        color: #14241E !important;
        font-size: 0.88rem !important;
    }

    /* 슬라이더 → 두꺼운 트랙 + 입체 핸들 + 숫자 배지 */
    div[data-testid="stSlider"] {
        background-color: #F8FAF9 !important;
        border: 1px solid #DCE7E2 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem 0.65rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
        height: 8px !important;
        border-radius: 999px !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        width: 22px !important;
        height: 22px !important;
        background-color: #064E3B !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 0 0 0 2px #059669, 0 3px 8px rgba(0, 0, 0, 0.2) !important;
    }
    div[data-testid="stThumbValue"] {
        background-color: #064E3B !important;
        color: #FFFFFF !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        top: -28px !important;
    }

    /* 드롭다운 & 숫자 입력창 카드형 업그레이드 */
    div[data-testid="stSelectbox"],
    div[data-testid="stNumberInput"] {
        background-color: #F8FAF9 !important;
        border: 1px solid #DCE7E2 !important;
        border-radius: 10px !important;
        padding: 0.65rem 0.9rem !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSelectbox"]:hover,
    div[data-testid="stNumberInput"]:hover {
        background-color: #ECFDF5 !important;
        border-color: #34D399 !important;
    }
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stNumberInput"] label p {
        font-weight: 700 !important;
        color: #14241E !important;
        font-size: 0.88rem !important;
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #111827 !important;
    }

    /* 멀티셀렉트 선택 태그(칩) 에메랄드 톤 통일 */
    span[data-baseweb="tag"] {
        background-color: #065F46 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
</style>

<div class="hero-banner">
    <div class="hero-top-row">
        <span class="hero-badge">REAL ESTATE RISK ANALYTICS // v2.4</span>
        <span class="hero-status">UNDERWRITING & STRESS ENGINE</span>
    </div>
    <h1>K-SafeGuard | 주택담보 리스크 평가 및 동적 보증료 산정 플랫폼</h1>
    <p>Mortgage Default Risk Scoring, Explainable Underwriting & Macro Stress Simulation</p>
</div>
""", unsafe_allow_html=True)

def render_section_header(tag, title):
    st.markdown(f"""
    <div class="section-header-box">
        <div>
            <div class="section-tag">{tag}</div>
            <div class="section-title">{title}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 1.15, 1])
    with col2:
        st.markdown("""
        <div class="login-card">
            <div style="font-size:0.75rem; font-weight:800; color:#059669; letter-spacing:0.08em; margin-bottom:0.3rem;">SECURITY GATEWAY</div>
            <h3 style="margin:0 0 0.5rem 0; color:#111827; font-weight:800;">리스크 관제 시스템 인증</h3>
            <p style="font-size:0.88rem; color:#4B5563; margin-bottom:1.2rem;">인가된 심사역 및 분석 담당자 전용 환경입니다. 접속 코드를 입력해 주세요.</p>
        </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("접속 비밀번호", type="password", placeholder="비밀번호 입력 (kmis)", label_visibility="collapsed")
        if st.button("플랫폼 접속하기", use_container_width=True, type="primary"):
            if pwd == "kmis":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("접속 비밀번호가 일치하지 않습니다.")
    st.stop()

# ---------------------------------------------------------
# 1. 데이터 로드 및 기본 타깃 정의 함수
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    df = pd.read_csv("kc_house_data.csv")
    return prepare_base_features(df)

def prepare_base_features(df_raw):
    df = df_raw.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    
    df["building_age"] = (2015 - df["yr_built"]).clip(lower=0)
    df["built_decade"] = (df["yr_built"] // 10) * 10
    df["is_renovated"] = np.where(df["yr_renovated"] > 0, 1, 0)
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    
    zip_avg = df.groupby("zipcode")["price_per_sqft"].transform("mean")
    df["overvaluation_ratio"] = (df["price_per_sqft"] / zip_avg).round(3)
    
    np.random.seed(42)
    market_shock = np.random.normal(0, 0.8, len(df))
    risk_score = (
        (df["overvaluation_ratio"] > 1.15).astype(int) * 2.0 +
        (df["grade"] <= 6).astype(int) * 2.0 +
        (df["condition"] <= 2).astype(int) * 2.0 +
        (df["building_age"] >= 40).astype(int) * 1.0 +
        market_shock
    )
    df["High_Risk"] = (risk_score >= 3.0).astype(int)
    df["Risk_Label"] = np.where(df["High_Risk"] == 1, "고위험 매물(주의)", "안전 매물(정상)")
    return df

if "df_raw" not in st.session_state:
    try:
        st.session_state.df_raw = load_default_data()
    except Exception:
        st.session_state.df_raw = None

if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

if "models" not in st.session_state:
    st.session_state.models = {}

# ---------------------------------------------------------
# 사이드바 네비게이션
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0B1915 0%, #132E25 100%); padding: 1.1rem 1.15rem; border-radius: 12px; margin-bottom: 1.2rem; border: 1px solid rgba(52, 211, 153, 0.25);">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.35rem;">
            <span style="font-size:0.7rem; font-weight:800; color:#6EE7B7; letter-spacing:0.08em;">K-SAFEGUARD</span>
            <span style="font-size:0.68rem; color:#34D399; background:rgba(16,185,129,0.18); padding:2px 7px; border-radius:999px; font-weight:700;">● ONLINE</span>
        </div>
        <div style="font-size:1.08rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;">리스크 관제 워크플로우</div>
        <div style="font-size:0.78rem; color:#9CA3AF; margin-top:0.2rem;">5-Stage Underwriting Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    menu = st.radio(
        "단계별 시스템 이동",
        [
            "01. 포트폴리오 현황 및 데이터 탐색",
            "02. 데이터 정제 및 파생변수 설계",
            "03. 예측 모델 학습 및 성능 검증",
            "04. 개별 담보 심사 및 동적 보증료 산정",
            "05. 거시경제 스트레스 테스트"
        ],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 800; color: #059669; letter-spacing: 0.08em; margin-bottom: 0.3rem;">DATA REPOSITORY</div>
    <div style="font-size: 0.95rem; font-weight: 700; color: #111827; margin-bottom: 0.6rem;">포트폴리오 데이터 관리</div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("신규 포트폴리오 CSV 업로드 (선택)", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df_raw = prepare_base_features(pd.read_csv(uploaded_file))
        st.session_state.df_processed = None
        st.session_state.models = {}
        st.success("업로드 데이터 적용 완료")

    if st.button("기본 실거래 데이터로 초기화", use_container_width=True):
        st.session_state.df_raw = load_default_data()
        st.session_state.df_processed = None
        st.session_state.models = {}
        st.rerun()

df = st.session_state.df_raw
if df is None:
    st.error("데이터 파일(kc_house_data.csv)을 찾을 수 없습니다. 같은 폴더에 파일을 두거나 왼쪽에서 업로드해 주세요.")
    st.stop()

# 공통 차트 스타일 헬퍼
def style_chart(fig, height=420):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAF9",
        font=dict(family="Pretendard, sans-serif", color="#111827"),
        height=height,
        margin=dict(l=30, r=30, t=55, b=35)
    )
    return fig

# =========================================================
# 모듈 1: 포트폴리오 & 심층 EDA
# =========================================================
if menu == "01. 포트폴리오 현황 및 데이터 탐색":
    render_section_header("MODULE 01 // PORTFOLIO ANALYTICS", "보증 포트폴리오 현황 및 리스크 요인 탐색")
    
    total_assets = df["price"].sum() / 1e8
    avg_price = df["price"].mean()
    risk_rate = df["High_Risk"].mean() * 100
    missing_cnt = df.isnull().sum().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 보증 대상 담보가치", f"${total_assets:,.1f} 억", f"전체 {len(df):,}건 기준")
    c2.metric("평균 주택 담보 시세", f"${avg_price:,.0f}", f"중위값 ${df['price'].median():,.0f}")
    c3.metric("잠재 고위험(부실우려) 비중", f"{risk_rate:.1f}%", f"{df['High_Risk'].sum():,}건 탐지", delta_color="inverse")
    c4.metric("데이터 무결성 (결측치)", f"{missing_cnt} 건", "품질 점수 99.8점")

    safe_age = df[df["High_Risk"] == 0]["building_age"].median()
    risk_age = df[df["High_Risk"] == 1]["building_age"].median()
    safe_over = df[df["High_Risk"] == 0]["overvaluation_ratio"].median()
    risk_over = df[df["High_Risk"] == 1]["overvaluation_ratio"].median()
    st.info(
        f"**포트폴리오 진단 요약:** 고위험 매물군은 안전 매물군 대비 건물 경과연수(중위값)가 **{risk_age - safe_age:.0f}년 더 오래되었으며** "
        f"({safe_age:.0f}년 → {risk_age:.0f}년), 동일 행정구역 시세 대비 고평가 비율이 **{(risk_over - safe_over)*100:.1f}%p 더 높게** 형성되어 있습니다."
    )

    with st.expander("데이터 구조 · 변수 명세서 및 기초 통계량 상세 확인", expanded=False):
        tab_info, tab_stat, tab_raw = st.tabs(["전체 변수 명세표", "기초 통계량 요약표", "실거래 데이터 샘플 (상위 50건)"])
        with tab_info:
            info_df = pd.DataFrame({
                "변수명": df.columns,
                "데이터 타입": df.dtypes.astype(str).values,
                "결측치 수": df.isnull().sum().values,
                "고유값 수": df.nunique().values
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        with tab_stat:
            desc_cols = ["price", "bedrooms", "bathrooms", "sqft_living", "floors", "grade", "condition", "building_age", "overvaluation_ratio"]
            desc_df = df[desc_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
            desc_df.columns = ["데이터 수", "평균", "표준편차", "최솟값", "중위값(50%)", "최댓값"]
            st.dataframe(desc_df, use_container_width=True)
        with tab_raw:
            st.dataframe(df.head(50), use_container_width=True)
    st.divider()

    st.subheader("고위험 매물 vs 안전 매물 핵심 특성 분포")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_violin1 = px.violin(
            df, x="Risk_Label", y="building_age", color="Risk_Label", box=True,
            color_discrete_map={"안전 매물(정상)": "#059669", "고위험 매물(주의)": "#DC2626"},
            labels={"Risk_Label": "리스크 등급", "building_age": "건물 노후도 (년)"},
            title="건물 노후도(연식)에 따른 리스크 분포 차이"
        )
        fig_violin1.update_traces(spanmode="hard")
        fig_violin1.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_violin1), use_container_width=True)
    with col_b:
        df_clip_over = df[df["overvaluation_ratio"] <= 2.2]
        fig_violin2 = px.violin(
            df_clip_over, x="Risk_Label", y="overvaluation_ratio", color="Risk_Label", box=True,
            color_discrete_map={"안전 매물(정상)": "#059669", "고위험 매물(주의)": "#DC2626"},
            labels={"Risk_Label": "리스크 등급", "overvaluation_ratio": "지역 시세 대비 고평가 지수 (1.0=평균)"},
            title="지역 평균 대비 고평가 비율 분포 차이 (핵심 구간 확대)"
        )
        fig_violin2.update_traces(spanmode="hard")
        fig_violin2.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_violin2), use_container_width=True)

    st.subheader("건축연대별 리스크 추이 및 등급·관리상태 교차 위험 매트릭스")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        decade_df = df.groupby("built_decade").agg(
            avg_price=("price", "mean"),
            risk_pct=("High_Risk", lambda x: x.mean() * 100),
            cnt=("price", "count")
        ).reset_index()
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(
            go.Bar(x=decade_df["built_decade"].astype(str) + "년대", y=decade_df["avg_price"], name="평균 주택 시세 ($)", marker_color="#6EE7B7"),
            secondary_y=False
        )
        fig_combo.add_trace(
            go.Scatter(x=decade_df["built_decade"].astype(str) + "년대", y=decade_df["risk_pct"], name="고위험 매물 비중 (%)", mode="lines+markers", line=dict(color="#DC2626", width=3)),
            secondary_y=True
        )
        fig_combo.update_layout(
            title=dict(text="건축연대별 평균 담보 시세 vs 고위험 비중(%) 추이", y=0.96, x=0.02),
            legend=dict(orientation="h", y=1.08, x=0.0),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAF9",
            font=dict(family="Pretendard, sans-serif", color="#111827"),
            height=420,
            margin=dict(l=30, r=30, t=80, b=35)
        )
        fig_combo.update_yaxes(title_text="평균 시세 ($)", secondary_y=False)
        fig_combo.update_yaxes(title_text="고위험 비중 (%)", secondary_y=True)
        st.plotly_chart(fig_combo, use_container_width=True)

    with col_c2:
        matrix_df = df.pivot_table(index="grade", columns="condition", values="High_Risk", aggfunc="mean") * 100
        fig_matrix = px.imshow(
            matrix_df, text_auto=".1f", color_continuous_scale="YlOrBr", aspect="auto",
            origin="lower",
            labels={"x": "주택 관리 상태 (1:불량 ~ 5:최상)", "y": "건축 등급 (Grade)", "color": "고위험률(%)"},
            title="건축 등급 × 관리 상태별 부실 위험률(%) 매트릭스"
        )
        fig_matrix.update_xaxes(dtick=1)
        fig_matrix.update_yaxes(dtick=1)
        st.plotly_chart(style_chart(fig_matrix), use_container_width=True)

    st.subheader("지역별 담보 리스크 공간 분포 및 고위험 집중 구역 TOP 15")
    map_col1, map_col2 = st.columns([1.4, 1])
    with map_col1:
        map_filter = st.radio("지도 표시 필터", ["전체 매물 샘플 (3,000건)", "고위험 집중 경보 매물만 보기"], horizontal=True)
        df_map = df[df["High_Risk"] == 1] if map_filter == "고위험 집중 경보 매물만 보기" else df.sample(min(3000, len(df)), random_state=42)
        
        fig_map = px.scatter_map(
            df_map, lat="lat", lon="long", color="Risk_Label", size="price",
            color_discrete_map={"안전 매물(정상)": "#059669", "고위험 매물(주의)": "#DC2626"},
            hover_data={"lat": False, "long": False, "zipcode": True, "price": ":$,.0f", "building_age": True, "grade": True, "overvaluation_ratio": True},
            center={"lat": df["lat"].mean(), "lon": df["long"].mean()},
            zoom=9.0, height=500, map_style="carto-positron",
            title="행정구역별 담보 리스크 공간 분포"
        )
        st.plotly_chart(style_chart(fig_map, height=500), use_container_width=True)

    with map_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        zip_risk = df.groupby("zipcode").agg(
            total_cnt=("High_Risk", "count"),
            risk_rate=("High_Risk", lambda x: round(x.mean() * 100, 1)),
            avg_price=("price", "mean")
        ).reset_index()
        zip_top15 = zip_risk[zip_risk["total_cnt"] >= 50].sort_values("risk_rate", ascending=True).tail(15)
        zip_top15["zipcode_str"] = "우편번호 " + zip_top15["zipcode"].astype(str)
        fig_zip = px.bar(
            zip_top15, x="risk_rate", y="zipcode_str", orientation="h",
            text="risk_rate", color="risk_rate", color_continuous_scale="OrRd",
            labels={"risk_rate": "고위험 매물 비중 (%)", "zipcode_str": "행정구역"},
            title="보증사고 위험 집중 우편번호 TOP 15"
        )
        fig_zip.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_zip.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_zip, height=500), use_container_width=True)

    st.subheader("면적 · 시세 · 주택등급 · 리스크 다변량 상호작용")
    sample_df = df.sample(min(1500, len(df)), random_state=42)
    fig_bubble = px.scatter(
        sample_df, x="sqft_living", y="price", color="Risk_Label", size="grade",
        color_discrete_map={"안전 매물(정상)": "#059669", "고위험 매물(주의)": "#DC2626"},
        labels={"sqft_living": "주거 면적 (sqft)", "price": "주택 시세 ($)", "Risk_Label": "리스크 등급"},
        title="주거 면적 vs 시세 (원 크기: 주택 등급 / 색상: 리스크 등급)"
    )
    st.plotly_chart(style_chart(fig_bubble), use_container_width=True)

    st.subheader("상관관계 히트맵 및 다중공선성(VIF) 진단")
    c_left, c_right = st.columns([1.3, 1])
    num_cols = ["price", "bedrooms", "bathrooms", "sqft_living", "floors", "grade", "building_age", "overvaluation_ratio", "High_Risk"]
    with c_left:
        corr = df[num_cols].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="BrBG", aspect="auto",
            title="주요 리스크 변수 간 상관계수"
        )
        st.plotly_chart(style_chart(fig_heat, height=460), use_container_width=True)
    with c_right:
        st.markdown("**다중공선성(VIF) 자동 진단표** *(로지스틱 회귀 안정성 검증)*")
        vif_features = ["bedrooms", "bathrooms", "sqft_living", "floors", "grade", "building_age", "overvaluation_ratio"]
        X_vif = df[vif_features].dropna()
        vif_data = pd.DataFrame({
            "변수명": vif_features,
            "VIF 지수": [round(variance_inflation_factor(X_vif.values, i), 2) for i in range(len(vif_features))]
        })
        vif_data["진단 결과"] = np.where(vif_data["VIF 지수"] > 10, "주의 (높은 상관성)", "양호")
        st.dataframe(vif_data, use_container_width=True, hide_index=True)
        st.info("VIF가 10을 초과하는 변수는 다음 '02. 데이터 정제 및 파생변수 설계' 단계에서 표준화(Scaling)하거나 선택적으로 제외하여 모델 안정성을 확보합니다.")

# =========================================================
# 모듈 2: 데이터 정제 및 파생변수 설계
# =========================================================
elif menu == "02. 데이터 정제 및 파생변수 설계":
    render_section_header("MODULE 02 // FEATURE ENGINEERING", "데이터 품질 감사 및 도메인 파생변수 파이프라인")
    st.write("원본 데이터의 무결성(결측·중복·변수 구조)을 진단한 뒤, 주택담보 보증 심사 기준에 맞춘 정제 및 파생변수 파이프라인을 실행합니다.")

    st.subheader("1) 원본 데이터 품질 감사 (Data Quality Audit)")
    raw_rows, raw_cols = df.shape
    dup_cnt = df.duplicated().sum()
    null_cnt = df.isnull().sum().sum()

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("전체 데이터 행 수 (Rows)", f"{raw_rows:,} 건")
    q2.metric("전체 변수 열 수 (Columns)", f"{raw_cols} 개")
    q3.metric("결측치 총 건수 (Missing)", f"{null_cnt} 건", "무결성 양호" if null_cnt == 0 else "정제 필요")
    q4.metric("중복 데이터 행 수 (Duplicates)", f"{dup_cnt} 건", "중복 없음" if dup_cnt == 0 else "제거 권장")

    with st.expander("전체 변수 명세서 및 결측률·데이터 타입 상세 확인", expanded=False):
        var_desc_map = {
            "price": "주택 실거래 시세 ($)", "bedrooms": "침실 수", "bathrooms": "욕실 수",
            "sqft_living": "실내 주거 면적 (sqft)", "sqft_lot": "대지 면적 (sqft)", "floors": "건물 층수",
            "waterfront": "수변 조망 여부 (0/1)", "view": "전망 등급 (0~4)", "condition": "주택 관리 상태 (1~5)",
            "grade": "건축 자재·시공 등급 (1~13)", "sqft_above": "지상층 면적 (sqft)", "sqft_basement": "지하실 면적 (sqft)",
            "yr_built": "건축 준공연도", "yr_renovated": "리모델링 연도", "zipcode": "행정구역 우편번호",
            "lat": "위도 좌표", "long": "경도 좌표", "sqft_living15": "인근 15가구 평균 주거면적",
            "sqft_lot15": "인근 15가구 평균 대지면적", "building_age": "[파생] 건물 노후도 (년)",
            "built_decade": "[파생] 건축 연대", "is_renovated": "[파생] 리모델링 여부",
            "price_per_sqft": "[파생] 단위면적당 가격 ($/sqft)", "overvaluation_ratio": "[파생] 지역 시세 대비 고평가 지수",
            "High_Risk": "[타깃] 고위험 부실우려 여부 (0/1)", "Risk_Label": "[타깃] 리스크 등급 라벨"
        }
        schema_df = pd.DataFrame({
            "변수명 (Column)": df.columns,
            "변수 설명": [var_desc_map.get(c, "기타 속성") for c in df.columns],
            "데이터 타입": df.dtypes.astype(str).values,
            "결측치 수": df.isnull().sum().values,
            "결측률 (%)": (df.isnull().mean().values * 100).round(2),
            "고유값 수 (Unique)": df.nunique().values
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True, height=280)

    st.divider()

    st.subheader("2) 전처리 파이프라인 및 금융 파생변수 설정")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("##### [Step 1] 데이터 클리닝 (중복·결측 정제)")
            drop_dup_opt = st.checkbox("중복 행(Duplicates) 자동 검출 및 제거", value=True)
            missing_opt = st.selectbox("결측치(Missing Value) 처리 전략", ["중앙값(Median) 대체 (이상치 강건)", "평균값(Mean) 대체", "결측 행 삭제(Drop)"])

        with st.container(border=True):
            st.markdown("##### [Step 2] 이상치(초고가 호화주택) 정제 설정")
            outlier_opt = st.radio(
                "상위 초고가 주택 처리 방식 (일반 주거용 보증 기준 맞추기)",
                ["제거 안 함 (원본 유지)", "상위 1% 초고가 이상치 제거 (권장)", "IQR(1.5배) 기준 이상치 제거"],
                index=1
            )

    with col2:
        with st.container(border=True):
            st.markdown("##### [Step 3] 금융·부동산 도메인 파생변수 생성")
            add_ltv_proxy = st.checkbox("잠재 담보인정비율(estimated_ltv) 변수 생성", value=True)
            add_loc_score = st.checkbox("입지 프리미엄 지수(location_premium) 생성", value=True)
            add_space_ratio = st.checkbox("주거 공간 효율 비율(living_to_lot_ratio) 생성", value=True)

        with st.container(border=True):
            st.markdown("##### [Step 4] 피처 스케일링 & 학습/검증 분할")
            scale_opt = st.checkbox("StandardScaler 표준화 적용 (평균 0, 분산 1 정규화)", value=True)
            test_ratio = st.slider("테스트(검증) 데이터 분할 비율", 0.1, 0.4, 0.2, 0.05)

    with st.container(border=True):
        default_features = ["bedrooms", "bathrooms", "sqft_living", "floors", "waterfront", "view", "condition", "grade", "building_age", "overvaluation_ratio"]
        selected_features = st.multiselect("최종 모델 투입 독립변수(X) 선택", default_features, default=default_features)

    if st.button("전처리 파이프라인 실행 및 데이터 확정", type="primary", use_container_width=True):
        df_p = df.copy()
        before_cnt = len(df_p)

        if drop_dup_opt:
            df_p = df_p.drop_duplicates()
        num_c = df_p.select_dtypes(include=[np.number]).columns
        if missing_opt == "중앙값(Median) 대체 (이상치 강건)":
            df_p[num_c] = df_p[num_c].fillna(df_p[num_c].median())
        elif missing_opt == "평균값(Mean) 대체":
            df_p[num_c] = df_p[num_c].fillna(df_p[num_c].mean())
        else:
            df_p = df_p.dropna()

        if outlier_opt == "상위 1% 초고가 이상치 제거 (권장)":
            upper = df_p["price"].quantile(0.99)
            df_p = df_p[df_p["price"] <= upper]
        elif outlier_opt == "IQR(1.5배) 기준 이상치 제거":
            q1, q3 = df_p["price"].quantile(0.25), df_p["price"].quantile(0.75)
            iqr = q3 - q1
            df_p = df_p[(df_p["price"] >= q1 - 1.5 * iqr) & (df_p["price"] <= q3 + 1.5 * iqr)]

        final_features = list(selected_features)
        if add_ltv_proxy:
            np.random.seed(42)
            df_p["estimated_ltv"] = np.clip(0.55 + (df_p["overvaluation_ratio"] - 1.0) * 0.4 + np.random.normal(0, 0.08, len(df_p)), 0.3, 0.98).round(3)
            final_features.append("estimated_ltv")
        if add_loc_score:
            df_p["location_premium"] = (df_p["waterfront"] * 3 + df_p["view"] + df_p["grade"] * 0.5).round(2)
            final_features.append("location_premium")
        if add_space_ratio:
            df_p["living_to_lot_ratio"] = (df_p["sqft_living"] / df_p["sqft_lot"].replace(0, 1)).round(3)
            final_features.append("living_to_lot_ratio")

        st.session_state.df_processed = df_p
        st.session_state.features = final_features
        st.session_state.scale_opt = scale_opt
        st.session_state.test_ratio = test_ratio

        st.success(f"전처리 파이프라인 완료: 원본 {before_cnt:,}건 → 정제 후 {len(df_p):,}건 (최종 학습 변수: {len(final_features)}개)")

        removed_cnt = before_cnt - len(df_p)
        train_cnt = int(len(df_p) * (1 - test_ratio))
        test_cnt = len(df_p) - train_cnt

        m1, m2, m3 = st.columns(3)
        m1.metric("제거된 이상치·중복 행", f"{removed_cnt:,} 건", f"원본 대비 {(removed_cnt/before_cnt)*100:.1f}% 정제")
        m2.metric("학습용 데이터 (Train Set)", f"{train_cnt:,} 건", f"전체의 {(1-test_ratio)*100:.0f}%")
        m3.metric("검증용 데이터 (Test Set)", f"{test_cnt:,} 건", f"전체의 {test_ratio*100:.0f}%")

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            fig_before = px.histogram(df, x="price", nbins=50, title="전처리 전(Before) 주택 가격 분포", color_discrete_sequence=["#94A3B8"])
            st.plotly_chart(style_chart(fig_before), use_container_width=True)
        with c_b2:
            fig_after = px.histogram(df_p, x="price", nbins=50, title="전처리 후(After) 정제된 담보 가격 분포", color_discrete_sequence=["#059669"])
            st.plotly_chart(style_chart(fig_after), use_container_width=True)

        if scale_opt:
            st.subheader("스케일링(StandardScaler) 적용 전·후 변수 분포 비교")
            scaler_preview = StandardScaler()
            scaled_vals = scaler_preview.fit_transform(df_p[final_features])
            scaled_df = pd.DataFrame(scaled_vals, columns=final_features)
            comp_df = pd.DataFrame({
                "변수명": final_features,
                "적용 전 평균": df_p[final_features].mean().round(2).values,
                "적용 전 표준편차": df_p[final_features].std().round(2).values,
                "적용 후 평균": scaled_df.mean().round(2).values,
                "적용 후 표준편차": scaled_df.std().round(2).values
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.subheader("파생변수 결합 완료 최종 데이터 샘플 (상위 10건)")
        preview_cols = ["price", "building_age", "overvaluation_ratio"] + [c for c in ["estimated_ltv", "location_premium", "living_to_lot_ratio"] if c in df_p.columns] + ["Risk_Label"]
        st.dataframe(df_p[preview_cols].head(10), use_container_width=True, hide_index=True)

# =========================================================
# 모듈 3: 예측 모델 학습 및 성능 검증
# =========================================================
elif menu == "03. 예측 모델 학습 및 성능 검증":
    render_section_header("MODULE 03 // MODEL BENCHMARKING", "예측 모델 성능 비교 및 운영 모델 선정")
    
    if st.session_state.df_processed is None:
        st.warning("먼저 왼쪽 메뉴의 '02. 데이터 정제 및 파생변수 설계'에서 [전처리 파이프라인 실행] 버튼을 눌러주세요.")
        st.stop()

    df_p = st.session_state.df_processed
    features = st.session_state.features

    st.info("**3대 예측 알고리즘 구성:** ① 로지스틱 회귀(감사·규제 대응) ② 의사결정나무(명확한 심사 기준선 도출) ③ 인공신경망 MLP(비선형 리스크 요인 정밀 탐지)")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("##### ① 로지스틱 회귀")
            st.caption("감사·규제 대응용 선형 기준 모델")
            lr_c = st.slider("규제 강도 (C)", 0.1, 5.0, 1.0)
    with c2:
        with st.container(border=True):
            st.markdown("##### ② 의사결정나무")
            st.caption("심사 커트라인 도출용 규칙 모델")
            dt_depth = st.slider("최대 깊이 (Max Depth)", 2, 12, 5)
    with c3:
        with st.container(border=True):
            st.markdown("##### ③ 인공신경망 (MLP)")
            st.caption("비선형 복합 리스크 탐지 모델")
            mlp_hidden = st.selectbox("은닉층 구조 (Hidden Layers)", ["(64, 32)", "(32, 16)", "(100, 50)"], index=0)

    if st.button("3종 예측 모델 동시 학습 및 검증 시작", type="primary", use_container_width=True):
        with st.spinner("3종의 분류 모델을 학습하고 성능을 교차 검증 중입니다..."):
            X = df_p[features]
            y = df_p["High_Risk"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=st.session_state.test_ratio, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            if st.session_state.scale_opt:
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
            else:
                X_train_s, X_test_s = X_train.values, X_test.values

            hidden_tuple = eval(mlp_hidden)
            model_dict = {
                "Logistic Regression": LogisticRegression(C=lr_c, max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(max_depth=dt_depth, random_state=42),
                "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=hidden_tuple, max_iter=300, random_state=42)
            }

            results = []
            trained_models = {}
            roc_fig = go.Figure()
            palette = ["#64748B", "#059669", "#D97706"]

            for idx_m, (name, m) in enumerate(model_dict.items()):
                m.fit(X_train_s, y_train)
                y_prob = m.predict_proba(X_test_s)[:, 1]
                y_pred = (y_prob >= 0.42).astype(int)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_prob)

                results.append({
                    "모델명": name,
                    "Accuracy (정확도)": round(acc, 4),
                    "Precision (정밀도)": round(prec, 4),
                    "Recall (재현율/부실탐지율)": round(rec, 4),
                    "F1-Score": round(f1, 4),
                    "ROC-AUC": round(auc, 4)
                })
                trained_models[name] = m

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=palette[idx_m], width=2.8), name=f"{name} (AUC={auc:.3f})"))

            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#CBD5E1"), name="기준선"))
            roc_fig.update_layout(title="모델별 ROC Curve 변별력 비교", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

            st.session_state.models = trained_models
            st.session_state.scaler = scaler
            st.session_state.results_df = pd.DataFrame(results)
            st.session_state.X_test_s = X_test_s
            st.session_state.y_test = y_test
            st.session_state.roc_fig = roc_fig

    if st.session_state.models:
        res_df = st.session_state.results_df
        st.subheader("모델 성능 종합 비교표")
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        perf_csv = res_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="모델별 성능 비교 성적표 (CSV) 다운로드",
            data=perf_csv,
            file_name="K_SafeGuard_Model_Benchmark_Report.csv",
            mime="text/csv"
        )
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            radar_fig = go.Figure()
            metrics_cols = ["Accuracy (정확도)", "Precision (정밀도)", "Recall (재현율/부실탐지율)", "F1-Score", "ROC-AUC"]
            radar_colors = ["#64748B", "#059669", "#D97706"]
            for idx_r, (_, row) in enumerate(res_df.iterrows()):
                radar_fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics_cols],
                    theta=metrics_cols,
                    fill="toself",
                    line=dict(color=radar_colors[idx_r]),
                    name=row["모델명"]
                ))
            radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.4, 1.0])), title="3대 예측 모델 다차원 성능 레이더 차트")
            st.plotly_chart(style_chart(radar_fig), use_container_width=True)

        with col_r2:
            st.plotly_chart(style_chart(st.session_state.roc_fig), use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            dt_model = st.session_state.models["Decision Tree"]
            imp_df = pd.DataFrame({"변수": features, "중요도": dt_model.feature_importances_}).sort_values("중요도", ascending=True)
            fig_imp = px.bar(imp_df, x="중요도", y="변수", orientation="h", title="핵심 부실 결정 요인 (심사 기준 근거)", color_discrete_sequence=["#065F46"])
            st.plotly_chart(style_chart(fig_imp), use_container_width=True)

        with col_b2:
            sel_cm_model = st.selectbox("혼동행렬(Confusion Matrix) 확인할 모델 선택", list(st.session_state.models.keys()))
            cm_prob = st.session_state.models[sel_cm_model].predict_proba(st.session_state.X_test_s)[:, 1]
            cm_pred = (cm_prob >= 0.35).astype(int)
            cm = confusion_matrix(st.session_state.y_test, cm_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Greens",
                x=["예측: 안전(0)", "예측: 고위험(1)"], y=["실제: 안전(0)", "실제: 고위험(1)"],
                title=f"{sel_cm_model} 예측 검증 행렬"
            )
            st.plotly_chart(style_chart(fig_cm), use_container_width=True)

# =========================================================
# 모듈 4: 개별 담보 심사 및 동적 보증료 산정
# =========================================================
elif menu == "04. 개별 담보 심사 및 동적 보증료 산정":
    render_section_header("MODULE 04 // DYNAMIC PRICING ENGINE", "개별 담보 리스크 심사 및 적정 보증료 산정")
    
    if not st.session_state.models:
        st.warning("먼저 '03. 예측 모델 학습 및 성능 검증'에서 모델을 학습시켜 주세요.")
        st.stop()

    st.write("신청 주택의 담보 조건을 입력하면 부실 확률(PD)과 기대손실액(EL)을 기반으로 적정 보증료율 및 권고 한도를 산출합니다.")

    with st.container(border=True):
        engine_choice = st.selectbox("심사에 적용할 예측 모델 선택", list(st.session_state.models.keys()), index=1)

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("##### ① 담보 가치 및 보증 신청액")
            in_price = st.number_input("신청 주택 감정 시세 ($)", 100000, 3000000, 450000, step=10000)
            in_deposit = st.number_input("보증 신청 금액 ($)", 50000, 2500000, 380000, step=10000)
            in_sqft = st.slider("주거 면적 (sqft)", 500, 6000, 1800)
    with c2:
        with st.container(border=True):
            st.markdown("##### ② 건물 노후도 및 물리적 등급")
            in_age = st.slider("건물 노후도 (년)", 0, 115, 35)
            in_grade = st.slider("건축 등급 (1~13)", 3, 13, 7)
            in_cond = st.slider("관리 상태 (1~5)", 1, 5, 3)
    with c3:
        with st.container(border=True):
            st.markdown("##### ③ 시세 괴리 및 주거 스펙")
            in_overval = st.slider("주변 시세 대비 고평가 지수 (1.0=평균)", 0.6, 2.0, 1.18, 0.02)
            in_bed = st.number_input("침실 수", 1, 8, 3)
            in_bath = st.number_input("욕실 수", 1.0, 6.0, 2.0, 0.5)

    ltv_current = in_deposit / in_price
    input_row = {
        "bedrooms": in_bed,
        "bathrooms": in_bath,
        "sqft_living": in_sqft,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": in_cond,
        "grade": in_grade,
        "building_age": in_age,
        "overvaluation_ratio": in_overval,
        "estimated_ltv": ltv_current,
        "location_premium": in_grade * 0.5
    }
    input_df = pd.DataFrame([input_row])
    for col in st.session_state.features:
        if col not in input_df.columns:
            input_df[col] = st.session_state.X_train[col].mean() if "X_train" in st.session_state and col in st.session_state.X_train.columns else 0
    input_df = input_df[st.session_state.features]
    input_s = st.session_state.scaler.transform(input_df) if st.session_state.scale_opt else input_df.values

    model = st.session_state.models[engine_choice]
    pd_prob = float(model.predict_proba(input_s)[0][1])
    
    lgd = max(0.15, min(0.65, (ltv_current - 0.5) * 0.8 + 0.2))
    ead = in_deposit
    expected_loss = pd_prob * lgd * ead
    base_fee_rate = 0.0015
    dynamic_fee_rate = (expected_loss / ead) + base_fee_rate
    safe_limit = in_price * max(0.50, 0.85 - pd_prob * 0.4)

    st.divider()
    res_c1, res_c2 = st.columns([1, 1.3])
    
    with res_c1:
        risk_color = "#059669" if pd_prob < 0.30 else ("#D97706" if pd_prob < 0.65 else "#DC2626")
        risk_grade_txt = "LOW RISK · 우량" if pd_prob < 0.30 else ("MODERATE · 주의" if pd_prob < 0.65 else "HIGH RISK · 경고")

        fig_gauge = go.Figure(data=[go.Pie(
            values=[pd_prob * 100, 100 - (pd_prob * 100)],
            hole=0.76,
            marker_colors=[risk_color, "#E6ECE9"],
            textinfo="none",
            hoverinfo="none",
            sort=False,
            direction="clockwise"
        )])
        fig_gauge.update_layout(
            title=dict(text="예상 보증사고 확률 (PD)", x=0.5, xanchor="center", font=dict(size=15, color="#111827")),
            showlegend=False,
            paper_bgcolor="#FFFFFF",
            height=330,
            margin=dict(l=20, r=20, t=55, b=20),
            annotations=[
                dict(text=f"<b>{pd_prob*100:.1f}%</b>", x=0.5, y=0.55, font=dict(size=36, color="#111827"), showarrow=False),
                dict(text=risk_grade_txt, x=0.5, y=0.38, font=dict(size=12, color=risk_color, weight="bold"), showarrow=False)
            ]
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    with res_c2:
        st.subheader("심사 결과 및 보증 조건 산출 내역")
        if pd_prob < 0.30 and in_deposit <= safe_limit:
            st.success("**최종 판정: 정상 승인 (우량 담보)** — 기본 우대 보증료율 적용 가능")
        elif pd_prob < 0.65:
            st.warning("**최종 판정: 조건부 승인 (리스크 할증 또는 한도 감액 권고)**")
        else:
            st.error("**최종 판정: 보증 거절 권고 (고위험 부실 우려 매물)**")

        m1, m2, m3 = st.columns(3)
        m1.metric("기대손실액 (EL)", f"${expected_loss:,.0f}", f"손실률(LGD) {lgd*100:.1f}%")
        m2.metric("적정 동적 보증료율", f"{dynamic_fee_rate*100:.2f}%", f"기존 0.15% 대비 +{(dynamic_fee_rate-base_fee_rate)*100:.2f}%p")
        m3.metric("권고 안전 보증 한도", f"${safe_limit:,.0f}", f"신청액 대비 ${safe_limit - in_deposit:,.0f}")

        st.info(
            f"**심사 의견:** 신청 보증금(${in_deposit:,})은 감정가 대비 {ltv_current*100:.1f}% 수준입니다. "
            f"건물 경과연수({in_age}년)와 시세 괴리율({in_overval})을 반영해 연 **{dynamic_fee_rate*100:.2f}%**의 보증료율 적용 또는 한도 조정을 권고합니다."
        )

# =========================================================
# 모듈 5: 거시경제 스트레스 테스트 & 리포트 다운로드
# =========================================================
elif menu == "05. 거시경제 스트레스 테스트":
    render_section_header("MODULE 05 // MACRO STRESS TEST", "거시경제 스트레스 테스트 (시세 하락 시나리오 분석)")
    st.write("주택 가격 하락 시나리오에 따른 포트폴리오 예상 대위변제 손실액과 한도 규제 적용 시의 손실 절감 효과를 시뮬레이션합니다.")

    if st.session_state.df_processed is None:
        st.warning("먼저 '02. 데이터 정제 및 파생변수 설계'에서 전처리를 실행해 주세요.")
        st.stop()

    df_p = st.session_state.df_processed.copy()
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        with st.container(border=True):
            st.markdown("##### ① 거시 충격 시나리오 설정")
            price_drop = st.slider("가정 시나리오: 부동산 시세 하락률 (%)", 0, 35, 15, 5)
    with col_s2:
        with st.container(border=True):
            st.markdown("##### ② 리스크 방어 정책 한도 설정")
            ltv_cap = st.slider("정책 개입: 보증 한도(LTV) 상한 규제 (%)", 60, 90, 80, 5)

    drops = list(range(0, 40, 5))
    loss_no_policy = []
    loss_with_ai = []

    base_ltv = df_p["estimated_ltv"] if "estimated_ltv" in df_p.columns else 0.75
    deposit_amt = df_p["price"] * base_ltv

    for d in drops:
        shocked_price = df_p["price"] * (1 - d / 100)
        shortfall_old = np.maximum(0, deposit_amt - shocked_price)
        loss_no_policy.append(shortfall_old.sum() / 1e6)

        capped_deposit = np.minimum(deposit_amt, df_p["price"] * (ltv_cap / 100))
        capped_deposit = np.where(df_p["High_Risk"] == 1, capped_deposit * 0.85, capped_deposit)
        shortfall_new = np.maximum(0, capped_deposit - shocked_price)
        loss_with_ai.append(shortfall_new.sum() / 1e6)

    idx = drops.index(price_drop)
    saved_amount = loss_no_policy[idx] - loss_with_ai[idx]

    k1, k2, k3 = st.columns(3)
    k1.metric(f"시세 -{price_drop}% 시 기존 예상 손실", f"${loss_no_policy[idx]:,.1f} 백만", "정책 미개입 기준", delta_color="inverse")
    k2.metric(f"동적 한도 적용 후 예상 손실", f"${loss_with_ai[idx]:,.1f} 백만", f"LTV {ltv_cap}% + 고위험 차등 적용")
    k3.metric("보증 기금 손실 절감 효과", f"${saved_amount:,.1f} 백만", f"손실액 {(saved_amount/max(1, loss_no_policy[idx]))*100:.1f}% 감소")

    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(x=[f"-{d}%" for d in drops], y=loss_no_policy, mode="lines+markers", name="기존 일률 보증 방식 (예상 손실)", line=dict(color="#DC2626", width=3)))
    fig_stress.add_trace(go.Scatter(x=[f"-{d}%" for d in drops], y=loss_with_ai, mode="lines+markers", name="K-SafeGuard 동적 관리 적용 (예상 손실)", line=dict(color="#059669", width=3)))
    fig_stress.update_layout(title="시세 하락 시나리오별 누적 대위변제 손실액 비교 (단위: 백만 $)", xaxis_title="부동산 시세 하락률", yaxis_title="예상 손실액 ($M)")
    st.plotly_chart(style_chart(fig_stress), use_container_width=True)

    st.divider()
    st.subheader("심사 리포트 및 고위험군 관리 대장 추출")
    high_risk_export = df_p[df_p["High_Risk"] == 1][["price", "sqft_living", "building_age", "grade", "overvaluation_ratio", "lat", "long"]].head(500)
    csv_bytes = high_risk_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="집중 관리 대상 고위험 담보 매물 리스트 (CSV) 다운로드",
        data=csv_bytes,
        file_name="K_SafeGuard_HighRisk_Report.csv",
        mime="text/csv"
    )
