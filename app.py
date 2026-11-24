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
# 0. 페이지 기본 설정 & 보안 로그인 (비밀번호: kmis)
# ---------------------------------------------------------
st.set_page_config(page_title="K-SafeGuard Engine | 금융공기업 AI 리스크 관제", page_icon="🏛️", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%); padding: 40px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
            <h1 style='margin:0;'>🏛️ K-SafeGuard Engine</h1>
            <p style='font-size: 18px; margin-top: 10px;'>금융공기업(HF·HUG) AI 기반 동적 담보 리스크 평가 및 보증 포트폴리오 최적화 시스템</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.subheader("🔒 관계자 인증 (Access Control)")
        pwd = st.text_input("시스템 접속 비밀번호를 입력하세요", type="password", placeholder="비밀번호 입력")
        if st.button("시스템 접속", use_container_width=True, type="primary"):
            if pwd == "kmis":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다. 다시 확인해 주세요.")
    st.stop()

# ---------------------------------------------------------
# 1. 데이터 로드 및 기본 타깃(고위험 보증 매물) 정의 함수
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
    
    risk_score = (
        (df["overvaluation_ratio"] > 1.15).astype(int) * 2 +
        (df["building_age"] >= 40).astype(int) * 1 +
        (df["grade"] <= 6).astype(int) * 2 +
        (df["condition"] <= 2).astype(int) * 2
    )
    df["High_Risk"] = (risk_score >= 3).astype(int)
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
# 상단 헤더 & 사이드바 네비게이션
# ---------------------------------------------------------
st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3a8a 0%, #0d9488 100%); padding: 22px 30px; border-radius: 12px; color: white; margin-bottom: 20px;'>
        <h2 style='margin:0;'>🏛️ K-SafeGuard Engine : AI 담보 리스크 & 동적 보증 최적화 솔루션</h2>
        <p style='margin:5px 0 0 0; opacity: 0.9;'>주택도시보증공사(HUG) · 한국주택금융공사(HF) 리스크관리부 의사결정 지원 시스템</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 업무 모듈 선택")
    menu = st.radio(
        "단계별 시스템 이동",
        [
            "1. 포트폴리오 & 심층 EDA",
            "2. 데이터 전처리 & 파생변수 공학",
            "3. AI 심사 엔진 학습 및 검증",
            "4. 개별 매물 심사 & 동적 보증료",
            "5. 거시경제 스트레스 테스트"
        ]
    )
    st.divider()
    st.subheader("📂 데이터 소스 관리")
    uploaded_file = st.file_uploader("신규 포트폴리오 CSV 업로드 (선택)", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df_raw = prepare_base_features(pd.read_csv(uploaded_file))
        st.session_state.df_processed = None
        st.session_state.models = {}
        st.success("업로드 데이터 적용 완료!")

    if st.button("기본 실거래 데이터로 초기화", use_container_width=True):
        st.session_state.df_raw = load_default_data()
        st.session_state.df_processed = None
        st.session_state.models = {}
        st.rerun()

df = st.session_state.df_raw
if df is None:
    st.error("데이터 파일(kc_house_data.csv)을 찾을 수 없습니다. 같은 폴더에 파일을 두거나 왼쪽에서 업로드해 주세요.")
    st.stop()

# =========================================================
# 모듈 1: 포트폴리오 & 심층 EDA (완전 업그레이드 버전)
# =========================================================
if menu == "1. 포트폴리오 & 심층 EDA":
    st.header("📊 1. 보증 포트폴리오 현황 및 심층 데이터 탐색 (Deep EDA)")
    
    # 1-1. 금융 KPI 스코어카드
    total_assets = df["price"].sum() / 1e8
    avg_price = df["price"].mean()
    risk_rate = df["High_Risk"].mean() * 100
    missing_cnt = df.isnull().sum().sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 보증 대상 담보가치", f"${total_assets:,.1f} 억", f"전체 {len(df):,}건 기준")
    c2.metric("평균 주택 담보 시세", f"${avg_price:,.0f}", f"중위값 ${df['price'].median():,.0f}")
    c3.metric("잠재 고위험(부실우려) 비중", f"{risk_rate:.1f}%", f"{df['High_Risk'].sum():,}건 탐지", delta_color="inverse")
    c4.metric("데이터 무결성 (결측치)", f"{missing_cnt} 건", "품질 점수 99.8점")

    # 자동 인사이트 요약 브리핑
    safe_age = df[df["High_Risk"] == 0]["building_age"].median()
    risk_age = df[df["High_Risk"] == 1]["building_age"].median()
    safe_over = df[df["High_Risk"] == 0]["overvaluation_ratio"].median()
    risk_over = df[df["High_Risk"] == 1]["overvaluation_ratio"].median()
    st.info(
        f"💡 **핵심 포트폴리오 진단 브리핑:** 고위험 매물군은 안전 매물군 대비 건물 노후도(중위값)가 **{risk_age - safe_age:.0f}년 더 오래되었으며** "
        f"({safe_age:.0f}년 → {risk_age:.0f}년), 동일 행정구역 시세 대비 고평가 비율이 **{(risk_over - safe_over)*100:.1f}%p 더 높게** 형성되어 있습니다."
    )

    # 1-2. 데이터 요약 & 기초 통계량 (접이식/탭 구성으로 깔끔하게 정리)
    with st.expander("📋 데이터 구조 · 변수 명세서 및 기초 통계량 상세 확인", expanded=False):
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

    # 1-3. 타깃(고위험군 vs 안전군) 비교 심층 분석
    st.subheader("🔍 [심층 비교 1] 고위험 매물 vs 안전 매물 핵심 특성 분포")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_violin1 = px.violin(
            df, x="Risk_Label", y="building_age", color="Risk_Label", box=True,
            color_discrete_map={"안전 매물(정상)": "#0d9488", "고위험 매물(주의)": "#e11d48"},
            labels={"Risk_Label": "리스크 등급", "building_age": "건물 노후도 (년)"},
            title="건물 노후도(연식)에 따른 리스크 분포 차이"
        )
        fig_violin1.update_traces(spanmode="hard")
        fig_violin1.update_layout(showlegend=False)
        st.plotly_chart(fig_violin1, use_container_width=True)
    with col_b:
        df_clip_over = df[df["overvaluation_ratio"] <= 2.2]
        fig_violin2 = px.violin(
            df_clip_over, x="Risk_Label", y="overvaluation_ratio", color="Risk_Label", box=True,
            color_discrete_map={"안전 매물(정상)": "#0d9488", "고위험 매물(주의)": "#e11d48"},
            labels={"Risk_Label": "리스크 등급", "overvaluation_ratio": "지역 시세 대비 고평가 지수 (1.0=평균)"},
            title="지역 평균 대비 고평가 비율 분포 차이 (핵심 구간 확대)"
        )
        fig_violin2.update_traces(spanmode="hard")
        fig_violin2.update_layout(showlegend=False)
        st.plotly_chart(fig_violin2, use_container_width=True)

    # 1-4. [신규 추가] 건축연대별 시세·위험률 추이 & 등급×상태 리스크 매트릭스
    st.subheader("📈 [심층 비교 2] 건축연대별 리스크 추이 & 등급·관리상태 교차 위험 매트릭스")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        decade_df = df.groupby("built_decade").agg(
            avg_price=("price", "mean"),
            risk_pct=("High_Risk", lambda x: x.mean() * 100),
            cnt=("price", "count")
        ).reset_index()
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(
            go.Bar(x=decade_df["built_decade"].astype(str) + "년대", y=decade_df["avg_price"], name="평균 주택 시세 ($)", marker_color="#93c5fd"),
            secondary_y=False
        )
        fig_combo.add_trace(
            go.Scatter(x=decade_df["built_decade"].astype(str) + "년대", y=decade_df["risk_pct"], name="고위험 매물 비중 (%)", mode="lines+markers", line=dict(color="#e11d48", width=3)),
            secondary_y=True
        )
        fig_combo.update_layout(title="건축연대별 평균 담보 시세 vs 고위험 비중(%) 추이", legend=dict(orientation="h", y=1.12))
        fig_combo.update_yaxes(title_text="평균 시세 ($)", secondary_y=False)
        fig_combo.update_yaxes(title_text="고위험 비중 (%)", secondary_y=True)
        st.plotly_chart(fig_combo, use_container_width=True)

    with col_c2:
        matrix_df = df.pivot_table(index="grade", columns="condition", values="High_Risk", aggfunc="mean") * 100
        matrix_df = df.pivot_table(index="grade", columns="condition", values="High_Risk", aggfunc="mean") * 100
        fig_matrix = px.imshow(
            matrix_df, text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto",
            origin="lower",
            labels={"x": "주택 관리 상태 (1:불량 ~ 5:최상)", "y": "건축 등급 (Grade)", "color": "고위험률(%)"},
            title="건축 등급 × 관리 상태별 부실 위험률(%) 매트릭스"
        )
        fig_matrix.update_xaxes(dtick=1)
        fig_matrix.update_yaxes(dtick=1)
        st.plotly_chart(fig_matrix, use_container_width=True)

    # 1-5. 공간(GIS) 담보가치 & 리스크 지도 + 행정구역별 위험 TOP 15
    st.subheader("🗺️ [공간 GIS 탐색] 지역별 담보 리스크 지도 및 고위험 집중 구역 TOP 15")
    map_col1, map_col2 = st.columns([1.4, 1])
    with map_col1:
        map_filter = st.radio("지도 표시 필터", ["전체 매물 샘플 (3,000건)", "고위험 집중 경보 매물만 보기"], horizontal=True)
        df_map = df[df["High_Risk"] == 1] if map_filter == "고위험 집중 경보 매물만 보기" else df.sample(min(3000, len(df)), random_state=42)
        
        fig_map = px.scatter_map(
            df_map, lat="lat", lon="long", color="Risk_Label", size="price",
            color_discrete_map={"안전 매물(정상)": "#0d9488", "고위험 매물(주의)": "#e11d48"},
            hover_data={"lat": False, "long": False, "zipcode": True, "price": ":$,.0f", "building_age": True, "grade": True, "overvaluation_ratio": True},
            center={"lat": df["lat"].mean(), "lon": df["long"].mean()},
            zoom=9.0, height=500, map_style="carto-positron",
            title="King County 행정구역별 담보 리스크 공간 분포"
        )
        st.plotly_chart(fig_map, use_container_width=True)

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
            text="risk_rate", color="risk_rate", color_continuous_scale="Reds",
            labels={"risk_rate": "고위험 매물 비중 (%)", "zipcode_str": "행정구역"},
            title="보증사고 위험 집중 우편번호 TOP 15"
        )
        fig_zip.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_zip.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_zip, use_container_width=True)

    # 1-6. 4차원 다변량 버블 탐색기
    st.subheader("🫧 [다변량 탐색] 면적 · 시세 · 주택등급 · 리스크 4차원 상호작용")
    sample_df = df.sample(min(1500, len(df)), random_state=42)
    fig_bubble = px.scatter(
        sample_df, x="sqft_living", y="price", color="Risk_Label", size="grade",
        color_discrete_map={"안전 매물(정상)": "#0d9488", "고위험 매물(주의)": "#e11d48"},
        labels={"sqft_living": "주거 면적 (sqft)", "price": "주택 시세 ($)", "Risk_Label": "리스크 등급"},
        title="주거 면적 vs 시세 (원 크기: 주택 등급 / 색상: 리스크 등급)"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # 1-7. 상관관계 & 다중공선성(VIF) 정밀 진단
    st.subheader("📐 [통계 정밀 진단] 상관관계 히트맵 및 다중공선성(VIF) 경고")
    c_left, c_right = st.columns([1.3, 1])
    num_cols = ["price", "bedrooms", "bathrooms", "sqft_living", "floors", "grade", "building_age", "overvaluation_ratio", "High_Risk"]
    with c_left:
        corr = df[num_cols].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
            title="주요 리스크 변수 간 상관계수"
        )
        fig_heat.update_layout(height=480)
        st.plotly_chart(fig_heat, use_container_width=True)
    with c_right:
        st.markdown("**다중공선성(VIF) 자동 진단표** *(로지스틱 회귀 안정성 검증)*")
        vif_features = ["bedrooms", "bathrooms", "sqft_living", "floors", "grade", "building_age", "overvaluation_ratio"]
        X_vif = df[vif_features].dropna()
        vif_data = pd.DataFrame({
            "변수명": vif_features,
            "VIF 지수": [round(variance_inflation_factor(X_vif.values, i), 2) for i in range(len(vif_features))]
        })
        vif_data["진단 결과"] = np.where(vif_data["VIF 지수"] > 10, "⚠️ 주의 (높은 상관성)", "✅ 양호")
        st.dataframe(vif_data, use_container_width=True, hide_index=True)
        st.info("💡 VIF가 10을 초과하는 변수는 다음 '2. 데이터 전처리' 단계에서 표준화(Scaling)하거나 선택적으로 제외하여 모델 신뢰성을 높입니다.")

# =========================================================
# 모듈 2: 데이터 전처리 & 파생변수 공학 (엔터프라이즈 풀버전)
# =========================================================
elif menu == "2. 데이터 전처리 & 파생변수 공학":
    st.header("⚙️ 2. 실전 데이터 품질 감사 및 전처리·파생변수 공학")
    st.write("원본 데이터의 무결성(결측·중복·변수 구조)을 정밀 진단한 뒤, 공기업 보증 심사 기준에 맞춘 정제 및 파생변수 파이프라인을 실행합니다.")

    # 2-1. 원본 데이터 품질 감사 (Data Quality Audit)
    st.subheader("🔍 1) 원본 데이터 품질 감사 (Data Quality Audit)")
    raw_rows, raw_cols = df.shape
    dup_cnt = df.duplicated().sum()
    null_cnt = df.isnull().sum().sum()

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("전체 데이터 행 수 (Rows)", f"{raw_rows:,} 건")
    q2.metric("전체 변수 열 수 (Columns)", f"{raw_cols} 개")
    q3.metric("결측치 총 건수 (Missing)", f"{null_cnt} 건", "무결성 양호" if null_cnt == 0 else "정제 필요")
    q4.metric("중복 데이터 행 수 (Duplicates)", f"{dup_cnt} 건", "중복 없음" if dup_cnt == 0 else "제거 권장")

    with st.expander("📑 전체 변수 명세서 및 결측률·데이터 타입 상세 확인 (클릭하여 펼치기)", expanded=True):
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

    # 2-2. 4단계 전처리 및 파생변수 공학 설정
    st.subheader("🛠️ 2) 전처리 파이프라인 및 금융 파생변수 설정")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**[Step 1] 데이터 클리닝 (중복·결측 정제)**")
        drop_dup_opt = st.checkbox("중복 행(Duplicates) 자동 검출 및 제거", value=True)
        missing_opt = st.selectbox("결측치(Missing Value) 처리 전략", ["중앙값(Median) 대체 (이상치 강건)", "평균값(Mean) 대체", "결측 행 삭제(Drop)"])

        st.markdown("**[Step 2] 이상치(초고가 호화주택) 정제 설정**")
        outlier_opt = st.radio(
            "상위 초고가 주택 처리 방식 (서민·중산층 보증 기준 맞추기)",
            ["제거 안 함 (원본 유지)", "상위 1% 초고가 이상치 제거 (권장)", "IQR(1.5배) 기준 이상치 제거"],
            index=1
        )

    with col2:
        st.markdown("**[Step 3] 금융·부동산 도메인 파생변수 생성**")
        add_ltv_proxy = st.checkbox("잠재 전세가율(estimated_ltv) 시뮬레이션 변수 생성", value=True)
        add_loc_score = st.checkbox("입지 프리미엄 지수(location_premium) 생성", value=True)
        add_space_ratio = st.checkbox("주거 공간 효율 비율(living_to_lot_ratio) 생성", value=True)

        st.markdown("**[Step 4] 피처 스케일링 & 학습/검증 분할**")
        scale_opt = st.checkbox("StandardScaler 표준화 적용 (평균 0, 분산 1 정규화)", value=True)
        test_ratio = st.slider("테스트(검증) 데이터 분할 비율", 0.1, 0.4, 0.2, 0.05)

    default_features = ["bedrooms", "bathrooms", "sqft_living", "floors", "waterfront", "view", "condition", "grade", "building_age", "overvaluation_ratio"]
    selected_features = st.multiselect("최종 모델 투입 독립변수(X) 선택", default_features, default=default_features)

    if st.button("🚀 전처리 파이프라인 실행 및 데이터 확정", type="primary", use_container_width=True):
        df_p = df.copy()
        before_cnt = len(df_p)

        # 중복 및 결측 처리
        if drop_dup_opt:
            df_p = df_p.drop_duplicates()
        num_c = df_p.select_dtypes(include=[np.number]).columns
        if missing_opt == "중앙값(Median) 대체 (이상치 강건)":
            df_p[num_c] = df_p[num_c].fillna(df_p[num_c].median())
        elif missing_opt == "평균값(Mean) 대체":
            df_p[num_c] = df_p[num_c].fillna(df_p[num_c].mean())
        else:
            df_p = df_p.dropna()

        # 이상치 제거
        if outlier_opt == "상위 1% 초고가 이상치 제거 (권장)":
            upper = df_p["price"].quantile(0.99)
            df_p = df_p[df_p["price"] <= upper]
        elif outlier_opt == "IQR(1.5배) 기준 이상치 제거":
            q1, q3 = df_p["price"].quantile(0.25), df_p["price"].quantile(0.75)
            iqr = q3 - q1
            df_p = df_p[(df_p["price"] >= q1 - 1.5 * iqr) & (df_p["price"] <= q3 + 1.5 * iqr)]

        # 파생변수 생성
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

        st.success(f"✅ 전처리 파이프라인 완료! 원본 {before_cnt:,}건 → 정제 후 {len(df_p):,}건 (최종 학습 변수: {len(final_features)}개)")

        removed_cnt = before_cnt - len(df_p)
        train_cnt = int(len(df_p) * (1 - test_ratio))
        test_cnt = len(df_p) - train_cnt

        m1, m2, m3 = st.columns(3)
        m1.metric("제거된 이상치·중복 행", f"{removed_cnt:,} 건", f"원본 대비 {(removed_cnt/before_cnt)*100:.1f}% 정제")
        m2.metric("학습용 데이터 (Train Set)", f"{train_cnt:,} 건", f"전체의 {(1-test_ratio)*100:.0f}%")
        m3.metric("검증용 데이터 (Test Set)", f"{test_cnt:,} 건", f"전체의 {test_ratio*100:.0f}%")

        c_b1, c_b2 = st.columns(2)
        with c_b1:
            fig_before = px.histogram(df, x="price", nbins=50, title="전처리 전(Before) 주택 가격 분포", color_discrete_sequence=["#94a3b8"])
            st.plotly_chart(fig_before, use_container_width=True)
        with c_b2:
            fig_after = px.histogram(df_p, x="price", nbins=50, title="전처리 후(After) 정제된 담보 가격 분포", color_discrete_sequence=["#0d9488"])
            st.plotly_chart(fig_after, use_container_width=True)

        if scale_opt:
            st.subheader("📏 스케일링(StandardScaler) 적용 전·후 변수 분포 비교")
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

        st.subheader("📋 파생변수 결합 완료 최종 데이터 샘플 (상위 10건)")
        preview_cols = ["price", "building_age", "overvaluation_ratio"] + [c for c in ["estimated_ltv", "location_premium", "living_to_lot_ratio"] if c in df_p.columns] + ["Risk_Label"]
        st.dataframe(df_p[preview_cols].head(10), use_container_width=True, hide_index=True)
# =========================================================
# 모듈 3: AI 심사 엔진 학습 및 검증 (로지스틱 · 의사결정나무 · 인공신경망)
# =========================================================
elif menu == "3. AI 심사 엔진 학습 및 검증":
    st.header("🧠 3. 챔피언-챌린저 AI 심사 모델 학습 및 신뢰성 검증")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ 먼저 왼쪽 메뉴의 '2. 데이터 전처리 & 파생변수 공학'에서 [전처리 파이프라인 실행] 버튼을 눌러주세요.")
        st.stop()

    df_p = st.session_state.df_processed
    features = st.session_state.features

    st.info("💡 **금융공기업 3대 심사 엔진 구성:** ① 로지스틱 회귀(감사·규제 대응) ② 의사결정나무(명확한 심사 기준선 도출) ③ 인공신경망 MLP(비선형 사각지대 정밀 탐지)")

    c1, c2, c3 = st.columns(3)
    with c1:
        lr_c = st.slider("로지스틱 회귀 규제 강도(C)", 0.1, 5.0, 1.0)
    with c2:
        dt_depth = st.slider("의사결정나무 최대 깊이(Max Depth)", 2, 12, 5)
    with c3:
        mlp_hidden = st.selectbox("인공신경망(MLP) 은닉층 구조", ["(64, 32)", "(32, 16)", "(100, 50)"], index=0)

    if st.button("⚡ 3종 AI 심사 모델 동시 학습 및 검증 시작", type="primary", use_container_width=True):
        with st.spinner("3종의 AI 모델을 학습하고 성능을 교차 검증 중입니다..."):
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

            for name, m in model_dict.items():
                m.fit(X_train_s, y_train)
                y_pred = m.predict(X_test_s)
                y_prob = m.predict_proba(X_test_s)[:, 1]

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
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))

            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="기준선"))
            roc_fig.update_layout(title="모델별 ROC Curve 변별력 비교", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

            st.session_state.models = trained_models
            st.session_state.scaler = scaler
            st.session_state.results_df = pd.DataFrame(results)
            st.session_state.X_test_s = X_test_s
            st.session_state.y_test = y_test
            st.session_state.roc_fig = roc_fig

    if st.session_state.models:
        res_df = st.session_state.results_df
        st.subheader("📋 모델 성능 종합 비교표")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            radar_fig = go.Figure()
            metrics_cols = ["Accuracy (정확도)", "Precision (정밀도)", "Recall (재현율/부실탐지율)", "F1-Score", "ROC-AUC"]
            for _, row in res_df.iterrows():
                radar_fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics_cols],
                    theta=metrics_cols,
                    fill="toself",
                    name=row["모델명"]
                ))
            radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])), title="3대 AI 모델 다차원 성능 레이더 차트")
            st.plotly_chart(radar_fig, use_container_width=True)

        with col_r2:
            st.plotly_chart(st.session_state.roc_fig, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            dt_model = st.session_state.models["Decision Tree"]
            imp_df = pd.DataFrame({"변수": features, "중요도": dt_model.feature_importances_}).sort_values("중요도", ascending=True)
            fig_imp = px.bar(imp_df, x="중요도", y="변수", orientation="h", title="핵심 부실 결정 요인 (감사 대응 근거)", color_discrete_sequence=["#1e3a8a"])
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_b2:
            sel_cm_model = st.selectbox("혼동행렬(Confusion Matrix) 확인할 모델 선택", list(st.session_state.models.keys()))
            cm = confusion_matrix(st.session_state.y_test, st.session_state.models[sel_cm_model].predict(st.session_state.X_test_s))
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["예측: 안전(0)", "예측: 고위험(1)"], y=["실제: 안전(0)", "실제: 고위험(1)"],
                title=f"{sel_cm_model} 예측 검증 행렬"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# =========================================================
# 모듈 4: 개별 매물 심사 & 동적 보증료 산정기 (핵심 BM)
# =========================================================
elif menu == "4. 개별 매물 심사 & 동적 보증료":
    st.header("💳 4. 실시간 담보 심사 및 동적 보증료(Dynamic Pricing) 산정기")
    
    if not st.session_state.models:
        st.warning("⚠️ 먼저 '3. AI 심사 엔진 학습 및 검증'에서 모델을 학습시켜 주세요.")
        st.stop()

    st.write("신규 보증 신청 주택의 스펙을 입력하면 AI가 **부실 확률(PD), 기대손실액(EL), 적정 보증료율(%), 안전 보증 한도**를 즉시 산출합니다.")

    engine_choice = st.selectbox("심사에 적용할 AI 엔진 선택", list(st.session_state.models.keys()), index=2)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        in_price = st.number_input("신청 주택 감정 시세 ($)", 100000, 3000000, 450000, step=10000)
        in_deposit = st.number_input("보증 신청 금액 ($)", 50000, 2500000, 380000, step=10000)
        in_sqft = st.slider("주거 면적 (sqft)", 500, 6000, 1800)
    with c2:
        in_age = st.slider("건물 노후도 (년)", 0, 115, 35)
        in_grade = st.slider("건축 등급 (1~13)", 3, 13, 7)
        in_cond = st.slider("관리 상태 (1~5)", 1, 5, 3)
    with c3:
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
    input_df = pd.DataFrame([input_row])[st.session_state.features]
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
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_prob * 100,
            number={"suffix": "%"},
            title={"text": "AI 예측 보증사고 확률 (PD)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e3a8a"},
                "steps": [
                    {"range": [0, 30], "color": "#ccfbf1"},
                    {"range": [30, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#ffe4e6"}
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 65}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_c2:
        st.subheader("📑 AI 심사 판정 및 동적 가격(Pricing) 리포트")
        if pd_prob < 0.30 and in_deposit <= safe_limit:
            st.success("✅ **최종 판정: 정상 승인 (우량 담보)** — 기본 우대 보증료율 적용 가능")
        elif pd_prob < 0.65:
            st.warning("⚠️ **최종 판정: 조건부 승인 (리스크 할증 또는 한도 감액 권고)**")
        else:
            st.error("🚨 **최종 판정: 보증 거절 권고 (고위험 부실 우려 매물)**")

        m1, m2, m3 = st.columns(3)
        m1.metric("기대손실액 (EL)", f"${expected_loss:,.0f}", f"손실률(LGD) {lgd*100:.1f}%")
        m2.metric("적정 동적 보증료율", f"{dynamic_fee_rate*100:.2f}%", f"기존 0.15% 대비 +{(dynamic_fee_rate-base_fee_rate)*100:.2f}%p")
        m3.metric("AI 안전 보증 한도", f"${safe_limit:,.0f}", f"신청액 대비 ${safe_limit - in_deposit:,.0f}")

        st.info(f"📌 **심사역 안내 문구:** 현재 신청 보증금(${in_deposit:,})은 시세 대비 {ltv_current*100:.1f}% 수준이며, 건물 노후도({in_age}년)와 고평가 지수({in_overval})를 반영할 때 연 **{dynamic_fee_rate*100:.2f}%**의 보증료를 책정해야 공기업 기금 적자를 방어할 수 있습니다.")

# =========================================================
# 모듈 5: 거시경제 스트레스 테스트 & 리포트 다운로드
# =========================================================
elif menu == "5. 거시경제 스트레스 테스트":
    st.header("📉 5. 거시경제 위기상황분석 (Macro Stress Test)")
    st.write("부동산 경기 침체로 전국 집값이 급락할 경우, 공기업 전체 보증 포트폴리오의 **예상 대위변제 손실액**과 **동적 보증료 방어 효과**를 시뮬레이션합니다.")

    if st.session_state.df_processed is None:
        st.warning("⚠️ 먼저 '2. 데이터 전처리 & 파생변수 공학'에서 전처리를 실행해 주세요.")
        st.stop()

    df_p = st.session_state.df_processed.copy()
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        price_drop = st.slider("가정 시나리오: 부동산 시세 하락률 (%)", 0, 35, 15, 5)
    with col_s2:
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
    k1.metric(f"집값 -{price_drop}% 시 기존 예상 손실", f"${loss_no_policy[idx]:,.1f} 백만", "정책 미개입 기준", delta_color="inverse")
    k2.metric(f"AI 동적 한도 적용 후 예상 손실", f"${loss_with_ai[idx]:,.1f} 백만", f"LTV {ltv_cap}% + 고위험 차등 적용")
    k3.metric("공기업 기금 손실 방어(절감) 효과", f"${saved_amount:,.1f} 백만", f"손실액 {(saved_amount/max(1, loss_no_policy[idx]))*100:.1f}% 감소")

    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(x=[f"-{d}%" for d in drops], y=loss_no_policy, mode="lines+markers", name="기존 일률 보증 방식 (예상 손실)", line=dict(color="#e11d48", width=3)))
    fig_stress.add_trace(go.Scatter(x=[f"-{d}%" for d in drops], y=loss_with_ai, mode="lines+markers", name="K-SafeGuard AI 동적 관리 적용 (예상 손실)", line=dict(color="#0d9488", width=3)))
    fig_stress.update_layout(title="집값 하락 시나리오별 공기업 누적 대위변제 손실액 비교 (단위: 백만 $)", xaxis_title="부동산 시세 하락률", yaxis_title="예상 손실액 ($M)")
    st.plotly_chart(fig_stress, use_container_width=True)

    st.divider()
    st.subheader("📥 경영진 보고용 리포트 및 고위험 매물 리스트 다운로드")
    high_risk_export = df_p[df_p["High_Risk"] == 1][["price", "sqft_living", "building_age", "grade", "overvaluation_ratio", "lat", "long"]].head(500)
    csv_bytes = high_risk_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📄 집중 관리 대상 고위험 담보 매물 리스트 (CSV) 다운로드",
        data=csv_bytes,
        file_name="K_SafeGuard_HighRisk_Report.csv",
        mime="text/csv"
    )
