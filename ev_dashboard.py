import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Adoption Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2a40;
}

[data-testid="stSidebar"] * {
    color: #a0aec0 !important;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0d1220 0%, #0f2040 50%, #0d1a30 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,200,150,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.hero-title span {
    color: #00c896;
}

.hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 300;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.hero-badge {
    display: inline-block;
    background: rgba(0,200,150,0.12);
    border: 1px solid rgba(0,200,150,0.3);
    color: #00c896;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-top: 16px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}

.kpi-card {
    background: #0d1220;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #00c896, #0080ff);
    opacity: 0.6;
}

.kpi-label {
    font-size: 0.72rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
    margin-bottom: 10px;
}

.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}

.kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 12px;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2a40;
    margin-left: 8px;
}

/* Chart container */
.chart-box {
    background: #0d1220;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Insight box */
.insight-box {
    background: rgba(0,200,150,0.05);
    border: 1px solid rgba(0,200,150,0.2);
    border-left: 3px solid #00c896;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.88rem;
    color: #a0aec0;
    line-height: 1.7;
}

.insight-box strong {
    color: #00c896;
}

/* Model comparison cards */
.model-card {
    background: #0d1220;
    border: 1px solid #1e2a40;
    border-radius: 12px;
    padding: 24px;
}

.model-card.winner {
    border-color: rgba(0,200,150,0.4);
    background: rgba(0,200,150,0.04);
}

.model-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 16px;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1a2436;
    font-size: 0.85rem;
}

.metric-row:last-child { border-bottom: none; }
.metric-label { color: #4a5568; }
.metric-val { color: #e2e8f0; font-weight: 500; font-family: 'Syne', sans-serif; }
.metric-val.best { color: #00c896; }

/* Sidebar nav */
.nav-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #2d3748 !important;
    padding: 8px 0 4px 0;
    font-weight: 600;
}

/* Success banner */
.success-banner {
    background: rgba(0,200,150,0.08);
    border: 1px solid rgba(0,200,150,0.25);
    border-radius: 8px;
    padding: 10px 16px;
    color: #00c896;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 24px;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
header {visibility: hidden;}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2a40;
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB THEME ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1220',
    'axes.facecolor': '#0d1220',
    'axes.edgecolor': '#1e2a40',
    'axes.labelcolor': '#718096',
    'axes.titlecolor': '#e2e8f0',
    'text.color': '#718096',
    'xtick.color': '#4a5568',
    'ytick.color': '#4a5568',
    'grid.color': '#1a2436',
    'grid.alpha': 0.8,
    'legend.facecolor': '#0d1220',
    'legend.edgecolor': '#1e2a40',
    'legend.labelcolor': '#a0aec0',
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ev_population.csv")
    df_clean = df.dropna(subset=['County', 'City', 'Model Year', 'Make'])
    df_clean = df_clean.copy()
    df_clean['Electric Range'] = df_clean['Electric Range'].fillna(0)
    return df_clean
@st.cache_data
def load_stations():
    try:
        df = pd.read_csv("charging_stations.csv")
        return df
    except:
        return pd.DataFrame()

stations_df = load_stations()
df = load_data()

# ── DATA CONTROL (FIXED SAFE INIT) ─────────────────────────────
if "active_df" not in st.session_state:
    st.session_state.active_df = df
df = st.session_state.active_df
# ── DATA CONTROL (ADD ONLY) ─────────────────────────────
if "active_df" not in st.session_state:
    st.session_state.active_df = df

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 8px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.1rem; font-weight: 800; color: #fff;'>⚡ EV Intelligence</div>
       
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

    selected_section = st.radio("", [
    "📊  Overview",
    "📈  EV Growth & Forecast",
    "🏭  Manufacturers",
    "🗺️  County Analysis",
    "🤖  ML Models",
    "⚡  Charging Stations",
    "➕  Add Dataset"
], label_visibility="collapsed")
   

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#2d3748; line-height:1.8;'>
        <div>📁 <span style='color:#4a5568'>Records</span> <span style='color:#fff; float:right'>{len(df):,}</span></div>
        <div>🗺️ <span style='color:#4a5568'>Counties</span> <span style='color:#fff; float:right'>{df['County'].nunique()}</span></div>
        <div>🏭 <span style='color:#4a5568'>Brands</span> <span style='color:#fff; float:right'>{df['Make'].nunique()}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-sub"></div>
    <div class="hero-title">EV Adoption <span>Intelligence</span> Platform</div>
    <div style='color:#64748b; font-size:0.9rem; margin-top:8px; font-weight:300;'>
        Predictive analytics & machine learning insights for Washington State electric vehicle adoption
    </div>
    <div class="hero-badge">⚡ Live Dashboard · 279,756 Records</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in selected_section:

    # KPI row
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">🔋</div>
            <div class="kpi-label">Total EV Records</div>
            <div class="kpi-value">279,756</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🗺️</div>
            <div class="kpi-label">Counties Covered</div>
            <div class="kpi-value">253</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🏭</div>
            <div class="kpi-label">Manufacturers</div>
            <div class="kpi-value">47</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">📅</div>
            <div class="kpi-label">Data Span</div>
            <div class="kpi-value" style='font-size:1.5rem;'>1999–2027</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">📋 Sample Dataset</div>', unsafe_allow_html=True)
        st.dataframe(
            df[['County', 'City', 'Model Year', 'Make', 'Model', 'Electric Vehicle Type', 'Electric Range']].head(8),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown('<div class="section-header">🥧 BEV vs PHEV Breakdown</div>', unsafe_allow_html=True)
        ev_type = df['Electric Vehicle Type'].value_counts()
        labels = ['BEV\n(Battery Electric)', 'PHEV\n(Plug-in Hybrid)']

        fig, ax = plt.subplots(figsize=(5, 4.2))
        wedges, texts, autotexts = ax.pie(
            ev_type.values,
            labels=labels,
            autopct='%1.1f%%',
            colors=['#00c896', '#0066cc'],
            startangle=90,
            wedgeprops={'linewidth': 2, 'edgecolor': '#0a0e1a'},
            textprops={'fontsize': 10}
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(11)
            at.set_fontweight('bold')
        for t in texts:
            t.set_color('#718096')

        ax.set_title('Vehicle Type Distribution', pad=16, fontsize=12, color='#e2e8f0')
        fig.patch.set_facecolor('#0d1220')
        st.pyplot(fig)

    st.markdown("""
    <div class="insight-box">
        🔍 <strong>Key Finding:</strong> Battery Electric Vehicles (BEV) dominate Washington State registrations,
        indicating strong consumer preference for fully electric vehicles. This trend supports continued
        investment in fast-charging infrastructure over plug-in hybrid support systems.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# EV GROWTH & FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif "Forecast" in selected_section:

    yearly = df.groupby('Model Year').size().reset_index()
    yearly.columns = ['Year', 'EV_Count']
    yearly = yearly[yearly['Year'] >= 2010]

    X = yearly[['Year']]
    y = yearly['EV_Count']
    lr = LinearRegression()
    lr.fit(X, y)
    future_years = [2025, 2026, 2027, 2028, 2029, 2030]
    future = pd.DataFrame({'Year': future_years})
    preds = lr.predict(future)

    st.markdown('<div class="section-header">📈 EV Adoption — Historical Trend & Forecast</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Shaded historical area
    ax.fill_between(yearly['Year'], yearly['EV_Count'], alpha=0.12, color='#00c896')
    ax.plot(yearly['Year'], yearly['EV_Count'], marker='o', color='#00c896',
            linewidth=2.5, markersize=6, label='Historical Data', zorder=5)

    # Forecast line
    all_x = list(yearly['Year'].values[-1:]) + future_years
    all_y = list(yearly['EV_Count'].values[-1:]) + list(preds)
    ax.plot(all_x, all_y, marker='s', color='#ff6b6b',
            linewidth=2, linestyle='--', markersize=5, label='Forecast (Linear Regression)', zorder=5)

    # Shaded forecast area
    ax.fill_between(all_x, all_y, alpha=0.06, color='#ff6b6b')

    # Annotation on last forecast point
    ax.annotate(f'{int(preds[-1]):,} EVs\nby 2030',
                xy=(2030, preds[-1]),
                xytext=(2028.2, preds[-1] * 0.75),
                color='#ff6b6b', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1220', edgecolor='#ff6b6b', alpha=0.8))

    ax.set_xlabel("Model Year", fontsize=10)
    ax.set_ylabel("EV Registrations", fontsize=10)
    ax.set_title("Washington State EV Growth Trajectory", fontsize=13, pad=16)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y')
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-header">📊 Forecast Table</div>', unsafe_allow_html=True)
    cols = st.columns(len(future_years))
    for i, (yr, pred) in enumerate(zip(future_years, preds)):
        with cols[i]:
            delta = f"+{int(pred - preds[i-1]):,}" if i > 0 else "baseline"
            st.metric(str(yr), f"{int(pred):,}", delta if i > 0 else None)

    st.markdown("""
    <div class="insight-box">
        📈 <strong>Forecast Insight:</strong> Linear Regression projects continued strong growth in EV adoption
        through 2030. The model captures the accelerating trend post-2018, driven by increased model availability,
        falling battery costs, and Washington State incentive programs.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MANUFACTURERS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Manufactur" in selected_section:

    st.markdown('<div class="section-header">🏭 Top 10 EV Manufacturers</div>', unsafe_allow_html=True)

    top_makes = df['Make'].value_counts().head(10)
    colors_bar = ['#00c896' if i == 0 else '#0d4f7a' for i in range(10)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(top_makes.index, top_makes.values, color=colors_bar,
                  width=0.65, edgecolor='#0a0e1a', linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, top_makes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                f'{val:,}', ha='center', va='bottom', fontsize=8, color='#718096')

    ax.set_title("EV Manufacturer Market Share — Washington State", fontsize=13, pad=16)
    ax.set_ylabel("Total Registrations", fontsize=10)
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, axis='y', alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        tesla_count = top_makes.get('TESLA', 0)
        total = len(df)
        st.markdown(f"""
        <div class="insight-box">
            🥇 <strong>Tesla</strong> leads with <strong>{tesla_count:,}</strong> registrations —
            approximately <strong>{tesla_count/total*100:.1f}%</strong> of all EVs in Washington State.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        top3 = top_makes.head(3)
        st.markdown(f"""
        <div class="insight-box">
            🏆 The <strong>top 3 manufacturers</strong> ({', '.join(top3.index)}) account for
            <strong>{top3.sum()/total*100:.1f}%</strong> of total EV registrations.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COUNTY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "County" in selected_section:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🗺️ Top 10 Counties</div>', unsafe_allow_html=True)
        top_counties = df['County'].value_counts().head(10)
        bar_colors = ['#00c896' if i == 0 else f'#{hex(13 + i*15)[2:]}4f7a' for i in range(10)]
        bar_colors = ['#00c896', '#00a87a', '#0080cc', '#0066aa', '#004f88',
                      '#003d6e', '#1a3a5c', '#0d3050', '#0a2540', '#071e30']

        fig, ax = plt.subplots(figsize=(7, 5.5))
        bars = ax.barh(top_counties.index[::-1], top_counties.values[::-1],
                       color=bar_colors[::-1], height=0.65, edgecolor='#0a0e1a')
        for bar, val in zip(bars, top_counties.values[::-1]):
            ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=8, color='#718096')
        ax.set_title("County EV Registration Rankings", fontsize=12, pad=12)
        ax.set_xlabel("Total EVs", fontsize=9)
        ax.grid(True, axis='x', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">🎯 Adoption Cluster Map</div>', unsafe_allow_html=True)
        county_stats = df.groupby('County').agg(
            EV_Count=('Make', 'count'),
            Avg_Range=('Electric Range', 'mean'),
            Avg_Year=('Model Year', 'mean')
        ).reset_index()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(county_stats[['EV_Count', 'Avg_Range', 'Avg_Year']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        county_stats['Cluster'] = kmeans.fit_predict(X_scaled)

        # Assign names based on EV count
        cluster_ev_means = county_stats.groupby('Cluster')['EV_Count'].mean().sort_values(ascending=False)
        name_map = {cluster_ev_means.index[0]: 'Early Adopter',
                    cluster_ev_means.index[1]: 'Moderate Adopter',
                    cluster_ev_means.index[2]: 'Slow Adopter'}
        county_stats['Cluster_Name'] = county_stats['Cluster'].map(name_map)

        cluster_colors = {'Early Adopter': '#00c896', 'Moderate Adopter': '#0080ff', 'Slow Adopter': '#ff6b6b'}

        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        for name, color in cluster_colors.items():
            subset = county_stats[county_stats['Cluster_Name'] == name]
            ax2.scatter(subset['EV_Count'], subset['Avg_Range'],
                        label=name, color=color, alpha=0.8, s=80, edgecolors='#0a0e1a', linewidth=0.5)
        ax2.set_title("K-Means County Clustering (3 Groups)", fontsize=12, pad=12)
        ax2.set_xlabel("Total EV Count", fontsize=9)
        ax2.set_ylabel("Avg Electric Range (miles)", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown('<div class="section-header">📋 County Classification Table</div>', unsafe_allow_html=True)
    display_df = county_stats[['County', 'EV_Count', 'Avg_Range', 'Avg_Year', 'Cluster_Name']].copy()
    display_df.columns = ['County', 'EV Count', 'Avg Range (mi)', 'Avg Model Year', 'Adoption Group']
    display_df['Avg Range (mi)'] = display_df['Avg Range (mi)'].round(1)
    display_df['Avg Model Year'] = display_df['Avg Model Year'].round(1)
    st.dataframe(display_df.sort_values('EV Count', ascending=False), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif "ML" in selected_section:

    yearly = df.groupby('Model Year').size().reset_index()
    yearly.columns = ['Year', 'EV_Count']
    yearly = yearly[yearly['Year'] >= 2010]
    X = yearly[['Year']]
    y = yearly['EV_Count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    lr_wins = lr_mae < rf_mae

    st.markdown('<div class="section-header">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        winner_class = "model-card winner" if lr_wins else "model-card"
        badge = " 🏆 Best Model" if lr_wins else ""
        st.markdown(f"""
        <div class="{winner_class}">
            <div class="model-name">📐 Linear Regression{badge}</div>
            <div class="metric-row">
                <span class="metric-label">Mean Absolute Error</span>
                <span class="metric-val {'best' if lr_wins else ''}">{lr_mae:,.0f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Root Mean Squared Error</span>
                <span class="metric-val">{lr_rmse:,.0f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Model Type</span>
                <span class="metric-val">Baseline / Linear</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Training Split</span>
                <span class="metric-val">80% / 20%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        winner_class = "model-card winner" if not lr_wins else "model-card"
        badge = " 🏆 Best Model" if not lr_wins else ""
        st.markdown(f"""
        <div class="{winner_class}">
            <div class="model-name">🌲 Random Forest{badge}</div>
            <div class="metric-row">
                <span class="metric-label">Mean Absolute Error</span>
                <span class="metric-val {'best' if not lr_wins else ''}">{rf_mae:,.0f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Root Mean Squared Error</span>
                <span class="metric-val">{rf_rmse:,.0f}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Estimators</span>
                <span class="metric-val">100 Trees</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Training Split</span>
                <span class="metric-val">80% / 20%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Visual Comparison</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    models = ['Linear\nRegression', 'Random\nForest']
    maes = [lr_mae, rf_mae]
    rmses = [lr_rmse, rf_rmse]
    x = np.arange(2)
    w = 0.35

    bars1 = axes[0].bar(x - w/2, maes, w, label='MAE', color=['#00c896', '#0080ff'], edgecolor='#0a0e1a')
    bars2 = axes[0].bar(x + w/2, rmses, w, label='RMSE', color=['#00a07a', '#0060cc'], edgecolor='#0a0e1a')
    for bar in list(bars1) + list(bars2):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                     f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8, color='#718096')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_title("MAE & RMSE Comparison", fontsize=11, pad=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Actual vs Predicted scatter
    axes[1].scatter(y_test, lr_pred, color='#00c896', alpha=0.8, s=80, label='Linear Regression', edgecolors='#0a0e1a', linewidth=0.5)
    axes[1].scatter(y_test, rf_pred, color='#0080ff', alpha=0.8, s=80, marker='s', label='Random Forest', edgecolors='#0a0e1a', linewidth=0.5)
    min_val = min(y_test.min(), lr_pred.min(), rf_pred.min())
    max_val = max(y_test.max(), lr_pred.max(), rf_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, alpha=0.6, label='Perfect Fit')
    axes[1].set_title("Actual vs Predicted", fontsize=11, pad=12)
    axes[1].set_xlabel("Actual Values", fontsize=9)
    axes[1].set_ylabel("Predicted Values", fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout(pad=2)
    st.pyplot(fig)

    winner_name = "Linear Regression" if lr_wins else "Random Forest"
    st.markdown(f"""
    <div class="insight-box">
        🏆 <strong>{winner_name}</strong> achieves the lowest MAE ({min(lr_mae, rf_mae):,.0f}), making it the
        recommended model for EV adoption forecasting on this dataset. The relatively small dataset (yearly
        aggregates) favours simpler models — Linear Regression captures the dominant growth trend effectively
        without overfitting.
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
# CHARGING STATIONS (MATCHES YOUR UI)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Charging" in selected_section:

    st.markdown('<div class="section-header">⚡ Charging Stations Infrastructure</div>', unsafe_allow_html=True)

    if stations_df.empty:
        st.markdown("""
        <div class="insight-box">
            ⚠️ Charging station data not found. Please ensure <strong>charging_stations.csv</strong> is in your project folder.
        </div>
        """, unsafe_allow_html=True)

    else:
        stations_df['level2'] = stations_df['ev_level2_evse_num'].fillna(0)
        stations_df['dc_fast'] = stations_df['ev_dc_fast_num'].fillna(0)

        total_stations = len(stations_df)
        total_level2 = int(stations_df['level2'].sum())
        total_fast = int(stations_df['dc_fast'].sum())

        # KPI STYLE MATCH
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">📍</div>
                <div class="kpi-label">Total Stations</div>
                <div class="kpi-value">{total_stations:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔌</div>
                <div class="kpi-label">Level 2 Chargers</div>
                <div class="kpi-value">{total_level2:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">⚡</div>
                <div class="kpi-label">DC Fast Chargers</div>
                <div class="kpi-value">{total_fast:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🏙️</div>
                <div class="kpi-label">Cities Covered</div>
                <div class="kpi-value">{stations_df['city'].nunique()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # TOP CITIES CHART (MATCH STYLE)
        st.markdown('<div class="section-header">🏙️ Top Cities by Charging Stations</div>', unsafe_allow_html=True)

        top_cities = stations_df['city'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top_cities.index, top_cities.values, color='#00c896', edgecolor='#0a0e1a')
        ax.set_title("Top Cities with Charging Infrastructure", fontsize=12, pad=12)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        # SAMPLE TABLE
        st.markdown('<div class="section-header">📋 Charging Station Sample</div>', unsafe_allow_html=True)

        st.dataframe(
            stations_df[['station_name', 'city', 'state', 'level2', 'dc_fast']].head(10),
            use_container_width=True,
            hide_index=True
        )

        # INSIGHT BOX (MATCH STYLE)
        st.markdown("""
        <div class="insight-box">
            ⚡ <strong>Infrastructure Insight:</strong> Charging stations are concentrated in major urban areas,
            with Level 2 chargers dominating the network. However, DC fast chargers remain limited, indicating
            potential infrastructure gaps for long-distance EV travel and future expansion opportunities.
        </div>
        """, unsafe_allow_html=True)


        # ═══════════════════════════════════════════════════════
# ➕ ADD DATASET (NEW FEATURE ONLY)
# ═══════════════════════════════════════════════════════
elif "Add Dataset" in selected_section:

    st.markdown('<div class="section-header">➕ Add Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        new_df = pd.read_csv(uploaded_file)

        st.markdown("### 👀 Preview Data")
        st.dataframe(new_df.head(), use_container_width=True)

        if st.button("🚀 Submit & Update Dashboard"):

            required_cols = ['County', 'City', 'Model Year', 'Make']

            missing = [c for c in required_cols if c not in new_df.columns]

            if missing:
                st.error(f"Missing columns: {missing}")

            else:
                new_df = new_df.dropna(subset=required_cols)

                if 'Electric Range' not in new_df.columns:
                    new_df['Electric Range'] = 0
                else:
                    new_df['Electric Range'] = new_df['Electric Range'].fillna(0)

                # 🔥 UPDATE GLOBAL DATASET (IMPORTANT)
                st.session_state.active_df = new_df

                st.success("Dataset updated successfully! Go to Overview.")