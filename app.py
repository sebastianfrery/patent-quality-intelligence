import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Patent Quality Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}

    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 18px 20px;
        border-left: 4px solid #6c757d;
        margin-bottom: 8px;
    }
    .metric-card.green  { border-left-color: #28a745; }
    .metric-card.blue   { border-left-color: #007bff; }
    .metric-card.orange { border-left-color: #fd7e14; }
    .metric-card.red    { border-left-color: #dc3545; }
    .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #212529; }
    .metric-sub   { font-size: 13px; color: #6c757d; margin-top: 2px; }

    .conclusion-box {
        padding: 14px 18px;
        border-radius: 10px;
        margin: 12px 0;
        font-size: 15px;
        line-height: 1.6;
    }
    .conclusion-box.top    { background:#d4edda; border:1px solid #c3e6cb; color:#155724; }
    .conclusion-box.avg    { background:#d1ecf1; border:1px solid #bee5eb; color:#0c5460; }
    .conclusion-box.below  { background:#fff3cd; border:1px solid #ffeeba; color:#856404; }
    .conclusion-box.bottom { background:#f8d7da; border:1px solid #f5c6cb; color:#721c24; }

    .sector-badge {
        display:inline-block; padding:3px 10px; border-radius:20px;
        font-size:12px; font-weight:600;
        background:#e9ecef; color:#495057; margin-bottom:8px;
    }
    .rank-badge {
        display:inline-block; padding:3px 10px; border-radius:20px;
        font-size:12px; font-weight:600;
        background:#cce5ff; color:#004085;
    }
    .trend-up   { color:#28a745; font-weight:600; }
    .trend-down { color:#dc3545; font-weight:600; }
    .trend-flat { color:#6c757d; font-weight:600; }

    .stDataFrame { border-radius: 10px; }
    div[data-testid="stSidebar"] { background:#1a1a2e; }
    div[data-testid="stSidebar"] * { color:#e0e0e0 !important; }
    div[data-testid="stSidebar"] .stTextInput input {
        background:#16213e; border:1px solid #0f3460; color:#e0e0e0;
    }
    div[data-testid="stSidebar"] hr { border-color:#333; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    for fname in ["company_scores_full.csv","panel_v2.csv",
                  "company_scores_full.parquet","panel_v2.parquet"]:
        f = DATA_DIR / fname
        if f.exists():
            return pd.read_csv(f) if fname.endswith(".csv") else pd.read_parquet(f)
    return pd.DataFrame()

@st.cache_data
def load_ticker_map():
    import json
    with open(DATA_DIR/"ticker_map.json") as f:
        raw = json.load(f)
    return {c: (i.get("ticker") if isinstance(i,dict) else str(i)) for c,i in raw.items()}

def norm_to_percentile(z):
    from scipy.stats import norm
    return norm.cdf(z) * 100

def score_color(z):
    if z >  0.5: return "green"
    if z > -0.5: return "blue"
    if z > -1.0: return "orange"
    return "red"

def conclusion_class(z):
    if z >  0.5: return "top"
    if z > -0.5: return "avg"
    if z > -1.0: return "below"
    return "bottom"

def metric_card(label, value, sub="", color="blue"):
    return f"""<div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {"<div class='metric-sub'>"+sub+"</div>" if sub else ""}
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patent Quality Intelligence")
    st.markdown("*Innovation scores for 667 public companies*")
    st.markdown("---")
    ticker_input   = st.text_input("", value="MSFT", placeholder="Ticker (e.g. AAPL)").upper().strip()
    compare_ticker = st.text_input("", value="", placeholder="Compare with...").upper().strip()
    st.markdown("---")
    df = load_data()
    year_min = int(df["grant_year"].min()) if not df.empty else 2000
    year_max = int(df["grant_year"].max()) if not df.empty else 2018
    year_range = st.slider("Year range", year_min, year_max, (2010, year_max))
    st.markdown("---")
    with st.expander("How to read scores"):
        st.markdown("""
**Raw score (0–1)**
Probability a patent will be in the top 20% of citations in its field within 5 years.

**Sector score (z-score)**
Position vs sector peers.
- `+1.0` = top 84%
- `0.0` = sector average
- `-1.0` = bottom 16%

**Model:** XGBoost M6, AUC 0.747
**Data:** 4.1M USPTO patents, PatentsView
        """)
    st.markdown("---")
    st.caption("Not financial advice.")

# ── Main ─────────────────────────────────────────────────────────────────────
if df.empty:
    st.error("Data not loaded."); st.stop()

if not ticker_input:
    st.info("Enter a ticker in the sidebar to get started."); st.stop()

company_data = df[df["tk"] == ticker_input].copy()
if company_data.empty:
    st.error(f"No data for **{ticker_input}**.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Technology examples:**")
        st.code("MSFT  AAPL  GOOGL  INTC\nNVDA  CSCO  QCOM  TXN\nIBM   ORCL  AMD   AMAT")
    with col2:
        st.markdown("**Life Sciences examples:**")
        st.code("JNJ   PFE   MRK   ABBV\nAMGN  GILD  BIIB  REGN\nNVS   RHHBY MDT   ABT")
    st.stop()

company_data = company_data[company_data["grant_year"].between(*year_range)]
if company_data.empty:
    st.warning(f"No patents in {year_range[0]}–{year_range[1]} for {ticker_input}."); st.stop()

# Datos
q_col    = "mean_quality"
norm_col = "quality_norm" if "quality_norm" in company_data.columns else None
sector   = company_data["sector"].mode().iloc[0] if "sector" in company_data.columns else "Unknown"
sector_df = df[df["sector"] == sector] if "sector" in df.columns else df

avg_raw    = float(company_data[q_col].mean())
avg_norm   = float(company_data[norm_col].mean()) if norm_col else 0.0
recent_data = company_data[company_data["grant_year"] >= year_range[1]-2]
recent_norm = float(recent_data[norm_col].mean()) if (norm_col and len(recent_data)) else avg_norm
delta_trend = recent_norm - avg_norm
total_pat   = int(company_data["n_patents"].sum()) if "n_patents" in company_data.columns else 0
pct         = norm_to_percentile(avg_norm) if norm_col else 50.0
color       = score_color(avg_norm) if norm_col else "blue"
cls         = conclusion_class(avg_norm) if norm_col else "avg"

# Ranking dentro del sector
col_rank = norm_col if norm_col else q_col
ranking = (sector_df[sector_df["grant_year"].between(*year_range)]
           .groupby("tk")[col_rank].mean()
           .sort_values(ascending=False).reset_index())
ranking.columns = ["Ticker","Score"]
rank_pos = int(ranking[ranking["Ticker"]==ticker_input].index[0]) + 1 if ticker_input in ranking["Ticker"].values else "N/A"
total_in_sector = len(ranking)

# Trend label
if delta_trend > 0.1:   trend_html = '<span class="trend-up">Improving</span>'
elif delta_trend < -0.1: trend_html = '<span class="trend-down">Declining</span>'
else:                    trend_html = '<span class="trend-flat">Stable</span>'

# ── Header ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3,1])
with h1:
    st.markdown(f"# {ticker_input}")
    st.markdown(
        f'<span class="sector-badge">{sector}</span>  '
        f'<span class="rank-badge">#{rank_pos} of {total_in_sector} in sector</span>',
        unsafe_allow_html=True)
with h2:
    st.markdown(f"<br>**{total_pat:,}** patents analyzed", unsafe_allow_html=True)
    st.caption(f"{year_range[0]} – {year_range[1]}")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Sector score", f"{avg_norm:+.2f}",
                             f"vs {sector} average", color), unsafe_allow_html=True)
with c2:
    pct_lbl = f"Top {100-pct:.0f}%" if pct >= 50 else f"Bottom {pct:.0f}%"
    st.markdown(metric_card("Peer ranking", pct_lbl,
                             f"#{rank_pos} of {total_in_sector}", color), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Raw quality", f"{avg_raw:.3f}",
                             "Absolute score (0–1)", "blue"), unsafe_allow_html=True)
with c4:
    trend_word = "Improving" if delta_trend > 0.1 else ("Declining" if delta_trend < -0.1 else "Stable")
    tc = "green" if delta_trend > 0.1 else ("red" if delta_trend < -0.1 else "blue")
    st.markdown(metric_card("Innovation trend", trend_word,
                             f"{delta_trend:+.2f} recent vs historical", tc), unsafe_allow_html=True)

# ── Conclusion ────────────────────────────────────────────────────────────────
if norm_col:
    if cls == "top":
        msg = (f"<b>{ticker_input}</b> is in the <b>top {100-pct:.0f}%</b> of "
               f"{sector} companies by patent quality ({avg_norm:+.2f} std above sector average). "
               f"Innovation trend: {trend_html}.")
    elif cls == "avg":
        msg = (f"<b>{ticker_input}</b> is near the <b>sector average</b> in {sector} "
               f"(sector score: {avg_norm:+.2f}). "
               f"Ranked #{rank_pos} of {total_in_sector} peers. Trend: {trend_html}.")
    elif cls == "below":
        msg = (f"<b>{ticker_input}</b> is <b>below sector average</b> in {sector} "
               f"(sector score: {avg_norm:+.2f}, bottom {pct:.0f}% of peers). "
               f"Trend: {trend_html}.")
    else:
        msg = (f"<b>{ticker_input}</b> is in the <b>bottom {pct:.0f}%</b> of "
               f"{sector} peers by patent quality (sector score: {avg_norm:+.2f}). "
               f"Trend: {trend_html}.")
    st.markdown(f'<div class="conclusion-box {cls}">{msg}</div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("Quality over time")
    use_col = norm_col if norm_col else q_col
    chart = company_data.groupby("grant_year")[use_col].mean().reset_index()
    chart.columns = ["Year", ticker_input]

    if compare_ticker and compare_ticker != ticker_input:
        comp = df[(df["tk"]==compare_ticker) & df["grant_year"].between(*year_range)]
        if not comp.empty:
            cc = comp.groupby("grant_year")[use_col].mean().reset_index()
            cc.columns = ["Year", compare_ticker]
            chart = chart.merge(cc, on="Year", how="outer")
        else:
            st.caption(f"No data for {compare_ticker}.")

    sec_line = (sector_df[sector_df["grant_year"].between(*year_range)]
                .groupby("grant_year")[use_col].mean().reset_index())
    sec_line.columns = ["Year", f"{sector} avg"]
    chart = chart.merge(sec_line, on="Year", how="outer").sort_values("Year").set_index("Year")
    lbl = "Sector-adjusted score (z-score)" if norm_col else "Raw quality score"
    st.caption(lbl)
    st.line_chart(chart, height=260)

with right:
    st.subheader(f"Top 20 in {sector}")
    ranking["Score"] = ranking["Score"].round(3)
    ranking.index = ranking.index + 1
    top20 = ranking.head(20).copy()

    # Highlight el ticker buscado
    def style_row(row):
        if row["Ticker"] == ticker_input:
            return ["background-color:#fff3cd; font-weight:bold"]*2
        return [""]*2

    st.dataframe(top20.style.apply(style_row, axis=1),
                 use_container_width=True, height=430)
    if isinstance(rank_pos, int) and rank_pos > 20:
        st.caption(f"{ticker_input} is ranked #{rank_pos} (outside top 20)")
