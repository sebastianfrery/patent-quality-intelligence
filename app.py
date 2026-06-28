import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

st.set_page_config(
    page_title="Patent Quality Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
    header    {visibility: hidden;}
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 18px 20px; border-left: 4px solid #6c757d; margin-bottom: 8px;
    }
    .metric-card.green  { border-left-color: #28a745; }
    .metric-card.blue   { border-left-color: #007bff; }
    .metric-card.orange { border-left-color: #fd7e14; }
    .metric-card.red    { border-left-color: #dc3545; }
    .metric-label { font-size:12px; color:#6c757d; text-transform:uppercase;
                    letter-spacing:0.5px; margin-bottom:4px; }
    .metric-value { font-size:28px; font-weight:700; color:#212529; }
    .metric-sub   { font-size:13px; color:#6c757d; margin-top:2px; }
    .conclusion-box { padding:14px 18px; border-radius:10px; margin:12px 0;
                      font-size:15px; line-height:1.6; }
    .conclusion-box.top    { background:#d4edda; border:1px solid #c3e6cb; color:#155724; }
    .conclusion-box.avg    { background:#d1ecf1; border:1px solid #bee5eb; color:#0c5460; }
    .conclusion-box.below  { background:#fff3cd; border:1px solid #ffeeba; color:#856404; }
    .conclusion-box.bottom { background:#f8d7da; border:1px solid #f5c6cb; color:#721c24; }
    .sector-badge { display:inline-block; padding:3px 10px; border-radius:20px;
                    font-size:12px; font-weight:600; background:#e9ecef; color:#495057; }
    .rank-badge   { display:inline-block; padding:3px 10px; border-radius:20px;
                    font-size:12px; font-weight:600; background:#cce5ff; color:#004085; }
    div[data-testid="stSidebar"] { background:#1a1a2e; }
    div[data-testid="stSidebar"] * { color:#e0e0e0 !important; }
    div[data-testid="stSidebar"] .stTextInput input {
        background:#16213e; border:1px solid #0f3460; color:#e0e0e0; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent

def clean_name(s):
    s = s.lower().strip()
    s = re.sub(r'\b(inc|corp|corporation|ltd|limited|llc|co|company|plc|'
               r'holdings|group|technologies|technology|systems|'
               r'international|the|and|&|sa|ag|gmbh|bv|nv|sas)\b',' ',s)
    s = re.sub(r'[^a-z0-9 ]',' ',s)
    return re.sub(r'\s+',' ',s).strip()

@st.cache_data
def load_ticker_scores():
    f = DATA_DIR/"company_scores_full.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_search_index():
    f = DATA_DIR/"company_index.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_search_scores():
    f = DATA_DIR/"company_search_scores.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

@st.cache_data
def load_names():
    f = DATA_DIR/"ticker_names.csv"
    if f.exists():
        df = pd.read_csv(f)
        return dict(zip(df["ticker"],df["name"]))
    return {}

def search_company(query, index_df, top_n=6):
    """Busca empresa por nombre o ticker. Devuelve lista de candidatos."""
    q = query.strip()
    if not q: return []
    qc = clean_name(q)
    # Substring match en nombre limpio
    mask = index_df["name_clean"].str.contains(qc, na=False)
    results = index_df[mask].sort_values("n_total", ascending=False).head(top_n)
    if len(results) == 0:
        # Fallback: cada palabra del query
        words = qc.split()
        if words:
            mask2 = index_df["name_clean"].str.contains(words[0], na=False)
            results = index_df[mask2].sort_values("n_total", ascending=False).head(top_n)
    return results[["company","sector","n_total","quality_avg","quality_norm_avg"]].to_dict("records")

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

def norm_to_pct(z):
    from scipy.stats import norm
    return norm.cdf(z)*100

def metric_card(label, value, sub="", color="blue"):
    return (f'<div class="metric-card {color}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            + (f'<div class="metric-sub">{sub}</div>' if sub else '') +
            '</div>')

# ── Load data ─────────────────────────────────────────────────────────────────
tk_df   = load_ticker_scores()
idx_df  = load_search_index()
scr_df  = load_search_scores()
names   = load_names()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Patent Quality Intelligence")
    st.markdown("*26,000+ companies from USPTO*")
    st.markdown("---")
    query = st.text_input("", placeholder="Company name or ticker (e.g. Moderna, MSFT, Palantir)...")
    st.markdown("---")
    year_min = 2000; year_max = 2018
    if not scr_df.empty:
        year_max = int(scr_df["grant_year"].max())
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

**Coverage:** 26,000+ companies with ≥10 USPTO patents (2000–2018)
**Model:** XGBoost M6, AUC 0.747
        """)
    st.markdown("---")
    st.markdown("### Get full access")
    st.markdown("""
**Free** — Current access\n- 26,000+ companies\n- 2000–2018 data\n- Sector comparison

**Pro** — €149/month\n- Data updated quarterly\n- Export to CSV / API\n- Priority support
    """)
    st.markdown(
        '<a href="mailto:sebastianfrery28@gmail.com?subject=Patent Quality Pro Access" '
        'style="display:inline-block;background:#28a745;color:white;padding:10px 16px;'
        'border-radius:8px;text-decoration:none;font-weight:bold;width:100%;'
        'text-align:center;box-sizing:border-box;margin-top:6px;">Request Pro Access</a>',
        unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Not financial advice.")

# ── Main ──────────────────────────────────────────────────────────────────────
if not query:
    st.markdown("## Patent Quality Intelligence")
    st.markdown("Search 26,000+ companies from USPTO patent data.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Technology**")
        st.code("Apple Inc.\nMicrosoft\nPalantir\nCrowdStrike\nNVIDIA")
    with col2:
        st.markdown("**Life Sciences**")
        st.code("Pfizer\nModerna\nJohnson & Johnson\nNovartis\nRoche")
    with col3:
        st.markdown("**Industrial**")
        st.code("General Electric\nBoeing\nHoneywell\nFord Motor\n3M")
    st.stop()

# ── Search ─────────────────────────────────────────────────────────────────────
# 1. Intentar como ticker primero
selected_company = None
data_source = None

q_upper = query.strip().upper()
if not tk_df.empty and q_upper in tk_df["tk"].values:
    # Es un ticker conocido — usar company_scores_full
    selected_company = q_upper
    data_source = "ticker"
    company_data = tk_df[tk_df["tk"] == q_upper].copy()
    display_name = names.get(q_upper, q_upper)
else:
    # Buscar por nombre en el indice
    results = search_company(query, idx_df)

    if not results:
        st.warning(f"No companies found for **{query}**. Try a different spelling.")
        st.stop()

    if len(results) == 1:
        chosen = results[0]["company"]
    else:
        options = [f"{r['company']} ({r['sector']}, {r['n_total']:,} patents)" for r in results]
        choice = st.selectbox(f"Found {len(results)} companies matching '{query}':", options)
        idx = options.index(choice)
        chosen = results[idx]["company"]

    selected_company = chosen
    data_source = "name"
    display_name = chosen
    company_data = scr_df[scr_df["company"] == chosen].copy() if not scr_df.empty else pd.DataFrame()

if company_data.empty:
    st.error(f"No data for **{selected_company}**."); st.stop()

company_data = company_data[company_data["grant_year"].between(*year_range)]
if company_data.empty:
    st.warning(f"No patents in {year_range[0]}–{year_range[1]}."); st.stop()

# ── Compute metrics ────────────────────────────────────────────────────────────
q_col   = "mean_quality"
n_col   = "quality_norm" if "quality_norm" in company_data.columns else None
sector  = company_data["sector"].mode().iloc[0] if "sector" in company_data.columns else "Unknown"
avg_raw = float(company_data[q_col].mean())
avg_norm= float(company_data[n_col].mean()) if n_col else 0.0
recent  = company_data[company_data["grant_year"] >= year_range[1]-2]
rec_norm= float(recent[n_col].mean()) if (n_col and len(recent)) else avg_norm
delta_t = rec_norm - avg_norm
total_p = int(company_data["n_patents"].sum()) if "n_patents" in company_data.columns else 0
pct     = norm_to_pct(avg_norm) if n_col else 50.0
color   = score_color(avg_norm) if n_col else "blue"
cls     = conclusion_class(avg_norm) if n_col else "avg"

# Ranking en sector
src = scr_df if data_source == "name" else tk_df
rank_col = n_col or q_col
if rank_col in src.columns and "sector" in src.columns:
    sector_src = src[(src["sector"]==sector) & src["grant_year"].between(*year_range)]
    ranking = (sector_src.groupby("company" if "company" in sector_src.columns else "tk")
               [rank_col].mean().sort_values(ascending=False).reset_index())
    ranking.columns = ["Name","Score"]
    rank_pos = int(ranking[ranking["Name"]==selected_company].index[0])+1 \
               if selected_company in ranking["Name"].values else "N/A"
    total_sector = len(ranking)
else:
    rank_pos = "N/A"; total_sector = 0; ranking = pd.DataFrame()

# ── Header ─────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3,1])
with h1:
    st.markdown(f"# {display_name}")
    if data_source == "ticker":
        st.caption(f"Ticker: **{selected_company}**")
    badges = f'<span class="sector-badge">{sector}</span>'
    if rank_pos != "N/A":
        badges += f'  <span class="rank-badge">#{rank_pos} of {total_sector} in sector</span>'
    st.markdown(badges, unsafe_allow_html=True)
with h2:
    st.markdown(f"<br>**{total_p:,}** patents", unsafe_allow_html=True)
    st.caption(f"{year_range[0]}–{year_range[1]}")

# ── KPIs ───────────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Sector score", f"{avg_norm:+.2f}",
                            f"vs {sector} average", color), unsafe_allow_html=True)
with c2:
    pct_lbl = f"Top {100-pct:.0f}%" if pct>=50 else f"Bottom {pct:.0f}%"
    r_lbl = f"#{rank_pos} of {total_sector}" if rank_pos!="N/A" else "N/A"
    st.markdown(metric_card("Peer ranking", pct_lbl, r_lbl, color), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Raw quality", f"{avg_raw:.3f}", "Absolute (0–1)", "blue"),
                unsafe_allow_html=True)
with c4:
    tw = "Improving" if delta_t>0.1 else ("Declining" if delta_t<-0.1 else "Stable")
    tc = "green" if delta_t>0.1 else ("red" if delta_t<-0.1 else "blue")
    st.markdown(metric_card("Trend", tw, f"{delta_t:+.2f} recent vs avg", tc),
                unsafe_allow_html=True)

# ── Conclusion ─────────────────────────────────────────────────────────────────
if n_col:
    if cls=="top":
        msg = (f"<b>{display_name}</b> is in the <b>top {100-pct:.0f}%</b> of {sector} "
               f"companies by patent quality ({avg_norm:+.2f} std above sector average).")
    elif cls=="avg":
        msg = (f"<b>{display_name}</b> is near the <b>sector average</b> in {sector} "
               f"(score: {avg_norm:+.2f}). Ranked #{rank_pos} of {total_sector} peers.")
    elif cls=="below":
        msg = (f"<b>{display_name}</b> is <b>below sector average</b> in {sector} "
               f"(score: {avg_norm:+.2f}, bottom {pct:.0f}% of peers).")
    else:
        msg = (f"<b>{display_name}</b> is in the <b>bottom {pct:.0f}%</b> of "
               f"{sector} peers (score: {avg_norm:+.2f}).")
    trend_txt = "Improving" if delta_t>0.1 else ("Declining" if delta_t<-0.1 else "Stable")
    msg += f" Trend: <b>{trend_txt}</b>."
    st.markdown(f'<div class="conclusion-box {cls}">{msg}</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────────
left, right = st.columns([3,2])

with left:
    st.subheader("Quality over time")
    use_col = n_col or q_col
    chart = company_data.groupby("grant_year")[use_col].mean().reset_index()
    chart.columns = ["Year", display_name[:25]]
    if not ranking.empty and "sector" in src.columns:
        sec_line = (sector_src.groupby("grant_year")[rank_col].mean().reset_index())
        sec_line.columns = ["Year", f"{sector} avg"]
        chart = chart.merge(sec_line, on="Year", how="outer")
    chart = chart.sort_values("Year").set_index("Year")
    st.line_chart(chart, height=260)

with right:
    st.subheader(f"Top 20 in {sector}")
    if not ranking.empty:
        ranking["Score"] = ranking["Score"].round(3)
        ranking.index = ranking.index + 1
        def highlight(row):
            return (["background-color:#fff3cd;font-weight:bold"]*2
                    if row["Name"]==selected_company else [""]*2)
        st.dataframe(ranking.head(20).style.apply(highlight, axis=1),
                     use_container_width=True, height=430)
        if isinstance(rank_pos,int) and rank_pos>20:
            st.caption(f"{display_name} is ranked #{rank_pos} (outside top 20)")
