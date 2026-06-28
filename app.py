"""
Patent Quality Intelligence -- MVP v2
Sector-normalized scores + actionable conclusions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Patent Quality Intelligence",
    page_icon="",
    layout="wide"
)

DATA_DIR = Path(__file__).parent


@st.cache_data
def load_data():
    for fname in ["company_scores_full.csv", "panel_v2.csv",
                  "company_scores_full.parquet", "panel_v2.parquet"]:
        f = DATA_DIR / fname
        if f.exists():
            return pd.read_csv(f) if fname.endswith(".csv") else pd.read_parquet(f)
    return pd.DataFrame()


@st.cache_data
def load_ticker_map():
    import json
    f = DATA_DIR / "ticker_map.json"
    with open(f) as file:
        raw = json.load(file)
    return {company: (info.get("ticker") if isinstance(info, dict) else str(info))
            for company, info in raw.items()}


def percentile_label(norm_score):
    """Convierte z-score normalizado a percentil y etiqueta."""
    from scipy.stats import norm as stats_norm
    pct = stats_norm.cdf(norm_score) * 100
    if pct >= 80:   return f"Top {100-pct:.0f}%", "green"
    if pct >= 50:   return f"Top {100-pct:.0f}%", "blue"
    if pct >= 20:   return f"Bottom {pct:.0f}%", "orange"
    return f"Bottom {pct:.0f}%", "red"


def trend_label(recent, historical):
    delta = recent - historical
    if delta > 0.05:  return "Improving", "green"
    if delta < -0.05: return "Declining", "red"
    return "Stable", "gray"


# ── Data ─────────────────────────────────────────────────────────────────────
df = load_data()
tmap = load_ticker_map()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Patent Quality Intelligence")
st.sidebar.caption("Innovation quality scores for 495 public companies")
ticker_input    = st.sidebar.text_input("Company ticker:", value="MSFT").upper().strip()
compare_ticker  = st.sidebar.text_input("Compare with (optional):", value="").upper().strip()
year_min = int(df["grant_year"].min()) if not df.empty else 2000
year_max = int(df["grant_year"].max()) if not df.empty else 2018
year_range = st.sidebar.slider("Year range", year_min, year_max, (2010, year_max))

st.sidebar.markdown("---")
st.sidebar.markdown("**Score interpretation**")
st.sidebar.markdown(
    "**Raw score (0–1):** probability a patent will be in the top 20% "
    "of citations in its field within 5 years.\n\n"
    "**Sector score (z-score):** position relative to sector peers. "
    "0 = sector average. +1 = top ~84%."
)

if not ticker_input:
    st.info("Enter a ticker in the sidebar.")
    st.stop()

if df.empty:
    st.error("No data loaded."); st.stop()

company_data = df[df["tk"] == ticker_input].copy()
if company_data.empty:
    st.warning(f"No data for **{ticker_input}**.")
    st.markdown("**Try these tickers:**")
    sample = sorted(df["tk"].dropna().unique())
    st.write(", ".join(sample[:60]))
    st.stop()

company_data = company_data[company_data["grant_year"].between(*year_range)]
if company_data.empty:
    st.warning(f"No patents in selected year range for {ticker_input}."); st.stop()

# Sector
sector = company_data["sector"].mode().iloc[0] if "sector" in company_data.columns else "Unknown"
sector_df = df[df["sector"] == sector] if "sector" in df.columns else df

# Scores
q_col     = "mean_quality"
norm_col  = "quality_norm" if "quality_norm" in company_data.columns else None
avg_raw   = float(company_data[q_col].mean())
avg_norm  = float(company_data[norm_col].mean()) if norm_col else 0.0
recent    = company_data[company_data["grant_year"] >= year_range[1]-2]
recent_raw  = float(recent[q_col].mean())   if len(recent) else avg_raw
recent_norm = float(recent[norm_col].mean()) if (norm_col and len(recent)) else avg_norm
total_patents = int(company_data["n_patents"].sum()) if "n_patents" in company_data.columns else 0

pct_label, pct_color = percentile_label(avg_norm) if norm_col else ("N/A", "gray")
trend_lbl, trend_col = trend_label(recent_norm, avg_norm) if norm_col else ("N/A","gray")

# ── Header ───────────────────────────────────────────────────────────────────
st.title(f"{ticker_input}")
st.caption(f"Sector: **{sector}** · Patents analyzed: **{total_patents:,}** "
           f"· Years: **{year_range[0]}–{year_range[1]}**")

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw quality score", f"{avg_raw:.3f}",
          help="Predicted citation impact (0-1). Absolute value across all sectors.")
if norm_col:
    c2.metric("Sector-adjusted score", f"{avg_norm:+.2f}",
              help="Z-score vs sector peers. 0=average, +1=top 84%, -1=bottom 16%.")
    c3.metric("Peer group rank", pct_label,
              help=f"Percentile position within {sector}")
    c4.metric("Innovation trend", trend_lbl,
              delta=f"{recent_norm-avg_norm:+.2f} vs historical avg",
              delta_color="normal" if trend_col == "green" else "inverse"
              if trend_col == "red" else "off",
              help="Recent 2 years vs full period average (sector-adjusted)")
else:
    c2.metric("Total patents", f"{total_patents:,}")

# ── Actionable conclusion ─────────────────────────────────────────────────────
if norm_col:
    st.markdown("---")
    if avg_norm > 0.5:
        st.success(
            f"**{ticker_input}** is in the **{pct_label}** of {sector} companies "
            f"by patent quality. Innovation output is above sector average "
            f"({avg_norm:+.2f} standard deviations). "
            f"Trend: **{trend_lbl}**.")
    elif avg_norm > -0.5:
        st.info(
            f"**{ticker_input}** is near the **sector average** in {sector}. "
            f"Patent quality is in line with peers (sector score: {avg_norm:+.2f}). "
            f"Trend: **{trend_lbl}**.")
    else:
        st.warning(
            f"**{ticker_input}** is in the **{pct_label}** of {sector} companies. "
            f"Patent quality is below sector average ({avg_norm:+.2f} std deviations). "
            f"Trend: **{trend_lbl}**.")

st.markdown("---")

# ── Quality over time ─────────────────────────────────────────────────────────
st.subheader("Innovation quality over time")
use_col = norm_col if norm_col else q_col
chart_data = company_data.groupby("grant_year")[use_col].mean().reset_index()
chart_data.columns = ["Year", ticker_input]

if compare_ticker:
    comp = df[(df["tk"] == compare_ticker) &
              df["grant_year"].between(*year_range)]
    if not comp.empty:
        cc = comp.groupby("grant_year")[use_col].mean().reset_index()
        cc.columns = ["Year", compare_ticker]
        chart_data = chart_data.merge(cc, on="Year", how="outer")
    else:
        st.caption(f"No data for {compare_ticker}.")

# Sector average
sec_chart = (sector_df[sector_df["grant_year"].between(*year_range)]
             .groupby("grant_year")[use_col].mean().reset_index())
sec_chart.columns = ["Year", f"{sector} avg"]
chart_data = chart_data.merge(sec_chart, on="Year", how="outer")
chart_data = chart_data.sort_values("Year").set_index("Year")

if norm_col:
    st.caption("Sector-adjusted score (z-score). Dashed line = sector average (0).")
st.line_chart(chart_data)

# ── Sector ranking ─────────────────────────────────────────────────────────────
st.subheader(f"Ranking within {sector}")
col_rank = norm_col if norm_col else q_col
ranking = (sector_df[sector_df["grant_year"].between(*year_range)]
           .groupby("tk")[col_rank].mean()
           .sort_values(ascending=False)
           .reset_index())
ranking.columns = ["Ticker", "Score"]
ranking["Score"] = ranking["Score"].round(3)
ranking.index = ranking.index + 1

def highlight(row):
    return ["background-color: #d4edda"] * 2 if row["Ticker"] == ticker_input else [""] * 2

st.dataframe(ranking.head(30).style.apply(highlight, axis=1),
             use_container_width=True, height=480)
pos = ranking[ranking["Ticker"] == ticker_input].index
if len(pos):
    st.caption(f"{ticker_input} ranks **#{pos[0]}** of {len(ranking)} "
               f"companies in {sector} ({year_range[0]}–{year_range[1]})")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Patent Quality Intelligence · Model trained on 4.1M USPTO patents · "
           "AUC 0.747 · Data: PatentsView · Not financial advice.")
