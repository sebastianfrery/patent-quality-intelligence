"""
Patent Quality Intelligence -- MVP Streamlit app.

Muestra la calidad de cartera de patentes de empresas cotizadas,
comparada contra su sector y con evolucion temporal.
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
    out = {}
    for company, info in raw.items():
        tk = info.get("ticker") if isinstance(info, dict) else str(info)
        out[company] = tk
    return out

def get_sector_name(art_center):
    m = {1:"Hardware / Electronics", 2:"Software / Computing",
         3:"Biotech / Pharma / Medtech", 4:"Chemistry / Materials",
         5:"Mechanical / Transport", 6:"Design"}
    return m.get(int(art_center), "Other") if pd.notna(art_center) else "Unknown"

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("Patent Quality Intelligence")
st.caption("Predicting innovation quality from structural patent features — "
           "trained on 4.1M USPTO patents (2000-2018), scored to 2025.")

df = load_data()
tmap = load_ticker_map()

# Sidebar
st.sidebar.header("Search")
all_tickers = sorted(df["tk"].dropna().unique()) if "tk" in df.columns else []
ticker_input = st.sidebar.text_input("Company ticker (e.g. AAPL, MSFT, RHHBY):",
                                      value="MSFT").upper().strip()
compare_ticker = st.sidebar.text_input("Compare with (optional):",
                                        value="").upper().strip()
year_range = st.sidebar.slider("Year range", 2010,
                                int(df["grant_year"].max()) if "grant_year" in df.columns else 2024,
                                (2015, int(df["grant_year"].max()) if "grant_year" in df.columns else 2024))

st.sidebar.markdown("---")
st.sidebar.markdown("**About the score**")
st.sidebar.markdown(
    "Quality score (0-1) predicts the probability that a patent "
    "will be in the top 20% of citations in its field within 5 years. "
    "Model: XGBoost M6, AUC = 0.747."
)

# Main
if not ticker_input:
    st.info("Enter a ticker in the sidebar to get started.")
    st.stop()

col_filt = "tk" if "tk" in df.columns else None
if col_filt is None:
    st.error("Data not loaded correctly.")
    st.stop()

company_data = df[df["tk"] == ticker_input].copy()

if company_data.empty:
    st.warning(f"No data found for **{ticker_input}**. "
               f"Try a different ticker or check the spelling.")
    st.markdown("**Available tickers (sample):**")
    st.write(", ".join(sorted(df["tk"].dropna().unique())[:50]))
    st.stop()

# Filter by year
company_data = company_data[
    company_data["grant_year"].between(year_range[0], year_range[1])
]

if company_data.empty:
    st.warning(f"No patents in the selected year range for {ticker_input}.")
    st.stop()

# Quality col
q_col = "mean_quality" if "mean_quality" in company_data.columns else "q_raw"
if q_col not in company_data.columns:
    st.error("Quality column not found.")
    st.stop()

# Sector
sector_col = "art_center" if "art_center" in company_data.columns else None
sector = get_sector_name(company_data[sector_col].mode().iloc[0]) \
         if sector_col and not company_data[sector_col].dropna().empty else "Unknown"

# ── KPIs ────────────────────────────────────────────────────────────────────
total_patents = int(company_data["n_patents"].sum()) if "n_patents" in company_data.columns \
                else len(company_data)
avg_quality   = float(company_data[q_col].mean())
recent_q      = float(company_data[company_data["grant_year"] >= year_range[1]-2][q_col].mean()) \
                if len(company_data[company_data["grant_year"] >= year_range[1]-2]) > 0 \
                else avg_quality

# Sector benchmark
if sector_col:
    sector_val = company_data[sector_col].mode().iloc[0]
    sector_df  = df[df[sector_col] == sector_val]
    sector_avg = float(sector_df[q_col].mean())
    vs_sector  = avg_quality - sector_avg
else:
    sector_avg = float(df[q_col].mean())
    vs_sector  = avg_quality - sector_avg

st.subheader(f"{ticker_input} — {sector}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Quality score", f"{avg_quality:.3f}",
          help="Average predicted citation impact (0-1). Higher = more impactful patents.")
c2.metric("vs Sector avg", f"{vs_sector:+.3f}",
          delta_color="normal",
          help=f"Sector average: {sector_avg:.3f}")
c3.metric("Recent quality (last 2y)", f"{recent_q:.3f}",
          delta=f"{recent_q-avg_quality:+.3f}",
          help="Quality trend — is innovation quality improving?")
c4.metric("Patents analyzed", f"{total_patents:,}")

st.markdown("---")

# ── Quality over time ────────────────────────────────────────────────────────
st.subheader("Innovation quality over time")

chart_data = company_data.groupby("grant_year")[q_col].mean().reset_index()
chart_data.columns = ["Year", ticker_input]

if compare_ticker:
    comp_data = df[(df["tk"] == compare_ticker) &
                   df["grant_year"].between(year_range[0], year_range[1])]
    if not comp_data.empty:
        comp_chart = comp_data.groupby("grant_year")[q_col].mean().reset_index()
        comp_chart.columns = ["Year", compare_ticker]
        chart_data = chart_data.merge(comp_chart, on="Year", how="outer")
    else:
        st.caption(f"No data for {compare_ticker} in this range.")

# Sector average line
sec_chart = sector_df[sector_df["grant_year"].between(
    year_range[0], year_range[1])].groupby("grant_year")[q_col].mean().reset_index()
sec_chart.columns = ["Year", "Sector avg"]
chart_data = chart_data.merge(sec_chart, on="Year", how="outer").sort_values("Year")
chart_data = chart_data.set_index("Year")

st.line_chart(chart_data)

# ── Sector ranking ───────────────────────────────────────────────────────────
st.subheader(f"Ranking in {sector}")
if sector_col:
    ranking = (df[(df[sector_col] == sector_val) &
                   df["grant_year"].between(year_range[0], year_range[1])]
               .groupby("tk")[q_col]
               .mean()
               .sort_values(ascending=False)
               .reset_index())
    ranking.columns = ["Ticker", "Quality Score"]
    ranking["Quality Score"] = ranking["Quality Score"].round(4)
    ranking.index = ranking.index + 1

    # Highlight the searched company
    def highlight_row(row):
        if row["Ticker"] == ticker_input:
            return ["background-color: #e8f4f8"] * len(row)
        return [""] * len(row)

    st.dataframe(
        ranking.head(30).style.apply(highlight_row, axis=1),
        use_container_width=True, height=500
    )
    # Show position
    pos = ranking[ranking["Ticker"] == ticker_input].index
    if len(pos) > 0:
        st.caption(f"{ticker_input} ranks **#{pos[0]}** of "
                   f"{len(ranking)} companies in {sector}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Patent Quality Intelligence | "
    "Model trained on 4.1M USPTO patents | "
    "Scores 590+ public companies | "
    "Data: PatentsView (USPTO) | "
    "Not financial advice."
)
