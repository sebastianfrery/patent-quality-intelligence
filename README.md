---
title: Patent Quality Intelligence
emoji: 🔬
colorFrom: blue
colorTo: teal
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Patent Quality Intelligence

Predict innovation quality from structural patent features.

Trained on 4.1M USPTO patents (2000-2018). Scores 490+ public companies.

**Model:** XGBoost M6, AUC = 0.747  
**Data:** PatentsView (USPTO, public domain)  
**Features:** Prior art coverage, claims breadth, art unit, assignee track record

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

- Enter any company ticker (AAPL, MSFT, RHHBY...)
- Get predicted patent quality score (0-1)
- Compare against sector average
- See quality trend over time
- Rank companies within their technology sector

## Methodology

Quality score = predicted probability that a patent will be in the top 20%
of forward citations within its CPC class over 5 years.

Features used (M6 model):
- `log_backward_cit` — breadth of prior art cited
- `log_num_claims` — patent scope
- `art_unit` — USPTO examination group
- `assignee_n_prior` — assignee track record
- `cpc_code` — technology classification
- `inventor_quality_rate` — inventor historical quality
- `log_n_figures` — technical complexity
- `log_app_cit` — applicant-disclosed prior art
- `log_family_size` — international filing scope
