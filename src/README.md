# src/

One script per part. Each one must run from start to finish with one command, from the top folder of the repo
(for example `python src/03_themes.py`), and save its own tables in `results/` and charts in `figures/`.
The professor will run them.

| Script | Owner | What it does |
|---|---|---|
| `01_prepare_data.py` | Desmond | Cleans the raw file and builds the features (including the theme columns) |
| `02_driver_analysis.py` | Desmond | Checks the model, computes SHAP values → `results/shap_values.csv`, `results/driver_ranking.csv` |
| `03_themes.py` | Person 2 | Complaint themes: ranking chart, theme table, themes over time |
| `04_hours.py` | Person 3 | Hours played: SHAP curve, hours table, 2021 vs 2024 reviewers |
| `05_language_length.py` | Person 4 | Chinese vs other reviews, review length: SHAP curve and length table |
| `06_gap.py` | Desmond | The 2021 → 2024 gap split and waterfall chart |

If AI helped write your script, say so in a comment on its first line.
