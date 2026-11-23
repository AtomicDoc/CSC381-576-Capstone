# What changed between Cyberpunk 2077's worst year and its comeback?

CSC 381/576 Data Science · Team 7 capstone · Fall 2026

We run a driver analysis on 760,330 Steam reviews of Cyberpunk 2077 (Dec 2020 – Aug 2024): which features push a
review toward **Not Recommended**, measured with SHAP values, and why the share of negative reviews fell from
about 25% in 2021 to 6.5% in 2024. The full plan, with each person's steps, is in `Team7_Capstone_Plan.docx`
(shared separately).

**Due: Monday, November 30, 12:00 PM (noon).**

## Where things go

| Folder | What goes in it | Who |
|---|---|---|
| `data/` | The raw Kaggle file. **Never commit it** (about 300 MB); see `data/README.md` | Everyone, locally |
| `src/` | One script per analysis, named as in the plan | Everyone |
| `results/` | Tables (CSV) and findings notes saved by the scripts | Everyone |
| `figures/` | Charts (PNG) saved by the scripts | Everyone |

Each folder's own README lists the exact file names.

## Who does what

| Person | Part | Script | Hands in (in `results/` and `figures/`) |
|---|---|---|---|
| 1 · Desmond | Data, model, SHAP results, the 2021 → 2024 gap split, slideshow, submitting | `src/01_prepare_data.py`, `src/02_driver_analysis.py`, `src/06_gap.py` | `shap_values.csv`, `driver_ranking.csv`, `gap_table.csv`, `gap_waterfall.png` |
| 2 · (name) | Complaint themes | `src/03_themes.py` | `theme_words.txt`, `theme_table.csv`, `notes_person2.txt`, `themes_ranking.png`, `themes_over_time.png` |
| 3 · (name) | Hours played and who was reviewing | `src/04_hours.py` | `hours_table.csv`, `notes_person3.txt`, `hours_shap.png`, `hours_2021_vs_2024.png` |
| 4 · (name) | Chinese-language reviews and review length | `src/05_language_length.py` | `chinese_table.csv`, `length_table.csv`, `notes_person4.txt`, `length_shap.png` |

Persons 2–4 hand in by **Friday, November 27, 9 PM**.

## How to run

From this folder, with the raw file in `data/`:

```
pip install -r requirements.txt
python src/01_prepare_data.py
python src/02_driver_analysis.py
python src/03_themes.py
python src/04_hours.py
python src/05_language_length.py
python src/06_gap.py
```

Run 01 and 02 first: the other scripts read `results/shap_values.csv`, which 02 makes.

## Rules

1. Every number comes from running code: no numbers typed from memory or copied from somewhere else.
2. Every chart and table is saved by a script, so running the script makes it again.
3. Back up every SHAP result with plain data (a % Not Recommended table next to it).
4. Say "drives the model", not "causes".
