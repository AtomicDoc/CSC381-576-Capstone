# results/

Tables (CSV) and findings notes. Every CSV here should be saved by a script in `src/`.

| File | Made by | Owner | Status |
|---|---|---|---|
| `theme_words.txt` | written by hand: the word list for each theme, tested on all reviews | | done |
| `model_scores.csv` | `02_driver_analysis.py` | Desmond | done |
| `shap_values.csv` | `02_driver_analysis.py` | Desmond | done |
| `driver_ranking.csv` | `02_driver_analysis.py` | Desmond | done |
| `numbers.json` | `02_driver_analysis.py` | Desmond | done |
| `theme_table.csv` | `03_themes.py` | Person 2 | |
| `notes_person2.txt` | written by hand | Person 2 | |
| `hours_table.csv` | `04_hours.py` | Person 3 | |
| `notes_person3.txt` | written by hand | Person 3 | |
| `chinese_table.csv` | `05_language_length.py` | Person 4 | |
| `length_table.csv` | `05_language_length.py` | Person 4 | |
| `notes_person4.txt` | written by hand | Person 4 | |
| `gap_table.csv` | `06_gap.py` | Desmond | |

## How to read `shap_values.csv`

20,000 reviews picked at random from all 760,330 (the same 20,000 every run). One row per review.

```python
import pandas as pd
s = pd.read_csv("results/shap_values.csv", parse_dates=["date"])
```

**About the review (the data):**

| Column | Meaning |
|---|---|
| `review_id`, `date`, `year` | Which review, and when it was posted |
| `not_recommended` | The real answer: 1 = Not Recommended, 0 = Recommended |
| `predicted_chance_not_rec` | The model's chance (0 to 1) that this review is Not Recommended |
| `hours_played` | The reviewer's total hours **as of Aug 2024** (not at the time of the review) |
| `review_words` | Length of the review, counted between spaces (Chinese/Japanese reviews count as 1–2 "words") |
| `days_since_launch` | Days between launch (Dec 10, 2020) and the review |
| `received_free` | 1 if the reviewer got the game free |
| `language` | `english_latin`, `chinese`, `russian_cyrillic`, `korean_japanese` or `other` (guessed from the writing system) |
| `theme_...` (6 columns) | 1 if the review mentions that theme (word lists in `theme_words.txt`) |

**The SHAP values (one per feature), all starting with `shap_`:**

- A SHAP value says how much that feature pushed **this review's** prediction.
  **Positive = pushed toward Not Recommended. Negative = pushed toward Recommended.** 0 = no effect.
- The unit is log-odds (the model's internal scale). You don't need to convert it: compare SHAP values with each
  other, average them, and look at the sign.
- Example: `shap_hours_played` = -0.40 means this reviewer's hours made the model **less** likely to call the review
  negative.
- `shap_language` is one number for language (the four language columns added together).
- For every review: `base_log_odds` (in `numbers.json`) + the sum of all its `shap_` columns = the model's output for
  that review. So averages of SHAP values can be added and subtracted (this is what the gap split uses).

**Handy one-liners:**

```python
s["shap_hours_played"].abs().mean()                        # how much hours matters on average
s.groupby("year")["shap_theme_bugs_performance"].mean()    # how the bugs push changed by year
s.groupby("language")["shap_review_words"].mean()          # length push, by language
```

## `driver_ranking.csv`

Every feature, ranked by its average absolute SHAP value (bigger = matters more to the model), with its direction.

## Findings notes (`notes_personN.txt`)

3 to 5 bullet points. Each bullet is one finding, has a number in it, and says which file the number came from.
Example:

```
- Reviews that mention bugs are 41% Not Recommended vs. 12% for the rest (results/theme_table.csv).
```
