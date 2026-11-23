# data/

Put the raw review file here: `Cyberpunk_2077_Steam_Reviews.csv` (about 300 MB).

Download it from Kaggle (free account needed): search for **"Cyberpunk 2077 Steam Reviews (as of Aug 8, 2024)"**,
or go to <https://www.kaggle.com/datasets/filas1212/cyberpunk-2077-steam-reviews-as-of-aug-8-2024>.
Unzip the download and move the CSV into this folder.

Then run `python src/01_prepare_data.py` from the top folder. It makes **`data/clean.csv`**: all 760,330 reviews,
one row each, with every feature (date, hours, language, review length, the theme columns...) but no review text.
**This is "the full review file" in the plan.** Use it for your % Not Recommended tables:

```python
import pandas as pd
d = pd.read_csv("data/clean.csv", parse_dates=["date"])
d.groupby("theme_bugs_performance")["not_recommended"].mean()   # % Not Recommended with / without the theme
```

**Do not commit any CSV in this folder.** They are too big for GitHub, and `.gitignore` already blocks them.
Everyone makes their own copy on their own computer.
