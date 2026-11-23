# AI note: this script was written with AI assistance (Claude).
"""
Step 1: clean the raw Steam reviews and build every feature the team uses.

Reads   data/Cyberpunk_2077_Steam_Reviews.csv   (raw Kaggle file, see data/README.md)
        results/theme_words.txt                  (the six complaint-theme word lists)
Writes  data/clean.csv                           (one row per review, no review text; NOT committed, too big)

Run from the top folder of the repo:   python src/01_prepare_data.py      (about 1 minute)

data/clean.csv is "the full review file" in the plan. Columns:
  review_id, date, year, not_recommended (1 = Not Recommended, 0 = Recommended),
  hours_played, received_free (1/0), days_since_launch, review_words, language,
  and one 1/0 column per theme: theme_bugs_performance, theme_story_characters, ...
"""
import os
import re

import pandas as pd

RAW = os.path.join("data", "Cyberpunk_2077_Steam_Reviews.csv")
THEMES_FILE = os.path.join("results", "theme_words.txt")
CLEAN = os.path.join("data", "clean.csv")
LAUNCH = pd.Timestamp("2020-12-10")   # Cyberpunk 2077's release day

if not os.path.exists(RAW):
    raise SystemExit(f"Can't find {RAW}. Download it from Kaggle and put it in data/ (see data/README.md).")

# keep_default_na=False keeps reviews whose text is literally "N/A" or "null" as text
raw = pd.read_csv(RAW, keep_default_na=False)
raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]   # the first column name has a hidden character
print(f"Raw file: {len(raw):,} reviews")

df = pd.DataFrame({"review_id": raw["ReviewID"]})

# Dates come in two formats: 12/9/2020 for most rows, and 8-Aug-24 for the last 3,461 (Aug 8-20, 2024)
date_main = pd.to_datetime(raw["Date Posted"], format="%m/%d/%Y", errors="coerce")
date_alt = pd.to_datetime(raw["Date Posted"], format="%d-%b-%y", errors="coerce")
df["date"] = date_main.fillna(date_alt)
df["year"] = df["date"].dt.year

# The target: 1 = Not Recommended, 0 = Recommended
df["not_recommended"] = (raw["Rating"] == "Not Recommended").astype(int)

# Hours Played: values of 1,000+ have a thousands separator ("1,043.90").
# CAUTION: this is each player's TOTAL when the data was collected (Aug 2024), not their hours when they posted.
df["hours_played"] = pd.to_numeric(raw["Hours Played"].astype(str).str.replace(",", ""), errors="coerce")
df["received_free"] = raw["Received for Free"].astype(str).str.upper().eq("TRUE").astype(int)
df["days_since_launch"] = (df["date"] - LAUNCH).dt.days

text = raw["Review"].astype(str).str.strip()
low = text.str.lower()

# Review length = chunks of text between spaces. Chinese and Japanese have no spaces,
# so a whole review in those languages counts as one or two "words".
df["review_words"] = text.str.split().str.len().fillna(0).astype(int)

# Language, guessed from the writing system (the same rules as Report 1). Order matters:
# Japanese text also contains Chinese characters, so kana/Hangul is checked first.
df["language"] = "other"
df.loc[text.str.contains(r"[A-Za-z]{2,}", regex=True), "language"] = "english_latin"
df.loc[text.str.contains(r"[Ѐ-ӿ]", regex=True), "language"] = "russian_cyrillic"
df.loc[text.str.contains(r"[一-鿿]", regex=True), "language"] = "chinese"
df.loc[text.str.contains(r"[가-힯぀-ヿ]", regex=True), "language"] = "korean_japanese"

# Complaint themes: 1 if the review contains any word or phrase on that theme's line (whole words only).
themes = {}
for line in open(THEMES_FILE, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#"):
        name, words = line.split(":", 1)
        themes[name.strip()] = [w.strip() for w in words.split(",") if w.strip()]
for name, words in themes.items():
    parts = [re.escape(w) if not w[0].isalnum() else r"\b" + re.escape(w) + r"\b" for w in words]
    df["theme_" + name] = low.str.contains("|".join(parts), regex=True).astype(int)

df = df.dropna(subset=["date", "hours_played"]).sort_values(["date", "review_id"]).reset_index(drop=True)
df.to_csv(CLEAN, index=False)

print(f"Saved {CLEAN}: {len(df):,} reviews, {df.shape[1]} columns (no review text)")
print("Languages:", df["language"].value_counts().to_dict())
print("Share of reviews mentioning each theme:")
print(df[[c for c in df.columns if c.startswith("theme_")]].mean().round(3).to_string())
