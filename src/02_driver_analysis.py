# AI note: this script was written with AI assistance (Claude).
"""
Step 2: check the model, then compute SHAP values for the driver analysis.

  1. Model check: train on 2020-2023, test on 2024 (the same split as Report 2).
  2. Refit gradient boosting on all reviews and compute SHAP values for 20,000 randomly chosen reviews.
  3. Save the files the rest of the team works from.

Reads   data/clean.csv   (made by 01_prepare_data.py)
Writes  results/model_scores.csv     the model check
        results/shap_values.csv      one row per sampled review: its data, the model's prediction, one SHAP value per feature
        results/driver_ranking.csv   every feature ranked by mean |SHAP|
        results/numbers.json         other numbers quoted in the deck
        figures/driver_ranking.png, figures/beeswarm.png

Run from the top folder of the repo:   python src/02_driver_analysis.py      (a few minutes)

HOW TO READ THE SHAP VALUES: they are in log-odds of Not Recommended. A positive value means that feature pushed
this review toward Not Recommended; a negative value pushed it toward Recommended. For each review,
base value + all its SHAP values = the model's log-odds for that review.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
SEED = 0
SAMPLE = 20_000

df = pd.read_csv(os.path.join("data", "clean.csv"), parse_dates=["date"])
THEMES = [c for c in df.columns if c.startswith("theme_")]
LANGS = ["chinese", "russian_cyrillic", "korean_japanese", "other"]   # english_latin is the comparison group

# Features the model sees. All are known when a review is posted, except that hours_played is each player's
# total in Aug 2024 (see the caution in 01_prepare_data.py).
X = df[["hours_played", "review_words", "days_since_launch", "received_free"] + THEMES].copy()
for lang in LANGS:
    X["lang_" + lang] = (df["language"] == lang).astype(int)
y = df["not_recommended"].values
LANG_COLS = ["lang_" + l for l in LANGS]

NAMES = {"hours_played": "Hours played", "review_words": "Review length (words)",
         "days_since_launch": "Date posted", "received_free": "Got the game free", "language": "Language",
         "theme_bugs_performance": "Mentions bugs / performance", "theme_story_characters": "Mentions story / characters",
         "theme_fixed_redemption": "Mentions it being fixed", "theme_refunds_promises": "Mentions refunds / broken promises",
         "theme_price_value": "Mentions price / value", "theme_gameplay": "Mentions gameplay"}
N = {}   # numbers quoted in the deck


def gbm():
    # early_stopping=False so the model is the same on every run (early stopping uses a random validation split)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=SEED)


def scores(name, y_true, p):
    pred = (p >= 0.5).astype(int)
    return {"model": name, "accuracy": round(accuracy_score(y_true, pred), 4),
            "precision_not_rec": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall_not_rec": round(recall_score(y_true, pred), 4),
            "roc_auc": round(roc_auc_score(y_true, p), 4), "pr_auc": round(average_precision_score(y_true, p), 4)}


# ------------------------------------------------------------------ 1. model check (train 2020-2023, test 2024)
train = (df["year"] <= 2023).values
rows = [scores("Baseline (always 'Recommended')", y[~train], np.zeros((~train).sum()))]

lr_X = X.copy()
lr_X[["hours_played", "review_words"]] = np.log1p(lr_X[["hours_played", "review_words"]])
scaler = StandardScaler().fit(lr_X[train])
lr = LogisticRegression(max_iter=5000, tol=1e-8).fit(scaler.transform(lr_X[train]), y[train])
rows.append(scores("Logistic regression", y[~train], lr.predict_proba(scaler.transform(lr_X[~train]))[:, 1]))

check = gbm().fit(X[train], y[train])
rows.append(scores("Gradient boosting (ours)", y[~train], check.predict_proba(X[~train])[:, 1]))
model_scores = pd.DataFrame(rows)
model_scores.to_csv(os.path.join("results", "model_scores.csv"), index=False)
print(model_scores.to_string(index=False))
N["model_check"] = rows
N["n_train"], N["n_test"], N["n_all"] = int(train.sum()), int((~train).sum()), len(df)
N["test_pct_not_recommended"] = round(float(y[~train].mean()), 4)

# ------------------------------------------------------------------ 2. refit on all reviews, SHAP on a sample
model = gbm().fit(X, y)
rng = np.random.default_rng(SEED)
pick = np.sort(rng.choice(len(df), SAMPLE, replace=False))
Xs = X.iloc[pick]
explainer = shap.TreeExplainer(model)
sv = pd.DataFrame(explainer.shap_values(Xs), columns=X.columns, index=Xs.index)
base = float(np.ravel(explainer.expected_value)[0])
margin = model.decision_function(Xs)
assert np.allclose(sv.sum(axis=1) + base, margin, atol=1e-6), "SHAP values should add up to the model's output"
N["base_log_odds"] = round(base, 4)
N["base_probability"] = round(float(1 / (1 + np.exp(-base))), 4)

# The four language columns are one feature: add their SHAP values together (like body type in the class example)
sv["language"] = sv[LANG_COLS].sum(axis=1)
sv = sv.drop(columns=LANG_COLS)
FEATURES = ["hours_played", "review_words", "days_since_launch", "received_free", "language"] + THEMES

out = df.iloc[pick][["review_id", "date", "year", "not_recommended", "hours_played", "review_words",
                     "days_since_launch", "received_free", "language"] + THEMES].copy()
out.insert(4, "predicted_chance_not_rec", model.predict_proba(Xs)[:, 1].round(5))
for f in FEATURES:
    out["shap_" + f] = sv[f].round(5).values
out.to_csv(os.path.join("results", "shap_values.csv"), index=False)
N["sample_size"] = SAMPLE
N["sample_reviews_per_year"] = out["year"].value_counts().sort_index().to_dict()
N["sample_reviews_per_language"] = out["language"].value_counts().to_dict()

# ------------------------------------------------------------------ 3. driver ranking
rank_rows = []
for f in FEATURES:
    s = sv[f]
    if f == "language":
        direction = "Chinese/other-language reviews vs English: see shap_values.csv"
    elif Xs[f].nunique() == 2:   # a yes/no feature: average push when yes minus when no
        d = s[Xs[f] == 1].mean() - s[Xs[f] == 0].mean()
        direction = "yes -> more Not Recommended" if d > 0 else "yes -> less Not Recommended"
    else:                        # a number: does a higher value push toward Not Recommended?
        r = pd.Series(Xs[f].values).rank().corr(pd.Series(s.values).rank())
        direction = "higher -> more Not Recommended" if r > 0 else "higher -> less Not Recommended"
    rank_rows.append({"feature": f, "name": NAMES[f], "mean_abs_shap": round(float(s.abs().mean()), 4),
                      "direction": direction})
ranking = pd.DataFrame(rank_rows).sort_values("mean_abs_shap", ascending=False)
ranking.to_csv(os.path.join("results", "driver_ranking.csv"), index=False)
print("\n" + ranking.to_string(index=False))
N["ranking"] = ranking.to_dict(orient="records")

# ------------------------------------------------------------------ figures (Desmond's slides)
fig, ax = plt.subplots(figsize=(8, 4.8))
r = ranking.iloc[::-1]
ax.barh(r["name"], r["mean_abs_shap"], color="#2a78d6")
ax.set_xlabel("Average |SHAP value| (log-odds of Not Recommended)")
ax.set_title("What drives a Not Recommended prediction", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join("figures", "driver_ranking.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

Xs_named = Xs.drop(columns=LANG_COLS).assign(language=(df.iloc[pick]["language"] != "english_latin").astype(int))
exp = shap.Explanation(values=sv[FEATURES].values, base_values=np.full(len(sv), base),
                       data=Xs_named[FEATURES].values,
                       feature_names=[NAMES[f] if f != "language" else "Not English/Latin script" for f in FEATURES])
shap.plots.beeswarm(exp, max_display=12, show=False)
fig = plt.gcf()
fig.set_size_inches(8.5, 5.5)
fig.savefig(os.path.join("figures", "beeswarm.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

with open(os.path.join("results", "numbers.json"), "w") as f:
    json.dump(N, f, indent=1, default=str)
print("\nSaved results/model_scores.csv, shap_values.csv, driver_ranking.csv, numbers.json and 2 figures")
