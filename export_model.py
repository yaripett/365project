import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "archive" / "weekly_team_stats_offense.csv"
OUT_PATH = Path(__file__).parent / "model.json"
TRAIN_THROUGH = 2023
FIRST_SEASON = 2019  # seasons exported to the app (keeps the file small)

RAW = ["passing_yards", "rushing_yards", "interception", "pass_touchdown",
       "rush_touchdown", "first_down_pass", "first_down_rush", "total_yards"]
FEATURES = ["past_4_avg_" + f for f in RAW]

df = pd.read_csv(DATA_PATH)
df = df[df["season_type"] == "REG"].copy()
df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
for f in RAW:
    df["past_4_avg_" + f] = (df.groupby(["team", "season"])[f]
                             .transform(lambda x: x.shift(1).rolling(4, min_periods=4).mean()))
prev_wins = df.groupby(["team", "season"])["wins"].shift(1).fillna(0)
df["won_game"] = df["wins"] - prev_wins

# Build games with the corrected labels 
rows = []
for game_id, game in df.groupby("game_id"):
    if len(game) != 2:
        continue
    _, _, away_name, home_name = game_id.split("_")[:4]
    h, a = game[game["team"] == home_name], game[game["team"] == away_name]
    if len(h) != 1 or len(a) != 1:
        continue
    h, a = h.iloc[0], a.iloc[0]
    if h[FEATURES].isna().any() or a[FEATURES].isna().any():
        continue
    pts = [h["total_off_points"], a["total_off_points"], h["total_def_points"], a["total_def_points"]]
    if any(pd.isna(v) for v in pts):
        continue
    hs, as_ = h["total_off_points"] + a["total_def_points"], a["total_off_points"] + h["total_def_points"]
    if hs == as_:
        continue
    target = int(hs > as_)
    if h["won_game"] == 1 and a["won_game"] == 0:
        recorded = 1
    elif h["won_game"] == 0 and a["won_game"] == 1:
        recorded = 0
    else:
        recorded = None
    if recorded is not None and recorded != target:
        continue
    row = {"season": h["season"], "y": target}
    for f in FEATURES:
        row[f + "_diff"] = h[f] - a[f]
    rows.append(row)

games = pd.DataFrame(rows).dropna()
cols = [f + "_diff" for f in FEATURES]
train, test = games[games["season"] <= TRAIN_THROUGH], games[games["season"] > TRAIN_THROUGH]

scaler = StandardScaler().fit(train[cols])
model = LogisticRegression(max_iter=1000).fit(scaler.transform(train[cols]), train["y"])
acc = accuracy_score(test["y"], model.predict(scaler.transform(test[cols])))
print(f"Held-out accuracy: {acc:.2%} on {len(test)} games")

# Export each team's games so React can compute 'last 4 before week W'
teams = {}
for (season, team), g in df[df["season"] >= FIRST_SEASON].groupby(["season", "team"]):
    g = g.sort_values("week")
    teams.setdefault(str(int(season)), {})[team] = [
        {"w": int(r.week), "s": [round(float(v), 1) for v in r[RAW].fillna(0)]}
        for _, r in g.iterrows()
    ]

out = {
    "trainThrough": TRAIN_THROUGH,
    "accuracy": round(float(acc), 4),
    "testGames": int(len(test)),
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "coef": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "teams": teams,
}
OUT_PATH.write_text(json.dumps(out))
print("Wrote", OUT_PATH)
