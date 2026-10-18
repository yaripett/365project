import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

# CSV Loaded
DATA_PATH = Path(__file__).parent / "archive" / "weekly_team_stats_offense.csv"
df = pd.read_csv(DATA_PATH)
df = df[df["season_type"] == "REG"].copy()
print(df.columns.tolist())   # <-- inspect this to find the true score/win column
print(df[df["game_id"] == "2024_07_LV_LA"][
    ["team", "total_off_points", "total_def_points",
     "home_win", "away_win", "wins", "losses"]])


# features
df = df.sort_values(["team", "season", "week"]).reset_index(drop=True)
raw_features = ["passing_yards", "rushing_yards", "interception",
                "pass_touchdown", "rush_touchdown",
                "first_down_pass", "first_down_rush", "total_yards"]

for f in raw_features:
    df["past_4_avg_" + f] = (
        df.groupby(["team", "season"])[f]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=4).mean())
    )
features = ["past_4_avg_" + f for f in raw_features]

games, skipped = [], {"bad_teams": 0, "missing_features": 0,"missing_outcome": 0, "tie": 0, "conflict": 0, "label_mismatch": 0}
prev_wins = df.groupby(["team", "season"])["wins"].shift(1).fillna(0)
df["won_game"] = df["wins"] - prev_wins

for game_id, game in df.groupby("game_id"):
   
    if len(game) != 2:
        skipped["bad_teams"] += 1
        continue
    _, _, away_name, home_name = game_id.split("_")[:4]
    h = game[game["team"] == home_name]
    a = game[game["team"] == away_name]
    if len(h) != 1 or len(a) != 1:
        skipped["bad_teams"] += 1
        continue
    h, a = h.iloc[0], a.iloc[0]
    if h[features].isna().any() or a[features].isna().any():
        skipped["missing_features"] += 1
        continue

    needed = [h["total_off_points"], a["total_off_points"],
              h["total_def_points"], a["total_def_points"]]
    if any(pd.isna(v) for v in needed):
        skipped["missing_outcome"] += 1
        continue

    home_score = h["total_off_points"] + a["total_def_points"]
    away_score = a["total_off_points"] + h["total_def_points"]

    if home_score == away_score:
        skipped["tie"] += 1
        continue

    home_win_target = int(home_score > away_score)

    # Cross-check against the dataset's own win totals
    if h["won_game"] == 1 and a["won_game"] == 0:
        recorded_home_win = 1
    elif h["won_game"] == 0 and a["won_game"] == 1:
        recorded_home_win = 0
    else:
        recorded_home_win = None
    if recorded_home_win is not None and recorded_home_win != home_win_target:
        skipped["label_mismatch"] += 1
        continue

    row = {"game_id": game_id, "season": h["season"], "week": h["week"],
           "home_team": home_name, "away_team": away_name,
           "home_win_target": home_win_target}
    for f in features:
        row[f + "_diff"] = h[f] - a[f]
    games.append(row)

games_df = pd.DataFrame(games)
feature_columns = [f + "_diff" for f in features]
games_df = games_df.dropna(subset=feature_columns + ["home_win_target"]).reset_index(drop=True)
print("Games kept:", len(games_df), "| skipped:", skipped)


feature_columns = [f + "_diff" for f in features]
train_df = games_df[games_df["season"] <= 2023]
test_df = games_df[games_df["season"] >= 2024].reset_index(drop=True)
X_train, y_train = train_df[feature_columns], train_df["home_win_target"]
X_test, y_test = test_df[feature_columns], test_df["home_win_target"]
print("Train:", len(train_df), "Test:", len(test_df))

# Model comparison
models = {
    "Always home (baseline)": DummyClassifier(strategy="constant", constant=1),
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                            random_state=42),
}
results = []
for name, m in models.items():
    m.fit(X_train, y_train)
    results.append({"Model": name,"Test accuracy": accuracy_score(y_test, m.predict(X_test))})
print("\n--- RESULTS ---")
print(pd.DataFrame(results).to_string(index=False, formatters={"Test accuracy": "{:.2%}".format}))

# Labeled confusion matrix
lr = models["Logistic Regression"]
cm = confusion_matrix(y_test, lr.predict(X_test))
print("\n--- CONFUSION MATRIX (Logistic Regression) ---")
print(pd.DataFrame(cm,index=["Actual: Away win", "Actual: Home win"],columns=["Pred: Away win", "Pred: Home win"]))

# One real held-out prediction 
i = 0
p_home = lr.predict_proba(X_test.iloc[[i]])[0, 1]
g = test_df.iloc[i]
pred = g["home_team"] if p_home >= 0.5 else g["away_team"]
actual = g["home_team"] if g["home_win_target"] == 1 else g["away_team"]
print("\n--- HELD-OUT PREDICTION ---")
print(f"Game: {g['game_id']} | Predicted winner: {pred} "
      f"(P(home win)={p_home:.1%}) | Actual winner: {actual}")
