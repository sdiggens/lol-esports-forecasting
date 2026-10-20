"""
Do team-stat and player-form features add predictive value ON TOP OF the rating model?

For each game we build PRE-GAME form for both teams (exponentially weighted average of
each team's previous games only, half-life = HL games), take blue-minus-red differences,
and fit a logistic regression:

    logit P(blue wins) = a * rating_logit + sum_k b_k * (feature_k,blue - feature_k,red) + c

Fitted on the middle of the season, scored on the last 20% (unseen). Compared with the
same regression using rating_logit alone, so any gain is purely from the extra features.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import math, numpy as np, pandas as pd
from ratings import data_file, data_files
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ratings import load_games, run, INTERNATIONAL, TIER1

HL = 10  # half-life in games for the form averages
raw = pd.read_csv(data_file(2026), low_memory=False)
raw["date"] = pd.to_datetime(raw["date"])
g = load_games([data_file(2026)])
_, p_rating = run(g)
g["rating_logit"] = np.log(p_rating / (1 - p_rating))

# ---------- team form ----------
t = raw[raw.position == "team"].copy()
t["soul"] = (pd.to_numeric(t.elementaldrakes, errors="coerce") >= 4).astype(float)
t["comeback"] = np.where(t.golddiffat15 < 0, t.result, np.nan)        # win rate when behind @15
t["throw"] = np.where(t.golddiffat15 > 0, 1 - t.result, np.nan)       # loss rate when ahead @15
team_feats = {"first_blood": "firstblood", "first_drake": "firstdragon", "first_tower": "firsttower",
              "first_herald": "firstherald", "grubs": "void_grubs", "soul": "soul", "atakhan": "atakhans",
              "gd10": "golddiffat10", "xpd10": "xpdiffat10", "gd15": "golddiffat15", "gspd": "gspd",
              "comeback": "comeback", "throw": "throw", "win_rate": "result"}
t = t.sort_values(["date", "gameid"])
for name, col in team_feats.items():
    t[name] = (t.groupby("teamid")[col]
               .transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).ewm(halflife=HL, ignore_na=True).mean()))

# ---------- player form (current starting five) ----------
pl = raw[raw.position != "team"].sort_values(["date", "gameid"]).copy()
for col in ["golddiffat15", "xpdiffat15", "csdiffat15"]:
    pl[f"pf_{col}"] = (pl.groupby("playerid")[col]
                       .transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).ewm(halflife=HL, ignore_na=True).mean()))
pl["player_games"] = pl.groupby("playerid").cumcount()
pagg = pl.groupby(["gameid", "teamid"]).agg(
    player_gd15=("pf_golddiffat15", "sum"), player_xpd15=("pf_xpdiffat15", "sum"),
    player_csd15=("pf_csdiffat15", "sum"), min_player_games=("player_games", "min")).reset_index()
t = t.merge(pagg, on=["gameid", "teamid"], how="left")

feat_cols = list(team_feats) + ["player_gd15", "player_xpd15", "player_csd15"]
b = t[t.side == "Blue"][["gameid"] + feat_cols].set_index("gameid")
r = t[t.side == "Red"][["gameid"] + feat_cols].set_index("gameid")
diff = (b - r).add_prefix("d_")
g = g.merge(diff, left_on="gameid", right_index=True, how="left")

d20, d80 = g.date.quantile(0.2), g.date.quantile(0.8)
train = (g.date >= d20) & (g.date < d80)
test = g.date >= d80
y = g.y.values


def fit_score(cols, label):
    X = g[cols].fillna(0.0).values
    sc = StandardScaler().fit(X[train])
    m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[train]), y[train])
    p = m.predict_proba(sc.transform(X))[:, 1]
    out = {}
    for mask, nm in [(test, "test_all"), (test & g.league.isin(TIER1 | INTERNATIONAL).values, "test_major")]:
        ll = -(y[mask] * np.log(p[mask]) + (1 - y[mask]) * np.log(1 - p[mask]))
        out[nm] = ll
    return p, out, m


base_p, base_ll, _ = fit_score(["rating_logit"], "rating")
rows = []
groups = {"objectives (FB, drake, tower, herald, grubs, soul, atakhan)":
              ["d_first_blood", "d_first_drake", "d_first_tower", "d_first_herald", "d_grubs", "d_soul", "d_atakhan"],
          "early economy (gold & xp diff @10)": ["d_gd10", "d_xpd10"],
          "late game (comeback rate, throw rate)": ["d_comeback", "d_throw"],
          "player form (starting five gold/xp/cs diff @15)": ["d_player_gd15", "d_player_xpd15", "d_player_csd15"],
          "everything above + gspd + win rate": [c for c in diff.columns]}
for gname, cols in groups.items():
    _, ll, m = fit_score(["rating_logit"] + cols, gname)
    for k in ["test_all", "test_major"]:
        d = base_ll[k] - ll[k]
        rows.append(dict(features=gname, slice=k, base=base_ll[k].mean(), with_feats=ll[k].mean(),
                         gain=d.mean(), se=d.std() / math.sqrt(len(d))))
# features WITHOUT the rating, for reference
_, ll_alone, _ = fit_score(list(diff.columns), "features only")
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))
print("\nFeatures only, no rating: test_all", round(ll_alone["test_all"].mean(), 4),
      " test_major", round(ll_alone["test_major"].mean(), 4))
res.to_csv("feature_test_results.csv", index=False)
