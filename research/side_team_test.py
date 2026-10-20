"""Do some teams have their own red/blue strength, beyond the league-wide blue edge?
Uses the model's pre-game predictions (which already include the global blue edge)
and asks whether a team's leftover blue-vs-red performance persists into the future."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import glob, numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, TIER1
files = data_files()
g = load_games(files).reset_index(drop=True)
pp = pd.read_pickle("game_preds_before.pkl")[["gameid", "p_pre"]]
g = g.merge(pp, on="gameid", how="left")
raw = pd.concat([pd.read_csv(f, usecols=["gameid", "game", "participantid"], low_memory=False) for f in files])
gn = raw[raw.participantid == 100].drop_duplicates("gameid").set_index("gameid")["game"]
g["gnum"] = g.gameid.map(gn)
g = g.dropna(subset=["p_pre"])
g["year"] = g.date.dt.year
# long format: one row per team per game, residual in logit-ish 'score' terms
rows = []
for side, tid, name in (("blue", "b_teamid", "b_teamname"), ("red", "r_teamid", "r_teamname")):
    p = g.p_pre if side == "blue" else 1 - g.p_pre
    y = g.y if side == "blue" else 1 - g.y
    rows.append(pd.DataFrame(dict(gameid=g.gameid, year=g.year, date=g.date, league=g.league, gnum=g.gnum,
                                  team=g[tid], name=g[name], side=side, p=p, y=y)))
L = pd.concat(rows, ignore_index=True)
L["res"] = L.y - L.p

def table(df):
    t = df.groupby(["team", "side"]).agg(n=("y", "size"), w=("y", "sum"), e=("p", "sum")).unstack("side")
    return t

# 1) Los Ratones
lr = L[L.name.str.contains("Ratones", case=False, na=False)]
print("Los Ratones by year/side (wins, games, model-expected wins):")
print(lr.groupby(["year", "side"]).agg(games=("y", "size"), wins=("y", "sum"), expected=("p", "sum")).round(1))
print("  game-1 only:")
print(lr[lr.gnum == 1].groupby(["year", "side"]).agg(games=("y", "size"), wins=("y", "sum"), expected=("p", "sum")).round(1))

# 2) Is there real between-team spread? per team-year: (blue over-performance) - (red over-performance) in win-rate units
def spread(df, label):
    t = df.groupby(["team", "year", "side"]).agg(n=("y", "size"), r=("res", "sum"), v=("p", lambda p: (p * (1 - p)).sum()))
    t = t.unstack("side").dropna()
    t = t[(t["n"]["blue"] >= 15) & (t["n"]["red"] >= 15)]
    d = t["r"]["blue"] / t["n"]["blue"] - t["r"]["red"] / t["n"]["red"]           # blue minus red over-performance
    se2 = t["v"]["blue"] / t["n"]["blue"] ** 2 + t["v"]["red"] / t["n"]["red"] ** 2  # pure-luck variance
    true_var = d.var() - se2.mean()
    print(f"{label}: team-seasons {len(d)}, observed sd {d.std():.3f}, luck sd {np.sqrt(se2.mean()):.3f}, "
          f"implied real sd {np.sqrt(max(true_var, 0)):.3f}")
    return d, se2
print()
dA, _ = spread(L, "All games")
d1, _ = spread(L[L.gnum == 1], "Game 1 only (side not chosen by loser)")

# 3) Persistence: does a team's side split in one year predict next year? (same team id)
x = dA.unstack("year") if isinstance(dA.index, pd.MultiIndex) else None
pairs = []
for (yr1, yr2) in ((2023, 2024), (2024, 2025), (2025, 2026)):
    if yr1 in x and yr2 in x:
        a = x[[yr1, yr2]].dropna(); pairs.append(a.set_axis(["a", "b"], axis=1))
        print(f"{yr1}->{yr2}: {len(a)} teams, correlation {a.corr().iloc[0,1]:+.3f}")
P = pd.concat(pairs); print(f"pooled year-to-year correlation {P.corr().iloc[0,1]:+.3f} (n={len(P)})")

# 4) Out-of-sample test: shrunk per-team side offset learned from all earlier games, applied to each new game
from math import log
def logit(p): return np.log(p / (1 - p))
g = g.sort_values("date").reset_index(drop=True)
for k in (50, 100, 200, 400):           # shrink strength: offset = sum(res) / (n + k)
    st = {}; ll0 = ll1 = 0; N = 0
    for r in g.itertuples(index=False):
        sb = st.get(r.b_teamid, [0.0, 0.0, 0, 0]); sr = st.get(r.r_teamid, [0.0, 0.0, 0, 0])
        # team's blue over-perf minus red over-perf, shrunk; convert win-rate units to logit (x4 ~ 1/(p(1-p)))
        def off(s): return (s[0] / (s[2] + k) - s[1] / (s[3] + k)) / 2 * 4
        delta = off(sb) / 2 - off(sr) / 2        # half from each team: blue team's blue skill, red team's red skill
        p0 = r.p_pre; p1 = 1 / (1 + np.exp(-(logit(p0) + delta)))
        if r.date.year == 2026 and r.b_home_league in TIER1 | {"VCS"} or r.date.year == 2026:
            yy = r.y; ll0 -= log(p0 if yy else 1 - p0); ll1 -= log(p1 if yy else 1 - p1); N += 1
        res = r.y - p0
        sb = [sb[0] + res, sb[1], sb[2] + 1, sb[3]]; sr = [sr[0], sr[1] - res, sr[2], sr[3] + 1]
        st[r.b_teamid] = sb; st[r.r_teamid] = sr
    print(f"shrink k={k}: 2026 log loss base {ll0/N:.5f} -> with team side {ll1/N:.5f} (diff {(ll1-ll0)/N:+.5f}, n={N})")
