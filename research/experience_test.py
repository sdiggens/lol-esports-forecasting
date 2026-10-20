"""Does lineup experience (career pro games in the data) predict results beyond the model?
Fit one coefficient on 2025, test on 2026 (and the reverse), pre-game information only."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from ratings import data_file, data_files
from scipy.optimize import minimize_scalar
G = pd.read_pickle("G_pre.pkl")
cols = ["gameid", "date", "side", "position", "playername", "participantid", "league"]
raw = pd.concat([pd.read_csv(data_file(y), usecols=cols, low_memory=False) for y in (2023, 2024, 2025, 2026)])
raw = raw.drop_duplicates(["gameid", "participantid"])
pl = raw[raw.position != "team"].copy(); pl["date"] = pd.to_datetime(pl.date)
pl = pl.sort_values("date"); pl["career"] = pl.groupby("playername").cumcount()   # games before this one (2023 onwards)
pl["lc"] = np.log1p(pl.career.clip(upper=400))
team = pl.groupby(["gameid", "side"]).lc.mean().unstack()
G = G.join(team, on="gameid").dropna(subset=["Blue", "Red"])
G["x"] = G.Blue - G.Red
G["z0"] = np.log(G.p / (1 - G.p))
from ratings import TIER1
def ll(b, d): q = 1 / (1 + np.exp(-(d.z0 + b * d.x))); return -np.mean(d.y * np.log(q) + (1 - d.y) * np.log(1 - q))
def run(train, test, label):
    b = minimize_scalar(lambda b: ll(b, train), bounds=(-2, 2), method="bounded").x
    base, new = ll(0, test), ll(b, test)
    d = [];  # paired se
    q0 = test.p; q1 = 1 / (1 + np.exp(-(test.z0 + b * test.x)))
    l0 = -(test.y * np.log(q0) + (1 - test.y) * np.log(1 - q0)); l1 = -(test.y * np.log(q1) + (1 - test.y) * np.log(1 - q1))
    se = (l1 - l0).std() / np.sqrt(len(test))
    print(f"{label}: coef {b:+.3f}  test log loss {base:.5f} -> {new:.5f}  ({new-base:+.5f} ± {se:.5f}, n={len(test)})")
    return b
yr = G.date.dt.year
b = run(G[yr == 2025], G[yr == 2026], "all leagues   fit 2025, test 2026")
run(G[yr == 2026], G[yr == 2025], "all leagues   fit 2026, test 2025")
top = G.b_home_league.isin(TIER1) & G.r_home_league.isin(TIER1)
run(G[(yr == 2025) & top], G[(yr == 2026) & top], "top leagues   fit 2025, test 2026")
# Jan-Mar (new rosters) vs rest of 2026
t = G[yr == 2026]
for lab, m in (("2026 Jan-Mar", t.date.dt.month <= 3), ("2026 Apr-Oct", t.date.dt.month > 3)):
    s = t[m]; q1 = 1 / (1 + np.exp(-(s.z0 + b * s.x)))
    print(f"  {lab}: {ll(0, s):.5f} -> {ll(b, s):.5f}")
# what does the coefficient mean in plain terms?
print(f"\nExample: a lineup averaging 300 career games v one averaging 50: shift of {b*(np.log1p(300)-np.log1p(50)):+.2f} logit")
print("Typical |x| in top-league games:", G[top & (yr == 2026)].x.abs().median().round(3))
