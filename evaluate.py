"""Headline evaluation. Every game is predicted before it is played (walk-forward through 2023-26),
then 2026 games are scored. Baseline: a standard Elo rating (K=24, 30-point blue-side bonus)."""
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from ratings import load_games, run, data_files, TIER1, INTERNATIONAL

files = data_files()
g = load_games(files)
_, p_model = run(g)

def elo(games, K=24, h=30):
    R, ps = {}, []
    for r in games.itertuples():
        rb, rr = R.get(r.b_teamid, 1500), R.get(r.r_teamid, 1500)
        p = 1 / (1 + 10 ** ((rr - rb - h) / 400)); ps.append(p)
        R[r.b_teamid] = rb + K * (r.y - p); R[r.r_teamid] = rr - K * (r.y - p)
    return np.array(ps)
p_elo = elo(g)

y26 = g.date.dt.year == 2026
top = g.league.isin(TIER1 | INTERNATIONAL)
slices = {"All leagues": y26, "Top leagues + international": y26 & top,
          "International only": y26 & g.league.isin(INTERNATIONAL), "Jan-Mar (new rosters)": y26 & (g.date.dt.month <= 3)}
rows = []
for name, m in slices.items():
    y = g.y[m]
    rows.append(dict(slice=name, games=int(m.sum()),
                     elo_logloss=round(log_loss(y, p_elo[m]), 3), model_logloss=round(log_loss(y, p_model[m]), 3),
                     model_accuracy=f"{((p_model[m] > .5) == y).mean():.1%}", model_auc=round(roc_auc_score(y, p_model[m]), 3)))
print(pd.DataFrame(rows).to_string(index=False))
q, y = p_model[y26], g.y[y26].values
cal = pd.DataFrame({"bin": pd.cut(q, [0, .2, .35, .45, .55, .65, .8, 1]), "p": q, "y": y}).groupby("bin", observed=True).agg(
    games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)
print("\nCalibration, 2026:\n" + cal.to_string())
print(f"\nCoin flip log loss = 0.693. Games in data: {len(g):,}, from {g.date.min():%b %Y} to {g.date.max():%d %b %Y}.")
