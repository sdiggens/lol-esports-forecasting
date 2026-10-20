"""Tune on 2025 predictions (model has seen 2023-24 + 2025-to-date), test on 2026."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import itertools, time, numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, run, Params, INTERNATIONAL, TIER1
t0 = time.time()
import glob
G = load_games(data_files())
print(f"{len(G):,} games loaded in {time.time()-t0:.0f}s")
y = G.y.values
ll = lambda p, m: -(y[m] * np.log(p[m]) + (1 - y[m]) * np.log(1 - p[m])).mean()
y25 = (G.date.dt.year == 2025).values; y26 = (G.date.dt.year == 2026).values
maj = G.league.isin(TIER1 | INTERNATIONAL).values; intl = G.league.isin(INTERNATIONAL).values
early = (G.date.dt.month <= 3).values
grid = dict(season_keep=[0.5, 0.7, 0.85, 1.0], season_var=[0.0, 0.1, 0.25],
            tau_team=[0.0005, 0.001], intl_var0=[0.05, 0.15])
res = []
for vals in itertools.product(*grid.values()):
    prm = Params(**dict(zip(grid, vals)))
    _, p = run(G, prm)
    res.append(dict(zip(grid, vals), val_2025=ll(p, y25), val_2025_major=ll(p, y25 & maj)))
r = pd.DataFrame(res).sort_values("val_2025")
print(r.head(8).round(4).to_string(index=False))
best = Params(**{k: r.iloc[0][k] for k in grid})
print("chosen:", {k: r.iloc[0][k] for k in grid})
_, p_multi = run(G, best)
g26 = load_games([data_file(2026)]); _, p_single = run(g26, Params())
# align single-season predictions to the multi-year frame by gameid
ps = pd.Series(p_single, index=g26.gameid).reindex(G.gameid).values
m26 = y26 & ~np.isnan(ps)
rows = []
for nm, m in [("2026 all", m26), ("2026 tier-1 + intl", m26 & maj), ("2026 international", m26 & intl),
              ("2026 Jan-Mar (new season)", m26 & early), ("2026 Jan-Mar tier-1", m26 & early & maj)]:
    a, b = ll(ps, m), ll(p_multi, m)
    d = -(y[m]*np.log(ps[m])+(1-y[m])*np.log(1-ps[m])) + (y[m]*np.log(p_multi[m])+(1-y[m])*np.log(1-p_multi[m]))
    rows.append(dict(slice=nm, games=int(m.sum()), only_2026=a, with_2023_25=b, gain=a-b, se=d.std()/np.sqrt(m.sum())))
print(pd.DataFrame(rows).round(4).to_string(index=False))
