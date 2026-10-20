"""Test the international offset and playoff weighting. Every prediction is made
BEFORE the game is processed, so all scores are out-of-sample."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import itertools, numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, run, Params, INTERNATIONAL, TIER1
g = load_games([data_file(2026)])
intl = g.league.isin(INTERNATIONAL).values
dom = (~intl) & (g.date >= g.date.quantile(0.4)).values
dom_t1 = dom & g.league.isin(TIER1).values
y = g.y.values
ll = lambda p, m: -(y[m] * np.log(p[m]) + (1 - y[m]) * np.log(1 - p[m]))
_, base = run(g, Params())
rows = []
for iv, pw in itertools.product([0.0, 0.05, 0.1, 0.2, 0.4], [1.0, 1.5, 2.0, 3.0]):
    _, p = run(g, Params(intl_var0=iv, playoff_w=pw))
    d_i = ll(p, intl) - ll(base, intl)
    d_d = ll(p, dom_t1) - ll(base, dom_t1)
    rows.append(dict(intl_var0=iv, playoff_w=pw,
                     intl_ll=ll(p, intl).mean(), intl_gain=-d_i.mean(), intl_se=d_i.std() / np.sqrt(len(d_i)),
                     dom_ll=ll(p, dom_t1).mean(), dom_gain=-d_d.mean(), dom_se=d_d.std() / np.sqrt(len(d_d))))
r = pd.DataFrame(rows).round(4)
print(f"international games: {intl.sum()}, tier-1 domestic games (2nd half+): {dom_t1.sum()}")
print(r.to_string(index=False))
r.to_csv("tune_weights_results.csv", index=False)
