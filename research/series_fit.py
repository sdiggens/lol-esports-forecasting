import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from series_math import scores, series_prob
d = pd.read_pickle('series_table.pkl'); d['yr'] = d.start.dt.year
b3 = d[d.bo == 3].groupby(['league', 'yr']).agg(n=('games', 'size'), full=('games', lambda s: (s == 3).mean()))
bo2 = set(b3[(b3.n >= 15) & (b3.full < 0.12)].index)
d = d[~d.apply(lambda r: (r.league, r.yr) in bo2, axis=1)]
d = d[d.bo.isin([3, 5])].copy()
d.to_pickle('series_clean.pkl')

def nll(sub, s):
    tot = 0.0
    for r in sub.itertuples():
        sc = scores(r.p, r.bo, s)
        tot -= np.log(max(sc[(r.w1, r.w2)], 1e-12))
    return tot / len(sub)

fit = d[d.yr <= 2024]
print('Fitting on 2023-24:', len(fit), 'Bo3/Bo5 series')
grid = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
res = {s: nll(fit, s) for s in grid}
for s, v in res.items(): print(f'  s={s:.2f}  scoreline log-loss {v:.4f}')
best = min(res, key=res.get)
fine = np.arange(max(best - 0.2, 0), best + 0.21, 0.05)
res2 = {round(s, 2): nll(fit, s) for s in fine}
s_hat = min(res2, key=res2.get)
print('best s =', s_hat, '| by format (2023-24):',
      {bo: min(np.arange(0, 1.6, 0.1), key=lambda s: nll(fit[fit.bo == bo], s)).round(2) for bo in (3, 5)})
open('series_s.txt', 'w').write(str(s_hat))
