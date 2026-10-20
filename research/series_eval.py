import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from series_math import scores, series_prob
d = pd.read_pickle('series_clean.pkl'); S = float(open('series_s.txt').read())
d = d[d.yr >= 2025].copy()
for lab, s in (('old', 0.0), ('new', S)):
    d[f'ps_{lab}'] = [series_prob(r.p, r.bo, s) for r in d.itertuples()]
    d[f'full_{lab}'] = [sum(v for (a, b), v in scores(r.p, r.bo, s).items() if a + b == r.bo) for r in d.itertuples()]
    d[f'sl_{lab}'] = [-np.log(scores(r.p, r.bo, s)[(r.w1, r.w2)]) for r in d.itertuples()]
ll = lambda p, y: -(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f'form-on-the-day spread s = {S}\n')
for yr in (2025, 2026):
    for bo in (3, 5):
        for tier in ('all', 'top tier'):
            g = d[(d.yr == yr) & (d.bo == bo) & ((tier == 'all') | d.tier1)]
            y = g.t1_won.values
            fo, fn = np.maximum(g.ps_old, 1 - g.ps_old), np.maximum(g.ps_new, 1 - g.ps_new)
            fw = np.where(g.ps_old > .5, y, 1 - y)
            dl = ll(g.ps_old.values, y) - ll(g.ps_new.values, y)
            print(f'{yr} Bo{bo} {tier:8} n={len(g):4d} | fav predicted old {fo.mean():.3f} new {fn.mean():.3f} actual {fw.mean():.3f} '
                  f'| winner log-loss gain {dl.mean():+.4f}±{dl.std()/np.sqrt(len(g)):.4f} '
                  f'| full-length pred old {g.full_old.mean():.3f} new {g.full_new.mean():.3f} actual {(g.games == bo).mean():.3f} '
                  f'| scoreline LL gain {(g.sl_old - g.sl_new).mean():+.4f}')
# calibration table for Bo5, 2025-26 combined
g = d[d.bo == 5].copy(); y = g.t1_won.values
for lab in ('old', 'new'):
    g['fav'] = np.maximum(g[f'ps_{lab}'], 1 - g[f'ps_{lab}']); g['fw'] = np.where(g[f'ps_{lab}'] > .5, y, 1 - y)
    g['bin'] = pd.cut(g.fav, [.5, .6, .7, .8, .9, 1])
    print(f'\nBo5 calibration 2025-26, {lab}:'); print(g.groupby('bin', observed=True).agg(n=('fav', 'size'), predicted=('fav', 'mean'), actual=('fw', 'mean')).round(3).to_string())
d.to_pickle('series_eval.pkl')
