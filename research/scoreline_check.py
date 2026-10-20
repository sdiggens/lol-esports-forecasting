"""Exact scoreline calibration: predicted vs actual, with standard errors."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import sys, numpy as np, pandas as pd
from series_math import scores
d = pd.read_pickle('series_clean.pkl')
yrs = [int(y) for y in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['2025', '2026'])]
d = d[d.yr.isin(yrs)].copy()
d['s'] = np.where(d.tier1, 0.7, 1.0)
# orient every series from the favourite's point of view
d['pf'] = np.maximum(d.p, 1 - d.p)
d['fw'] = np.where(d.p >= 0.5, d.w1, d.w2); d['uw'] = np.where(d.p >= 0.5, d.w2, d.w1)
def table(g, bo, label):
    w = bo // 2 + 1
    keys = [(w, j) for j in range(w)] + [(j, w) for j in range(w - 1, -1, -1)]
    pred = np.array([[scores(r.pf, bo, r.s).get(k, 0) for k in keys] for r in g.itertuples()])
    act = np.array([[1.0 if (r.fw, r.uw) == k else 0.0 for k in keys] for r in g.itertuples()])
    n = len(g); P, A = pred.mean(0), act.mean(0); se = np.sqrt(P * (1 - P) / n)
    z = (A - P) / se
    print(f"\n{label}  n={n}   (favourite's score first; z = (actual-predicted)/s.e.)")
    print('   ' + '  '.join(f'{a}-{b}'.rjust(14) for a, b in keys))
    print('   ' + '  '.join(f'{100*p:5.1f}→{100*a:5.1f}%'.rjust(14) for p, a in zip(P, A)))
    print('   ' + '  '.join(f'z={zz:+.1f}'.rjust(14) for zz in z))
    full = [i for i, (a, b) in enumerate(keys) if a + b == bo]
    pf, af = P[full].sum(), A[full].sum(); sef = np.sqrt(pf * (1 - pf) / n)
    print(f'   goes the full {bo}: predicted {100*pf:.1f}%  actual {100*af:.1f}%  (z={(af-pf)/sef:+.1f})')
    return keys, P, A
for bo in (3, 5):
    for tier, sub in (('TOP TIER', d[d.tier1]), ('LOWER TIER', d[~d.tier1])):
        g = sub[sub.bo == bo]
        table(g, bo, f'Bo{bo} {tier} ALL')
        for lo, hi, nm in ((0.5, 0.6, 'even (fav 50-60%)'), (0.6, 0.75, 'clear fav (60-75%)'), (0.75, 1.01, 'heavy fav (75%+)')):
            gg = g[(g.pf >= lo) & (g.pf < hi)]
            if len(gg) >= 60:
                table(gg, bo, f'Bo{bo} {tier} {nm}')
