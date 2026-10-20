"""Out-of-sample test: fixed form-on-the-day spread vs one re-estimated from a trailing window."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from series_math import scores
d = pd.read_pickle('series_clean.pkl').sort_values('start').reset_index(drop=True)
GRID = np.round(np.arange(0, 1.61, 0.1), 2)
# per-series negative log-likelihood of the exact scoreline at each candidate spread
NLL = np.array([[-np.log(max(scores(r.p, r.bo, s)[(r.w1, r.w2)], 1e-12)) for s in GRID] for r in d.itertuples()])
np.save('series_nll_grid.npy', NLL); d.to_pickle('series_clean_sorted.pkl')
t = d.start.values; tier = d.tier1.values
def adaptive(days, prior_n=0, prior_s=None):
    """s chosen by max likelihood on same-tier series in the trailing window (strictly before)."""
    out = np.full(len(d), np.nan)
    for i in range(len(d)):
        m = (t < t[i]) & (t >= t[i] - np.timedelta64(days, 'D')) & (tier == tier[i])
        tot = NLL[m].sum(0)
        if prior_n:   # shrink towards a prior spread, worth prior_n series
            tot = tot + prior_n * (GRID - prior_s[tier[i]]) ** 2 * 0.5
        out[i] = GRID[int(np.argmin(tot))] if m.sum() >= 50 else (0.7 if tier[i] else 1.0)
    return out
test = (d.yr >= 2025).values
fixed = np.where(tier, 0.7, 1.0)
cands = {'fixed 0.7/1.0 (current)': fixed, 'independent (s=0)': np.zeros(len(d))}
for days in (180, 365):
    cands[f'adaptive, last {days} days'] = adaptive(days)
def nll_at(svec):
    idx = np.searchsorted(GRID, np.round(svec, 2)); return NLL[np.arange(len(d)), idx]
base = nll_at(fixed)
print('Scoreline log loss (lower = better), gain vs current fixed setting:')
for nm, sv in cands.items():
    for lab, m in (('2025-26 all', test), ('2025-26 top', test & tier), ('2026 top', (d.yr == 2026).values & tier), ('2025-26 lower', test & ~tier)):
        x = base[m] - nll_at(sv)[m]
        print(f'  {nm:28} {lab:13} n={m.sum():5d} gain {x.mean():+.4f} ± {x.std()/np.sqrt(m.sum()):.4f}')
for nm in ('adaptive, last 180 days', 'adaptive, last 365 days'):
    sv = cands[nm]; s = pd.Series(sv[test]); print(f'\n{nm}: spread used in 2025-26 — top tier range {sv[test & tier].min()}–{sv[test & tier].max()}, '
          f'lower {sv[test & ~tier].min()}–{sv[test & ~tier].max()}')
    q = pd.DataFrame({'start': d.start[test], 'tier': tier[test], 's': sv[test]}); q['q'] = q.start.dt.to_period('Q')
    print(q.groupby(['q', 'tier']).s.mean().unstack().round(2).to_string())
pd.to_pickle(cands, 'adaptive_cands.pkl')
