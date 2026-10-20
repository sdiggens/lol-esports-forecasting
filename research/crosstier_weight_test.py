"""Count games between a top-league team and a lower-tier team as a fraction of a game (cross_w).
Usage: python crosstier_weight_test.py 0.5 0.25 0.1 0
Scores top-v-top international games 2024-25 and 2026, and all games."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import sys
import numpy as np, pandas as pd, ratings
from ratings import data_files
from ratings import RatingEngine, Params, load_games, _margin, TIER1
from league_tilt import c9kc
G = load_games(data_files())
cross = ((G.b_home_league.isin(TIER1)) != (G.r_home_league.isin(TIER1))).values
yr = G.date.dt.year.values
top = (G.b_home_league.isin(TIER1) & G.r_home_league.isin(TIER1) & (G.b_home_league != G.r_home_league)).values
def go(cw):
    eng = RatingEngine(Params(cross_w=1.0)); pre = []
    for i, r in enumerate(G.itertuples(index=False)):
        pre.append(eng.predict(r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, as_of=r.date)[0])
        lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None; lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
        w = (eng.p.cup_w if r.league in ratings.CUPS else 1.0) * (cw if cross[i] else 1.0)
        if w > 0:
            eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur, _margin(r, "gd15"),
                       intl=r.league in ratings.INTERNATIONAL, margin2=_margin(r, "gspd"), weight=w)
    return eng, np.array(pre)
ll = lambda p, m: -(G.y.values[m] * np.log(p[m]) + (1 - G.y.values[m]) * np.log(1 - p[m]))
_, base = go(1.0)
print("cross-tier share of all games:", cross.mean().round(3))
for cw in tuple(float(x) for x in sys.argv[1:]):
    eng, pr = go(cw); p, ps = c9kc(eng, G)
    out = f"cross-tier weight {cw}: C9 v KC series {ps:.3f}  LCS {eng.s.mean[('L','LCS')]:+.2f} LEC {eng.s.mean[('L','LEC')]:+.2f} |"
    for name, m in (("intl top 24-25", top & ((yr == 2024) | (yr == 2025))), ("intl top 26", top & (yr == 2026)), ("all 24-26", yr >= 2024)):
        d = ll(pr, m) - ll(base, m); out += f" {name} {d.mean():+.5f}±{d.std()/np.sqrt(m.sum()):.5f}"
    print(out)
