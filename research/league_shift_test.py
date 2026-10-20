"""Off-season games and league levels.
(1) Promotion tournaments: a bottom tier-1 team v top tier-2 teams moved the whole LCS level by +0.46.
(2) The Demacia Cup Global Invitational (DCGI) wasn't on the cup/international list, so it counted at full weight.
Variants are scored on international games (where league levels matter most) 2024-26, and on all games."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import sys, numpy as np, pandas as pd, ratings
from ratings import data_file, data_files
from ratings import load_games, RatingEngine, Params, _margin, TIER1

class FreezeTop(RatingEngine):
    """In a game between a tier-1-league team and a lower-tier team, the tier-1 league level is not moved."""
    def update(self, date, blue, red, lb, lr, *a, **k):
        frozen = [("L", l) for l in (lb, lr) if l in TIER1] if ((lb in TIER1) != (lr in TIER1)) else []
        keep = {f: (self.s.mean.get(f), self.s.var.get(f)) for f in frozen if f in self.s.mean}
        p = super().update(date, blue, red, lb, lr, *a, **k)
        for f, (m, v) in keep.items(): self.s.mean[f], self.s.var[f] = m, v
        return p

def score(engine_cls, dcgi_fix, cross_w=1.0):
    if dcgi_fix: ratings.CUPS.add("DCGI"); ratings.INTERNATIONAL.add("DCGI")
    else: ratings.CUPS.discard("DCGI"); ratings.INTERNATIONAL.discard("DCGI")
    G = load_games(data_files())
    eng = engine_cls(Params()); pre = []
    for r in G.itertuples(index=False):
        pre.append(eng.predict(r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, as_of=r.date)[0])
        lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None; lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
        cross = (r.b_home_league in TIER1) != (r.r_home_league in TIER1)
        w = (eng.p.cup_w if r.league in ratings.CUPS else 1.0) * (cross_w if cross else 1.0)
        eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur, _margin(r, "gd15"),
                   intl=r.league in ratings.INTERNATIONAL, margin2=_margin(r, "gspd"), weight=w)
    G["p"] = pre
    L = {l: round(eng.s.mean[("L", l)], 2) for l in ["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL"]}
    return G, L

def ll(d): return -(d.y * np.log(d.p) + (1 - d.y) * np.log(1 - d.p))
variants = [("current", RatingEngine, False, 1.0), ("DCGI as cup+intl", RatingEngine, True, 1.0),
            ("+ freeze tier-1 level in cross-tier games", FreezeTop, True, 1.0),
            ("+ cross-tier games at half weight", RatingEngine, True, 0.5)]
res = {}
for name, cls, fix, cw in variants:
    G, L = score(cls, fix, cw); res[name] = G.p.values
    print(f"{name:45s} levels now {L}")
G["y_"] = G.y
base = res["current"]
yr = G.date.dt.year
masks = {"Worlds/MSI/EWC/FST 2024-26": G.league.isin(["WLDs", "MSI", "EWC", "FST"]) & (yr >= 2024),
         "Worlds only 2024-25": (G.league == "WLDs"), "all games 2024-26": yr >= 2024,
         "top-league domestic 2024-26": G.league.isin(TIER1) & G.b_home_league.isin(TIER1) & G.r_home_league.isin(TIER1) & (yr >= 2024)}
for mname, m in masks.items():
    d = G[m]; l0 = ll(d.assign(p=base[m.values]))
    print(f"\n{mname} (n={m.sum()}): current {l0.mean():.5f}")
    for name in list(res)[1:]:
        l1 = ll(d.assign(p=res[name][m.values])); diff = l1 - l0
        print(f"   {name:45s} {diff.mean():+.5f} ± {diff.std()/np.sqrt(len(diff)):.5f}")
# how much each variant changes the play-in opener
