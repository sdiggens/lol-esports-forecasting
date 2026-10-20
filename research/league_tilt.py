"""League 'tilt': a league's level can differ between its top and its bottom.
strength_i = theta_i + mu_L + c_i * delta_L,   c_i = 0.5 * tanh(theta_i / SCALE)  (about -0.5 bottom ... +0.5 top)
mu_L is the league's overall level; delta_L is how much stronger the league's top is, relative to other leagues,
than its bottom. A bottom team's cross-league games (c ~ -0.4) mostly inform the bottom of the league;
a top team's games mostly inform the top. Only used when the two teams are from different leagues."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import math, numpy as np, pandas as pd, ratings
from ratings import data_file, data_files
from ratings import RatingEngine, Params, load_games, _margin, TIER1, series_prob

class TiltEngine(RatingEngine):
    TILT_VAR, SCALE, TILT_TAU = 0.15, 0.7, 0.0002
    def _c(self, team):
        return 0.5 * math.tanh(self.s.mean.get(("T", team), 0.0) / self.SCALE)
    def _components(self, blue, red, lb, lr, intl=False):
        comps = super()._components(blue, red, lb, lr, intl)
        if lb != lr and self.TILT_VAR > 0:
            comps += [(("D", lb), self._c(blue)), (("D", lr), -self._c(red))]
        return comps
    def _default_var(self, k):
        return self.TILT_VAR if k[0] == "D" else super()._default_var(k)
    def _var_at(self, k, as_of):
        if k[0] != "D": return super()._var_at(k, as_of)
        v = self.s.var.get(k, self.TILT_VAR)
        if as_of is None or k not in self.s.last_seen: return v
        return min(v + self.TILT_TAU * max((as_of - self.s.last_seen[k]).total_seconds() / 86400, 0), self.TILT_VAR)
    def _ensure_tilt(self, key, date):
        if key not in self.s.mean:
            self.s.mean[key], self.s.var[key] = 0.0, self.TILT_VAR
        else:
            days = max((date - self.s.last_seen[key]).total_seconds() / 86400, 0)
            self.s.var[key] = min(self.s.var[key] + self.TILT_TAU * days, self.TILT_VAR)
        self.s.last_seen[key] = date
    def update(self, date, blue, red, lb, lr, *a, **k):
        if lb != lr and self.TILT_VAR > 0:
            self._ensure_tilt(("D", lb), date); self._ensure_tilt(("D", lr), date)
        return super().update(date, blue, red, lb, lr, *a, **k)

def run_engine(G, eng):
    pre = []
    for r in G.itertuples(index=False):
        pre.append(eng.predict(r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, as_of=r.date)[0])
        lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None; lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
        eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur, _margin(r, "gd15"),
                   intl=r.league in ratings.INTERNATIONAL, margin2=_margin(r, "gspd"),
                   weight=eng.p.cup_w if r.league in ratings.CUPS else 1.0)
    return np.array(pre)

def c9kc(eng, G):
    f = lambda n: next((G[G[s + "_teamname"] == n][s + "_teamid"].iloc[-1], G[G[s + "_teamname"] == n][s + "_home_league"].iloc[-1]) for s in "br" if (G[s + "_teamname"] == n).any())
    (a, la), (b, lb) = f("Cloud9"), f("Karmine Corp"); t = pd.Timestamp("2026-10-15")
    p = (eng.predict(a, b, la, lb, as_of=t)[0] + 1 - eng.predict(b, a, lb, la, as_of=t)[0]) / 2
    return p, series_prob(p, 5, 0.7)

if __name__ == "__main__":
    G = load_games(data_files())
    yr = G.date.dt.year.values
    cross_top = (G.b_home_league.isin(TIER1) & G.r_home_league.isin(TIER1) & (G.b_home_league != G.r_home_league)).values
    masks = {"intl top v top 2024-25": cross_top & ((yr == 2024) | (yr == 2025)),
             "intl top v top 2026": cross_top & (yr == 2026),
             "all games 2024-26": yr >= 2024}
    ll = lambda p, m: -(G.y.values[m] * np.log(p[m]) + (1 - G.y.values[m]) * np.log(1 - p[m]))
    base_eng = RatingEngine(Params()); base = run_engine(G, base_eng)
    p, ps = c9kc(base_eng, G)
    print(f"current model: C9 v KC game {p:.3f} series {ps:.3f}")
    for m, v in masks.items(): print(f"   {m:25s} n={v.sum():5d} log loss {ll(base, v).mean():.5f}")
    for tv in (0.05, 0.15, 0.4):
        for sc in (0.5, 1.0):
            TiltEngine.TILT_VAR, TiltEngine.SCALE = tv, sc
            eng = TiltEngine(Params()); pr = run_engine(G, eng)
            p, ps = c9kc(eng, G)
            tilts = {l: round(eng.s.mean.get(("D", l), 0), 2) for l in ["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL"]}
            print(f"\ntilt var {tv}, scale {sc}: C9 v KC game {p:.3f} series {ps:.3f}  LCS level {eng.s.mean[('L','LCS')]:+.2f}  tilts {tilts}")
            for m, v in masks.items():
                d = ll(pr, v) - ll(base, v)
                print(f"   {m:25s} {d.mean():+.5f} ± {d.std()/np.sqrt(v.sum()):.5f}")
