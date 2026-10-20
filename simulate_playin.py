"""Worlds 2026 Play-In: 4 teams, double elimination, every series Bo5, one team goes to Swiss.
Round 1 (random draw): FURIA v MVK, Cloud9 v Karmine Corp.
Each simulation draws every team's true strength once from the model's belief (so a team that is
secretly strong is strong in all its matches), then plays the bracket with the 'form on the day' series maths."""
import glob, numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, run, series_spread, series_prob, series_scores
from series_math import series_prob as sp

files = data_files()
g = load_games(files); eng, _ = run(g)
AS_OF = pd.Timestamp("2026-10-15")
WANT = {"FURIA": "CBLOL", "MVK Esports": "LCP", "Cloud9": "LCS", "Karmine Corp": "LEC"}
T = {}
for name, lg in WANT.items():
    for side in "br":
        r = g[(g[f"{side}_teamname"] == name)]
        if len(r): T[name] = dict(id=r[f"{side}_teamid"].iloc[-1], league=lg); break
print({k: v["id"] for k, v in T.items()})

def belief(key):
    return eng._mean_at(key, AS_OF), eng._var_at(key, AS_OF)

names = list(T)
rng = np.random.default_rng(2026); N = 200_000
H = eng.p.side_adv
# draw strengths: team + league, each Normal(mean, var)
S = {}
for n in names:
    mt, vt = belief(("T", T[n]["id"])); ml, vl = belief(("L", T[n]["league"]))
    S[n] = rng.normal(mt, np.sqrt(vt), N) + rng.normal(ml, np.sqrt(vl), N)
    print(f"{n:13s} team {mt:+.2f} (sd {np.sqrt(vt):.2f})  league {T[n]['league']} {ml:+.2f} (sd {np.sqrt(vl):.2f})")

# Bo5 series-win probability as a function of single-game p (all four are top-league teams: s = 0.7)
s = series_spread("LEC", "LCS")
grid = np.linspace(0.001, 0.999, 999); sgrid = np.array([sp(p, 5, s) for p in grid])
def game_p(a, b):  # average of a on blue and a on red, given drawn strengths (no extra shrink: uncertainty is in the draws)
    d = S[a] - S[b]; return 0.5 * (1 / (1 + np.exp(-(d + H))) + 1 / (1 + np.exp(-(d - H))))
def play(a, b, idx_a, idx_b):
    """a, b: arrays of team-name indices per sim. Returns winner/loser index arrays."""
    D = np.stack([S[n] for n in names])            # 4 x N
    cols = np.arange(N)
    d = D[a, cols] - D[b, cols]
    p = 0.5 * (1 / (1 + np.exp(-(d + H))) + 1 / (1 + np.exp(-(d - H))))
    pw = np.interp(p, grid, sgrid)
    aw = rng.random(N) < pw
    return np.where(aw, a, b), np.where(aw, b, a)

ix = {n: i for i, n in enumerate(names)}
full = lambda n: np.full(N, ix[n])
w1, l1 = play(full("FURIA"), full("MVK Esports"), None, None)
w2, l2 = play(full("Cloud9"), full("Karmine Corp"), None, None)
uw, ul = play(w1, w2, None, None)         # upper final
lw, ll = play(l1, l2, None, None)         # lower round 1 (loser out)
dw, dl = play(ul, lw, None, None)         # lower final
gw, gl = play(uw, dw, None, None)         # grand final -> Swiss (no bracket reset assumed)

out = pd.DataFrame({n: {"Qualify for Swiss": (gw == ix[n]).mean(),
                        "Reach grand final": ((gw == ix[n]) | (gl == ix[n])).mean(),
                        "Win round 1": ((w1 == ix[n]) | (w2 == ix[n])).mean(),
                        "Out 0-2": ((ll == ix[n])).mean()} for n in names}).T
out = out.sort_values("Qualify for Swiss", ascending=False)
print("\n", (out * 100).round(1).to_string())

# head-to-head table using the model's normal (shrunk) single-game probability, for pricing each possible series
print("\nHead-to-head Bo5 (row beats column), single game in brackets:")
rows = {}
for a in names:
    rows[a] = {}
    for b in names:
        if a == b: rows[a][b] = ""; continue
        ta, tb = T[a], T[b]
        pb = eng.predict(ta["id"], tb["id"], ta["league"], tb["league"], as_of=AS_OF)[0]
        pr = 1 - eng.predict(tb["id"], ta["id"], tb["league"], ta["league"], as_of=AS_OF)[0]
        p = (pb + pr) / 2
        rows[a][b] = f"{100*series_prob(p, 5, s):.0f}% ({100*p:.0f}%)"
print(pd.DataFrame(rows).T.to_string())
for a, b in (("FURIA", "MVK Esports"), ("Cloud9", "Karmine Corp")):
    ta, tb = T[a], T[b]
    p = (eng.predict(ta["id"], tb["id"], ta["league"], tb["league"], as_of=AS_OF)[0]
         + 1 - eng.predict(tb["id"], ta["id"], tb["league"], ta["league"], as_of=AS_OF)[0]) / 2
    sc = series_scores(p, 5, s)
    print(f"\n{a} v {b}: P({a}) = {100*series_prob(p,5,s):.1f}%  fair odds {1/series_prob(p,5,s):.2f} / {1/(1-series_prob(p,5,s)):.2f}")
    print("  " + "  ".join(f"{x}-{y}: {100*q:.1f}%" for (x, y), q in sorted(sc.items(), key=lambda kv: kv[0][1]-kv[0][0])))
out.to_csv("playin_sim.csv")
