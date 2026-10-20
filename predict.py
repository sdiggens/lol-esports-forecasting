"""
Price a match and check it against bookmaker odds.

  python predict.py rankings                       # current power rankings
  python predict.py match "Gen.G" "Bilibili Gaming" --bo 5 --odds 1.45 2.75
  python predict.py match T1 G2 --bo 1 --odds 4/9 7/4 --bankroll 500
  python predict.py match T1 "Gen.G" --bo 5 --context intl   # same-region teams at Worlds

Odds can be decimal (2.10), fractional (11/10) or American (+110 / -150).
"""
import argparse, glob, math
from ratings import data_file, data_files
from fractions import Fraction
import pandas as pd
from ratings import load_games, run, series_prob, series_scores, series_spread, sigmoid, INTERNATIONAL, TIER1


def parse_odds(s):
    s = str(s).strip()
    if "/" in s:
        return float(Fraction(s)) + 1
    v = float(s)
    if s.startswith("+") or v >= 100:
        return v / 100 + 1
    if v <= -100:
        return 100 / abs(v) + 1
    return v


def devig(d1, d2):
    """Remove the bookmaker margin. Returns (fair p1, fair p2, overround).
    Uses the 'power' method, which corrects for favourite-longshot bias better
    than simple normalisation: find k with (1/d1)^k + (1/d2)^k = 1."""
    i1, i2 = 1 / d1, 1 / d2
    lo, hi = 0.5, 3.0
    for _ in range(100):
        k = (lo + hi) / 2
        if i1 ** k + i2 ** k > 1: lo = k
        else: hi = k
    return i1 ** k, i2 ** k, i1 + i2 - 1


def build(paths):
    g = load_games(paths)
    eng, _ = run(g)
    names = (pd.concat([g[["date", "b_teamid", "b_teamname", "b_home_league"]].set_axis(["date", "id", "name", "lg"], axis=1),
                        g[["date", "r_teamid", "r_teamname", "r_home_league"]].set_axis(["date", "id", "name", "lg"], axis=1)])
             .sort_values("date").groupby("id").last())
    return eng, names


def find(names, q):
    q = q.lower().strip()
    exact = names[names.name.str.lower() == q]
    hits = exact if len(exact) else names[names.name.str.lower().str.contains(q, regex=False)]
    if len(hits) == 0:
        raise SystemExit(f"No team matching '{q}'")
    hits = hits.assign(_t=~hits.lg.isin(TIER1), _l=hits["name"].str.len())
    hit = hits.sort_values(["_t", "_l", "date"], ascending=[True, True, False]).iloc[0]
    return hit.name, hit["name"], hit.lg


def game_prob(eng, a, b, intl, neutral_side=True):
    """P(a wins a game). With neutral_side, averages a-on-blue and a-on-red."""
    now = pd.Timestamp.now()
    pb, m1, v = eng.predict(a[0], b[0], a[2], b[2], intl=intl, as_of=now)
    pr = 1 - eng.predict(b[0], a[0], b[2], a[2], intl=intl, as_of=now)[0]
    m = m1 - eng.p.side_adv
    lo = sigmoid((m - 1.2816 * math.sqrt(v)))
    hi = sigmoid((m + 1.2816 * math.sqrt(v)))
    return ((pb + pr) / 2 if neutral_side else pb), pb, pr, lo, hi


def cmd_rankings(eng, names, top, all_leagues=False):
    rows = [(r["name"], r.lg, eng.strength(tid, r.lg), eng.s.mean.get(("I", tid), 0.0),
             eng.strength(tid, r.lg) + eng.s.mean.get(("I", tid), 0.0),
             math.sqrt(eng.s.var.get(("T", tid), 0)))
            for tid, r in names.iterrows() if ("T", tid) in eng.s.mean and (all_leagues or r.lg in TIER1)]
    df = pd.DataFrame(rows, columns=["team", "league", "domestic", "big_stage_offset", "international",
                                     "uncertainty"]).sort_values("international", ascending=False)
    print(df.head(top).round(3).to_string(index=False))
    lv = sorted(((k[1], v) for k, v in eng.s.mean.items() if k[0] == "L" and k[1] in TIER1), key=lambda t: -t[1])
    print("\nTier-1 league levels:", ", ".join(f"{l} {v:+.2f}" for l, v in lv))


def cmd_match(eng, names, a_q, b_q, bo, odds, bankroll, kelly_frac, min_edge, context):
    a, b = find(names, a_q), find(names, b_q)
    intl = {"intl": True, "domestic": False}.get(context, a[2] != b[2])
    p, pb, pr, lo, hi = game_prob(eng, a, b, intl)
    spread = series_spread(a[2], b[2])
    ps = series_prob(p, bo, spread) if bo > 1 else p
    ctx = ("INTERNATIONAL" + (" (big-stage offsets on)" if eng.p.intl_var0 > 0 else "")) if intl else "DOMESTIC"
    print(f"\n{a[1]} ({a[2]})  vs  {b[1]} ({b[2]})   Bo{bo}   {ctx}")
    print(f"  per-game: {p:.1%}  (on blue {pb:.1%}, on red {pr:.1%}; 80% band {lo:.1%}-{hi:.1%})")
    if bo > 1:
        print(f"  series  : {ps:.1%}   fair odds {1/ps:.2f} / {1/(1-ps):.2f}   (form-on-the-day spread {spread})")
        sc = sorted(series_scores(p, bo, spread).items(), key=lambda kv: -(kv[0][0] - kv[0][1]))
        print("  scores  : " + "  ".join(f"{x}-{y} {v:.1%}" for (x, y), v in sc))
    if not odds:
        return
    d1, d2 = map(parse_odds, odds)
    f1, f2, over = devig(d1, d2)
    print(f"\n  bookmaker: {d1:.2f} / {d2:.2f}  margin {over:.1%}  ->  market fair {f1:.1%} / {f2:.1%}")
    for name, pm, d, fm in [(a[1], ps, d1, f1), (b[1], 1 - ps, d2, f2)]:
        ev = pm * d - 1
        kelly = max((pm * d - 1) / (d - 1), 0) * kelly_frac
        flag = "VALUE" if ev > min_edge else ""
        if flag and abs(pm - fm) > 0.12:
            flag = "CHECK: model and market differ by >12 pts - look for news/roster info before trusting this"
        print(f"  {name:<24} model {pm:6.1%}  market {fm:6.1%}  EV {ev:+6.1%}  "
              f"stake {kelly:5.1%} of bankroll" + (f" = £{kelly*bankroll:.2f}" if bankroll else "") + f"  {flag}")
    print(f"\n  (stakes are {kelly_frac:.0%} Kelly; flagged when EV > {min_edge:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rankings", "match"])
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--data", nargs="*", default=data_files())
    ap.add_argument("--bo", type=int, default=1)
    ap.add_argument("--odds", nargs=2)
    ap.add_argument("--bankroll", type=float, default=0)
    ap.add_argument("--kelly", type=float, default=0.25)
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--all-leagues", action="store_true")
    ap.add_argument("--context", choices=["auto", "intl", "domestic"], default="auto",
                    help="auto = international if the teams are from different leagues. "
                         "Use --context intl for same-region teams meeting at Worlds/MSI.")
    a = ap.parse_args()
    eng, names = build(a.data)
    if a.cmd == "rankings":
        cmd_rankings(eng, names, a.top, a.all_leagues)
    else:
        cmd_match(eng, names, a.teams[0], a.teams[1], a.bo, a.odds, a.bankroll, a.kelly, a.min_edge, a.context)
