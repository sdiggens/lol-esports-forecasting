"""
Model vs the betting market.

For every series that has odds history in odds_raw/hist/:
  * model probability   = series probability from the ratings just before game 1
  * market probability  = Pinnacle's de-vigged price (power method), at the OPEN and at the CLOSE
                          (close = last price before the scheduled start; later prices are in-play)
Then three tests:
  1. Log loss of model vs Pinnacle close on the series result (who forecasts better?)
  2. Blend: does mixing the model into Pinnacle's close improve on Pinnacle alone?
  3. Line movement: when the model disagrees with the opening price, does the price move
     towards the model by the close? (Closing-line value - the standard test of a real edge.)
Plus a paper-betting log at bet365 / Pinnacle closing prices.

Run:  python market_test.py
"""
import glob, json, math, re, difflib
from ratings import data_file, data_files
from pathlib import Path
import numpy as np, pandas as pd
from ratings import load_games, RatingEngine, Params, series_prob, series_spread

RAW = Path("odds_raw")
MIN_EDGE = 0.03


# ---------------------------------------------------------------- model side
def model_series_table(paths):
    """One row per series in the Oracle's Elixir data, with the model's pre-series per-game p."""
    g = load_games(paths)
    eng = RatingEngine(Params())
    pre = {}
    for r in g.itertuples(index=False):
        if r.game_no == 1:
            pb = eng.predict(r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league)[0]
            pr = 1 - eng.predict(r.r_teamid, r.b_teamid, r.r_home_league, r.b_home_league)[0]
            pre[r.gameid] = (pb + pr) / 2            # P(blue-side team of game 1 wins a game), side-neutral
        lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None
        lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
        eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur,
                   float(r.gd15) if pd.notna(r.gd15) else None)
    # stitch games into series: consecutive games between the same pair, starting at game 1
    g = g.assign(pair=[tuple(sorted((a, b))) for a, b in zip(g.b_teamid, g.r_teamid)])
    series, open_ = [], {}
    for r in g.itertuples(index=False):
        winner = r.b_teamid if r.y == 1 else r.r_teamid
        if r.game_no == 1 or r.pair not in open_:
            s = dict(start=r.date, league=r.league, t1=r.b_teamid, t1_name=r.b_teamname,
                     t1_lg=r.b_home_league, t2_lg=r.r_home_league,
                     t2=r.r_teamid, t2_name=r.r_teamname, p_game=pre.get(r.gameid, np.nan),
                     wins={r.b_teamid: 0, r.r_teamid: 0}, games=0)
            series.append(s); open_[r.pair] = s
        s = open_[r.pair]
        s["wins"][winner] = s["wins"].get(winner, 0) + 1
        s["games"] += 1
    rows = []
    for s in series:
        w1, w2 = s["wins"].get(s["t1"], 0), s["wins"].get(s["t2"], 0)
        need = max(w1, w2)
        bo = {1: 1, 2: 3, 3: 5}.get(need)
        if bo is None or np.isnan(s["p_game"]):
            continue
        rows.append(dict(start=s["start"], league=s["league"], t1_name=s["t1_name"], t2_name=s["t2_name"],
                         t1=s["t1"], t2=s["t2"], t1_lg=s["t1_lg"], t2_lg=s["t2_lg"],
                         bo=bo, t1_won=int(w1 > w2), p_game=s["p_game"],
                         p_model=series_prob(s["p_game"], bo, series_spread(s["t1_lg"], s["t2_lg"])) if bo > 1 else s["p_game"]))
    return pd.DataFrame(rows)


def model_at_times(paths, queries):
    """queries: list of (timestamp, t1, t2, lg1, lg2). Returns side-neutral per-game P(t1 wins)
    using ONLY games played before each timestamp."""
    g = load_games(paths)
    eng = RatingEngine(Params())
    order = sorted(range(len(queries)), key=lambda i: queries[i][0])
    out, qi = [np.nan] * len(queries), 0
    def answer(upto):
        nonlocal qi
        while qi < len(order) and queries[order[qi]][0] <= upto:
            ts, a, b, la, lb = queries[order[qi]]
            pa = eng.predict(a, b, la, lb, as_of=ts)[0]; pb = 1 - eng.predict(b, a, lb, la, as_of=ts)[0]
            out[order[qi]] = (pa + pb) / 2
            qi += 1
    for r in g.itertuples(index=False):
        answer(r.date)
        lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None
        lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
        eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur,
                   float(r.gd15) if pd.notna(r.gd15) else None)
    answer(pd.Timestamp.max)
    return out


# ---------------------------------------------------------------- market side
def devig(o1, o2):
    i1, i2 = 1 / o1, 1 / o2
    lo, hi = 0.5, 3.0
    for _ in range(80):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if i1 ** k + i2 ** k > 1 else (lo, k)
    return i1 ** k


def _hist(hist, book, outcome):
    try:
        h = next(iter(hist["bookmakers"][book]["markets"]["181"]["outcomes"][outcome]["players"].values()))
    except (KeyError, StopIteration):
        return {}
    # only ACTIVE prices: inactive entries include suspended/glitched quotes with the sides swapped
    return {x["createdAt"]: x["price"] for x in h if x.get("price") and x.get("active")}


def price_pair(hist, book, cutoff):
    """(open1, open2, close1, close2) using quotes where BOTH sides were priced at the same moment,
    strictly before `cutoff` (ISO string). Returns Nones if not available."""
    a, b = _hist(hist, book, "181"), _hist(hist, book, "182")
    both = sorted(t for t in a if t in b and t < cutoff)
    both = [t for t in both if 0.02 <= 1 / a[t] + 1 / b[t] - 1 <= 0.12]     # sane margin = genuine 2-way quote
    if not both:
        return None, None, None, None, None
    return a[both[0]], b[both[0]], a[both[-1]], b[both[-1]], both[0]


def market_table(first_game_times=None):
    fixtures = {f["fixtureId"]: f for f in json.loads((RAW / "fixtures_2026.json").read_text())}
    rows = []
    for p in (RAW / "hist").glob("*.json"):
        d = json.loads(p.read_text()); f = fixtures.get(d.get("fixtureId"))
        if not f:
            continue
        row = dict(fixtureId=f["fixtureId"], mk_start=pd.Timestamp(f["startTime"]).tz_localize(None),
                   p1=f["participant1Name"], p2=f["participant2Name"], tournament=f["tournamentName"],
                   true_start=f.get("trueStartTime"), histjson=d)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- matching
def norm(s):
    s = str(s).lower()
    s = re.sub(r"\b(esports?|e-sports|gaming|team|club|fc)\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.95
    return difflib.SequenceMatcher(None, a, b).ratio()


def match(model, market):
    out = []
    for m in market.itertuples(index=False):
        cand = model[(model.start - m.mk_start).abs() <= pd.Timedelta(hours=30)]
        best, score, flip = None, 0, False
        for c in cand.itertuples(index=False):
            s_same = min(sim(m.p1, c.t1_name), sim(m.p2, c.t2_name))
            s_flip = min(sim(m.p1, c.t2_name), sim(m.p2, c.t1_name))
            sc = max(s_same, s_flip) - abs((c.start - m.mk_start).total_seconds()) / 86400 / 100
            if sc > score:
                best, score, flip = c, sc, s_flip > s_same
        if best is None or score < 0.6:
            continue
        r = m._asdict()
        # cutoff = earliest of: listed start, recorded true start, first game in Oracle's Elixir; minus 10 min
        cands = [m.mk_start, best.start]
        if isinstance(m.true_start, str):
            cands.append(pd.Timestamp(m.true_start).tz_localize(None))
        cutoff = (min(cands) - pd.Timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")
        for book in ("pinnacle", "bet365", "williamhill"):
            o1, o2, c1, c2, t_open = price_pair(m.histjson, book, cutoff)
            r.update({f"{book}_open1": o1, f"{book}_open2": o2, f"{book}_close1": c1, f"{book}_close2": c2,
                      f"{book}_open_time": t_open})
        r["cutoff"] = cutoff
        del r["histjson"]
        p1_model = 1 - best.p_model if flip else best.p_model
        t1, t2 = (best.t2, best.t1) if flip else (best.t1, best.t2)
        r.update(t1_id=t1, t2_id=t2, t1_lg=best.t2_lg if flip else best.t1_lg, t2_lg=best.t1_lg if flip else best.t2_lg)
        r.update(bo=best.bo, p1_won=(1 - best.t1_won) if flip else best.t1_won, p1_model=p1_model,
                 match_score=round(score, 2), oe_names=f"{best.t1_name} v {best.t2_name}")
        out.append(r)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- tests
def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    files = data_files()
    print(f"Running model over {len(files)} data files...")
    model = model_series_table(files)
    market = market_table()
    df = match(model, market)
    print(f"{len(market)} series with odds history; {len(df)} matched to Oracle's Elixir results.")
    if df.empty:
        return
    pin = df.dropna(subset=["pinnacle_close1", "pinnacle_close2"]).copy()
    pin["p1_close"] = [devig(a, b) for a, b in zip(pin.pinnacle_close1, pin.pinnacle_close2)]
    has_open = pin.pinnacle_open1.notna() & pin.pinnacle_open2.notna()
    pin.loc[has_open, "p1_open"] = [devig(a, b) for a, b in zip(pin[has_open].pinnacle_open1, pin[has_open].pinnacle_open2)]
    y = pin.p1_won.values
    n = len(pin)
    print(f"\n=== 1. Forecast accuracy on {n} series (log loss, lower is better) ===")
    l_mod, l_pin = ll(pin.p1_model.values, y), ll(pin.p1_close.values, y)
    d = l_mod - l_pin
    print(f"  model            {l_mod.mean():.4f}")
    print(f"  Pinnacle close   {l_pin.mean():.4f}")
    print(f"  model - Pinnacle {d.mean():+.4f} ± {d.std() / math.sqrt(n):.4f}  (negative = model better)")

    print("\n=== 2. Blend: logit(p) = w*logit(model) + (1-w)*logit(Pinnacle) ===")
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    for w in (0.0, 0.1, 0.2, 0.3, 0.5):
        pb = 1 / (1 + np.exp(-(w * lg(pin.p1_model.values) + (1 - w) * lg(pin.p1_close.values))))
        print(f"  w={w:.1f}  log loss {ll(pb, y).mean():.4f}")

    # model as it stood when the OPENING price was posted (no peeking at later results)
    q = [(pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t), a, b, la, lb)
         for t, a, b, la, lb in zip(pin.pinnacle_open_time, pin.t1_id, pin.t2_id, pin.t1_lg, pin.t2_lg)]
    pg = model_at_times(files, q)
    pin["p1_model_at_open"] = [series_prob(p, bo, series_spread(la, lb)) if bo > 1 else p
                               for p, bo, la, lb in zip(pg, pin.bo, pin.t1_lg, pin.t2_lg)]
    pin["hours_open_to_start"] = [(pd.Timestamp(c) - pd.Timestamp(o).tz_localize(None)).total_seconds() / 3600
                                  for c, o in zip(pin.cutoff, pin.pinnacle_open_time)]
    print(f"\n(opening prices were posted a median {pin.hours_open_to_start.median():.0f} hours before the start)")
    o = pin[pin.p1_open.notna()]
    if len(o) >= 5:
        gap0 = o.p1_model_at_open - o.p1_open
        move = o.p1_close - o.p1_open
        big0 = gap0.abs() > 0.05
        same0 = (np.sign(gap0) == np.sign(move)) & (move != 0)
        print(f"\n=== 3a. FAIR line-movement test (model frozen at the opening time), {len(o)} series ===")
        print(f"  correlation(model gap, line move) = {np.corrcoef(gap0, move)[0, 1]:+.3f}")
        print(f"  model differs from open by >5 pts ({big0.sum()} series): line moved towards model "
              f"{same0[big0].mean():.0%} of the time (50% = no signal)")
        lo = ll(o.p1_model_at_open.values, o.p1_won.values).mean()
        print(f"  log loss: model-at-open {lo:.4f} | Pinnacle open {ll(o.p1_open.values, o.p1_won.values).mean():.4f}"
              f" | Pinnacle close {ll(o.p1_close.values, o.p1_won.values).mean():.4f}")
    if len(o) >= 5:
        print(f"\n=== 3b. Same test, model as of match start (UNFAIR: has seen results the open hadn't) ===")
        gap = o.p1_model - o.p1_open          # model's disagreement with the opening line
        move = o.p1_close - o.p1_open          # how the line actually moved
        big = gap.abs() > 0.05
        same_dir = (np.sign(gap) == np.sign(move)) & (move != 0)
        print(f"  correlation(model gap, line move) = {np.corrcoef(gap, move)[0, 1]:+.3f}")
        print(f"  when model differs from open by >5 pts ({big.sum()} series): "
              f"line moved towards model {same_dir[big].mean():.0%} of the time "
              f"(50% = no signal)")

    print(f"\n=== 4. Paper bets (EV > {MIN_EDGE:.0%}, 1 unit each) ===")
    for book in ("pinnacle", "bet365", "williamhill"):
        b = pin.dropna(subset=[f"{book}_close1", f"{book}_close2"])
        bets = []
        for r in b.itertuples(index=False):
            for side, p, odds, won in ((1, r.p1_model, getattr(r, f"{book}_close1"), r.p1_won),
                                       (2, 1 - r.p1_model, getattr(r, f"{book}_close2"), 1 - r.p1_won)):
                if p * odds - 1 > MIN_EDGE:
                    bets.append(dict(odds=odds, ev=p * odds - 1, won=won, pnl=(odds - 1) if won else -1))
        bets = pd.DataFrame(bets)
        if len(bets):
            se = bets.pnl.std() / math.sqrt(len(bets))
            print(f"  {book:<12} {len(bets):4d} bets, avg odds {bets.odds.mean():.2f}, "
                  f"avg claimed EV {bets.ev.mean():+.1%}, actual ROI {bets.pnl.mean():+.1%} ± {se:.1%}")
        else:
            print(f"  {book:<12} no bets")
    pin.to_csv("market_comparison.csv", index=False)
    print("\nRow-by-row detail saved to market_comparison.csv")
    if n < 200:
        print(f"NOTE: {n} series is a small sample; treat every number above as provisional.")


if __name__ == "__main__":
    main()
