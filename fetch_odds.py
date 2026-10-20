"""
Collect League of Legends odds from OddsPapi (https://oddspapi.io), free tier.

Setup (once):
  1. Sign up at oddspapi.io and copy your API key.
  2. pip install requests pandas
  3. Put the key in an environment variable (never in a file in this repo):
       PowerShell:  $env:ODDSPAPI_KEY = "your_key"
       bash/zsh:    export ODDSPAPI_KEY="your_key"

Usage:
  python fetch_odds.py discover            # find the LoL sport id + list bookmakers & markets
  python fetch_odds.py tournaments         # find LoL tournament IDs (needed for snapshot)
  python fetch_odds.py probe               # 5 requests: check which recent top matches have odds history
  python fetch_odds.py backfill            # historical odds for every LoL fixture since Jan 2026
  python fetch_odds.py snapshot            # odds right now for all upcoming top-league fixtures (~1 request)
  python fetch_odds.py flatten             # turn the raw JSON into odds_history.csv
  python fetch_odds.py usage               # requests used this month (free plan = 250)
  python fetch_odds.py setusage 120        # correct the counter to match your OddsPapi dashboard

The script stops by itself after 3 failed requests in a row.

Free plan: 250 requests/month. discover uses 3; listing 2026 fixtures uses ~30 (once, then cached);
each match's odds history uses 1. Backfill does top leagues + internationals first, oldest first,
and stops itself at the monthly budget. Add --all to include minor leagues.

Everything raw is cached under ./odds_raw/, so backfill can be stopped and resumed.
Written against OddsPapi's v4 API.
If a call fails, run `discover` and send me the output.
"""
import json, os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

VERSION = "2026-10-01d"

BASE = "https://api.oddspapi.io/v4"
KEY = os.environ.get("ODDSPAPI_KEY")
RAW = Path("odds_raw"); RAW.mkdir(exist_ok=True)
CFG = Path("odds_config.json")
COOLDOWN = 5.2  # seconds; the API enforces 5000 ms per endpoint
# Max 3 per request. Pinnacle = sharp benchmark; bet365 + William Hill = UK books you can actually use.
BOOKMAKERS = os.environ.get("ODDSPAPI_BOOKS", "pinnacle,bet365,williamhill")
_last = {}
_errors_in_a_row = 0
MAX_ERRORS_IN_A_ROW = 3   # stop instead of burning the budget on a request that keeps failing
# Free plan: 250 requests per month. Keep a few spare for checks.
MONTHLY_BUDGET = int(os.environ.get("ODDSPAPI_BUDGET", "240"))
USAGE = RAW / "usage.json"
# Tournaments worth spending requests on (exact OddsPapi tournament names)
PRIORITY_NAMES = {"LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL", "LCK Cup", "Mid-Season Invitational",
                  "First Stand", "Esports World Cup", "World Championship", "Worlds", "LoL World Championship"}
# looser keywords, only used to find tournaments (e.g. this year's Worlds) in the /tournaments list
PRIORITY = ["world", "mid-season", "first stand", "esports world cup"]


def is_priority(f):
    name = str(f.get("tournamentName", ""))
    return name in PRIORITY_NAMES or "world championship" in name.lower()


def _usage():
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    u = json.loads(USAGE.read_text()) if USAGE.exists() else {}
    return month, u


def used_this_month():
    month, u = _usage()
    return u.get(month, 0)


def _count():
    month, u = _usage()
    u[month] = u.get(month, 0) + 1
    USAGE.write_text(json.dumps(u))


def get(path, **params):
    if not KEY:
        sys.exit('Set your key first:  $env:ODDSPAPI_KEY = "your_key"')
    if used_this_month() >= MONTHLY_BUDGET:
        sys.exit(f"Stopped: {used_this_month()} requests used this month (budget {MONTHLY_BUDGET}). "
                 "Everything fetched so far is saved; run again next month or raise ODDSPAPI_BUDGET.")
    wait = COOLDOWN - (time.time() - _last.get(path, 0))
    if wait > 0:
        time.sleep(wait)
    params["apiKey"] = KEY
    for attempt in range(4):
        r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
        _last[path] = time.time()
        _count()
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1)); continue
        global _errors_in_a_row
        if r.status_code >= 400:
            _errors_in_a_row += 1 if r.status_code != 404 else 0.34   # ~9 "not found" in a row, or 3 real errors
            print(f"  ! {path} {r.status_code}: {r.text[:300]}")
            if _errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                sys.exit(f"Stopped after {MAX_ERRORS_IN_A_ROW} failed requests in a row, to protect your "
                         "monthly allowance. Send Claude the error above.")
            return None
        _errors_in_a_row = 0
        return r.json()
    return None


def cfg():
    if not CFG.exists():
        sys.exit("Run `python fetch_odds.py discover` first.")
    return json.loads(CFG.read_text())


def discover():
    sports = get("sports") or []
    (RAW / "sports.json").write_text(json.dumps(sports, indent=1))
    lol = [s for s in sports if "legends" in json.dumps(s).lower()]
    print("League of Legends sport entries:", json.dumps(lol, indent=1))
    if not lol:
        sys.exit("Couldn't find LoL in /sports - see odds_raw/sports.json")
    sid = lol[0].get("sportId") or lol[0].get("id")
    CFG.write_text(json.dumps({"sportId": sid}))
    for ep in ["bookmakers", "markets"]:
        data = get(ep, sportId=sid) if ep == "markets" else get(ep)
        if data is not None:
            (RAW / f"{ep}.json").write_text(json.dumps(data, indent=1))
            print(f"saved odds_raw/{ep}.json ({len(data)} entries)")
    print("Saved sportId", sid, "to", CFG)


def tournaments():
    """List LoL tournaments (1 request) and save the priority ones' IDs for snapshots."""
    sid = cfg()["sportId"]
    data = get("tournaments", sportId=sid) or []
    (RAW / "tournaments.json").write_text(json.dumps(data, indent=1))
    pri = [t for t in data if str(t.get("tournamentName", "")) in PRIORITY_NAMES
           or any(k in str(t.get("tournamentName", "")).lower() for k in PRIORITY)]
    for t in pri:
        print(" ", {k: v for k, v in t.items() if k in ("tournamentId", "tournamentName", "categoryName",
                                                        "upcomingFixtures", "liveFixtures", "futureFixtures")})
    c = cfg(); c["tournamentIds"] = [t.get("tournamentId") for t in pri if t.get("tournamentId") is not None]
    CFG.write_text(json.dumps(c))
    print(f"{len(data)} LoL tournaments; saved {len(c['tournamentIds'])} priority IDs for snapshots")


def fixtures_between(start, end):
    sid = cfg()["sportId"]
    out, d = [], start
    while d < end:
        e = min(d + timedelta(days=9), end)
        data = get("fixtures", sportId=sid, **{"from": d.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                "to": e.strftime("%Y-%m-%dT%H:%M:%SZ")}) or []
        out += data
        print(f"  fixtures {d:%Y-%m-%d}..{e:%Y-%m-%d}: {len(data)}")
        d = e
    return out


def backfill():
    fx_file = RAW / "fixtures_2026.json"
    if fx_file.exists():
        fixtures = json.loads(fx_file.read_text())
    else:
        fixtures = fixtures_between(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))
        fx_file.write_text(json.dumps(fixtures))
    (RAW / "hist").mkdir(exist_ok=True)
    todo = [f for f in fixtures if not (RAW / "hist" / f"{f['fixtureId']}.json").exists()]
    top = [f for f in todo if is_priority(f)]
    if "--all" not in sys.argv:
        todo = top
    todo.sort(key=lambda f: str(f.get("startTime", "")), reverse="--oldest" not in sys.argv)  # newest first
    left = MONTHLY_BUDGET - used_this_month()
    print(f"{len(fixtures)} fixtures found; {len(top)} in priority leagues still to fetch. "
          f"{left} requests left this month, so this run fetches up to {min(left, len(todo))}.")
    for i, f in enumerate(todo, 1):
        data = get("historical-odds", fixtureId=f["fixtureId"], bookmakers=BOOKMAKERS)
        if data is not None:
            (RAW / "hist" / f"{f['fixtureId']}.json").write_text(json.dumps(data))
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")


def probe(n=5):
    """Spend n requests on recent top-league matches to see what history the API really holds."""
    fixtures = json.loads((RAW / "fixtures_2026.json").read_text())
    fx = sorted([f for f in fixtures if is_priority(f) and f.get("statusName") == "Finished"],
                key=lambda f: f["startTime"], reverse=True)
    picks, seen = [], set()
    for f in fx:                       # one per tournament, most recent first
        if f["tournamentName"] not in seen:
            picks.append(f); seen.add(f["tournamentName"])
        if len(picks) == n:
            break
    (RAW / "hist").mkdir(exist_ok=True)
    for f in picks:
        data = get("historical-odds", fixtureId=f["fixtureId"], bookmakers=BOOKMAKERS)
        label = f'{f["startTime"][:10]} {f["tournamentName"]:<12} {f["participant1Name"]} v {f["participant2Name"]}'
        if not data:
            print(f"  none  {label}"); continue
        (RAW / "hist" / f"{f['fixtureId']}.json").write_text(json.dumps(data))
        books = data.get("bookmakers") or {}
        summary = []
        for bk, b in books.items():
            mk = b.get("markets") or {}
            pts = sum(len(h) for m in mk.values() for o in (m.get("outcomes") or {}).values()
                      for h in (o.get("players") or {}).values())
            summary.append(f"{bk}: {len(mk)} markets, {pts} prices")
        print(f"  FOUND {label} -> " + ("; ".join(summary) or "empty"))
    print(f"({used_this_month()} of {MONTHLY_BUDGET} requests used this month)")


def snapshot():
    """Current odds for every fixture in the priority tournaments: ~1 request per run."""
    ids = cfg().get("tournamentIds")
    if not ids:
        sys.exit("Run `python fetch_odds.py tournaments` first.")
    now = datetime.now(timezone.utc)
    snap = RAW / "snapshots"; snap.mkdir(exist_ok=True)
    data = get("odds-by-tournaments", tournamentIds=",".join(map(str, ids)), bookmakers=BOOKMAKERS)
    if data is None:
        return
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    (snap / f"{stamp}.json").write_text(json.dumps(data))
    n = len(data) if isinstance(data, list) else len(data.get("data", data))
    print(f"saved odds for {n} fixtures to odds_raw/snapshots/{stamp}.json "
          f"({used_this_month()} of {MONTHLY_BUDGET} requests used this month)")


def flatten():
    import pandas as pd
    fixtures = {f["fixtureId"]: f for f in json.loads((RAW / "fixtures_2026.json").read_text())}
    rows = []
    for p in (RAW / "hist").glob("*.json"):
        d = json.loads(p.read_text()); f = fixtures.get(d.get("fixtureId"), {})
        for bk, b in (d.get("bookmakers") or {}).items():
            for mid, m in (b.get("markets") or {}).items():
                for oid, o in (m.get("outcomes") or {}).items():
                    for pid, hist in (o.get("players") or {}).items():
                        for h in hist:
                            rows.append(dict(fixtureId=d.get("fixtureId"), startTime=f.get("startTime"),
                                             team1=f.get("participant1Name"), team2=f.get("participant2Name"),
                                             tournament=f.get("tournamentName"), bookmaker=bk, marketId=mid,
                                             outcomeId=oid, playerId=pid, createdAt=h.get("createdAt"),
                                             price=h.get("price"), limit=h.get("limit"), active=h.get("active")))
    df = pd.DataFrame(rows)
    df.to_csv("odds_history.csv", index=False)
    print(f"odds_history.csv: {len(df):,} price points, {df.fixtureId.nunique() if len(df) else 0} fixtures")


if __name__ == "__main__":
    print(f"fetch_odds.py version {VERSION}")
    if len(sys.argv) > 1 and sys.argv[1] == "usage":
        print(f"{used_this_month()} of {MONTHLY_BUDGET} requests used this month"); sys.exit()
    if len(sys.argv) > 2 and sys.argv[1] == "setusage":
        # e.g. after checking the real figure on your OddsPapi dashboard
        month, u = _usage(); u[month] = int(sys.argv[2]); USAGE.write_text(json.dumps(u))
        print(f"usage this month set to {u[month]}"); sys.exit()
    {"discover": discover, "tournaments": tournaments, "probe": probe, "backfill": backfill, "snapshot": snapshot,
     "flatten": flatten}[
        sys.argv[1] if len(sys.argv) > 1 else "discover"]()
