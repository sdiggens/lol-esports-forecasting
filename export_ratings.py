"""Export the model's current beliefs to ratings.json for the Rift Pricer page.
Run after downloading fresh Oracle's Elixir data:  python export_ratings.py"""
import glob, json, sys
from ratings import data_file, data_files
from dataclasses import asdict
import pandas as pd
from ratings import load_games, run, TIER1, INTERNATIONAL, SERIES_SPREAD_TOP, SERIES_SPREAD_LOW
from series_math import _X, _W

files = data_files()
g = load_games(files)
eng, _ = run(g)
last_year = g.date.dt.year.max()

# latest name / home league per team id
side = pd.concat([
    g[["date", "b_teamid", "b_teamname", "b_home_league", "league"]].set_axis(["date", "id", "name", "home", "league"], axis=1),
    g[["date", "r_teamid", "r_teamname", "r_home_league", "league"]].set_axis(["date", "id", "name", "home", "league"], axis=1)])
side = side[side.date.dt.year == last_year].sort_values("date")
latest = side.groupby("id").last()
games = side.groupby("id").size()

teams = []
for tid, r in latest.iterrows():
    k = ("T", tid)
    if k not in eng.s.mean:
        continue
    teams.append(dict(id=str(tid), name=r["name"], league=r.home, tier1=r.home in TIER1,
                      mean=round(eng.s.mean[k], 5), var=round(eng.s.var[k], 5),
                      last=eng.s.last_seen[k].strftime("%Y-%m-%d"), games=int(games.get(tid, 0))))
leagues = {k[1]: dict(mean=round(v, 5), var=round(eng.s.var[k], 5), last=eng.s.last_seen[k].strftime("%Y-%m-%d"))
           for k, v in eng.s.mean.items() if k[0] == "L"}
p = asdict(eng.p)
out = dict(as_of=g.date.max().strftime("%Y-%m-%d %H:%M"), games=len(g), files=[f.split("/")[-1] for f in files],
           params={k: p[k] for k in ("side_adv", "tau_team", "tau_league", "season_keep", "season_var",
                                     "var_cap", "tier2_mean", "tier2_var", "league_var0", "tau_tier2")},
           series=dict(spread_top=SERIES_SPREAD_TOP, spread_low=SERIES_SPREAD_LOW,
                       gh_x=[round(float(x), 12) for x in _X], gh_w=[float(w) for w in _W]),
           tier1=sorted(TIER1), teams=sorted(teams, key=lambda t: -(t["mean"] + leagues.get(t["league"], {"mean": 0})["mean"])),
           leagues=leagues)
json.dump(out, open("ratings.json", "w"), indent=0)
print(f"ratings.json: {len(teams)} teams, data to {out['as_of']}")
