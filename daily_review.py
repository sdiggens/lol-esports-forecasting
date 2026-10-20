"""Compare the model's pre-game predictions with results for games that are new in the latest download."""
import sys, numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, RatingEngine, Params, _margin, INTERNATIONAL, CUPS, TIER1
old_ids = set(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'oe_2026_old.csv', low_memory=False, usecols=['gameid']).gameid)
G = load_games(data_files())
raw = pd.read_csv(data_file(2026), low_memory=False, usecols=['gameid', 'position', 'side', 'gamelength', 'firstblood', 'firstdragon', 'dragons', 'barons', 'kills', 'deaths'])
tm = raw[raw.position == 'team']
blue = tm[tm.side == 'Blue'].set_index('gameid')
G['gamelength'] = G.gameid.map(blue.gamelength) / 60
eng = RatingEngine(Params()); pre = []
for r in G.itertuples(index=False):
    pb = eng.predict(r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, as_of=r.date)[0]
    pre.append(pb)
    lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None; lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
    eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur, _margin(r, 'gd15'),
               intl=r.league in INTERNATIONAL, margin2=_margin(r, 'gspd'), weight=eng.p.cup_w if r.league in CUPS else 1.0)
G['p'] = pre
new_ids = set(tm.gameid) - old_ids
new = G[G.gameid.isin(new_ids)].copy()
y = new.y.values; p = new.p.values
ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"NEW GAMES: {len(new)}  |  model accuracy {((p > .5) == y).mean():.0%}  |  log loss {ll.mean():.3f} (coin flip 0.693)")
print(f"model's average confidence in the favourite {np.maximum(p, 1-p).mean():.0%}; favourites actually won {((p > .5) == y).mean():.0%}")
print("\nBy event:")
for lg, g in new.groupby('league'):
    yy, pp = g.y.values, g.p.values
    print(f"  {lg:6} {len(g):3d} games  acc {((pp>.5)==yy).mean():.0%}  expected fav wins {np.maximum(pp,1-pp).mean():.0%}  log loss {(-(yy*np.log(pp)+(1-yy)*np.log(1-pp))).mean():.3f}"
          f"  blue wins {yy.mean():.0%}  avg length {g.gamelength.mean():.1f} min")
# baseline norms for comparison: same leagues, Jan-Sep 2026
base = G[(G.date.dt.year == 2026) & G.gameid.isin(old_ids)]
print("\nGame length vs each league's 2026 average (minutes):")
for lg, g in new.groupby('league'):
    b = base[base.league == lg].gamelength
    if len(b) > 30 and len(g) >= 5:
        z = (g.gamelength.mean() - b.mean()) / (b.std() / np.sqrt(len(g)))
        print(f"  {lg:6} new {g.gamelength.mean():.1f} vs season {b.mean():.1f}  (z = {z:+.1f})")
print(f"\nBlue side: new games {new.y.mean():.0%} blue wins (season {base.y.mean():.0%})")
# biggest upsets
new['upset'] = np.where(new.y == 1, 1 - new.p, new.p)
print("\nBiggest upsets (model gave the winner the lowest chance):")
for r in new.sort_values('upset', ascending=False).head(8).itertuples():
    w, l = (r.b_teamname, r.r_teamname) if r.y == 1 else (r.r_teamname, r.b_teamname)
    print(f"  {r.date:%m-%d} {r.league:6} {w} beat {l}  (model gave {w} {1 - r.upset:.0%})")
# cross-region results within EM / WSCI: did leagues over/under-perform?
print("\nCross-league games: actual minus expected wins, by home league")
cross = new[new.b_home_league != new.r_home_league]
rows = []
for r in cross.itertuples():
    rows.append((r.b_home_league, r.y - r.p)); rows.append((r.r_home_league, (1 - r.y) - (1 - r.p)))
cr = pd.DataFrame(rows, columns=['league', 'surplus']).groupby('league').surplus.agg(['count', 'sum'])
print(cr[cr['count'] >= 4].sort_values('sum').round(2).to_string())
new[['date', 'league', 'b_teamname', 'r_teamname', 'p', 'y', 'gamelength']].to_csv('daily_review_latest.csv', index=False)
