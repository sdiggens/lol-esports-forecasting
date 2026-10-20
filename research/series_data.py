"""Every series 2023-26 with the model's pre-series per-game probability (side-neutral) and exact scoreline."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, RatingEngine, Params, _margin, INTERNATIONAL, CUPS, TIER1
G = load_games(data_files())
eng = RatingEngine(Params()); pre = {}; gp = []
for r in G.itertuples(index=False):
    pb = eng.predict(r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, as_of=r.date)[0]
    gp.append(pb)
    if r.game_no == 1:
        pr = 1 - eng.predict(r.r_teamid, r.b_teamid, r.r_home_league, r.b_home_league, as_of=r.date)[0]
        pre[r.gameid] = (pb + pr) / 2
    lub = r.b_lineup if isinstance(r.b_lineup, tuple) else None; lur = r.r_lineup if isinstance(r.r_lineup, tuple) else None
    eng.update(r.date, r.b_teamid, r.r_teamid, r.b_home_league, r.r_home_league, r.y, lub, lur, _margin(r, 'gd15'),
               intl=r.league in INTERNATIONAL, margin2=_margin(r, 'gspd'), weight=eng.p.cup_w if r.league in CUPS else 1.0)
G['p_pre'] = gp
G['pair'] = [tuple(sorted(x)) for x in zip(G.b_teamid, G.r_teamid)]
ser, cur = [], {}
for r in G.itertuples(index=False):
    if r.game_no == 1 or r.pair not in cur:
        s = dict(start=r.date, league=r.league, t1=r.b_teamid, t1_name=r.b_teamname, t2=r.r_teamid, t2_name=r.r_teamname,
                 p=pre.get(r.gameid, np.nan), w1=0, w2=0, games=0, gameids=[])
        ser.append(s); cur[r.pair] = s
    s = cur[r.pair]; s['games'] += 1; s['gameids'].append(r.gameid)
    win = r.b_teamid if r.y == 1 else r.r_teamid
    if win == s['t1']: s['w1'] += 1
    else: s['w2'] += 1
d = pd.DataFrame(ser)
d['need'] = d[['w1', 'w2']].max(axis=1)
d['bo'] = d.need.map({1: 1, 2: 3, 3: 5})
d = d.dropna(subset=['bo', 'p'])
# consistency: games must equal w1+w2 and loser < need; drop malformed (e.g. Bo2 formats, missing games)
d = d[(d.games == d.w1 + d.w2) & (d[['w1', 'w2']].min(axis=1) < d.need)]
d['bo'] = d.bo.astype(int); d['t1_won'] = (d.w1 > d.w2).astype(int)
d['tier1'] = d.league.isin(TIER1 | INTERNATIONAL)
d.drop(columns=['gameids']).to_pickle('series_table.pkl')
G[['gameid', 'date', 'league', 'p_pre', 'y']].to_pickle('game_preds_before.pkl')
print(d.groupby([d.start.dt.year, 'bo']).size().unstack())
