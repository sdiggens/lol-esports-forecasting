import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import pandas as pd, numpy as np
from ratings import data_file, data_files
from ratings import load_games, run, Params, INTERNATIONAL, TIER1
files=data_files()
raw=pd.concat([pd.read_csv(f,low_memory=False,usecols=['gameid','date','league','position','teamid','teamname','playerid','playername','result','side','golddiffat15','game','datacompleteness']) .assign(src=f) for f in files])
raw['date']=pd.to_datetime(raw.date)
t=raw[raw.position=='team']
print('1. duplicate team rows (same gameid+side):', t.duplicated(['gameid','side']).sum(), '| gameids in >1 file:', t.groupby('gameid').src.nunique().gt(1).sum())
print('2. games without exactly 2 team rows:', (t.groupby('gameid').size()!=2).sum())
r2=t.groupby('gameid').result.sum(); print('3. games where results do not sum to 1:', (r2!=1).sum())
g=t.pivot_table(index='gameid',columns='side',values='golddiffat15',aggfunc='first').dropna()
print('4. gd15 blue == -red:', np.isclose(g.Blue,-g.Red).mean().round(4))
p=raw[raw.position!='team']
pc=p.groupby(['gameid','teamid']).playerid.apply(lambda s:s.notna().sum())
print('5. team-games with <5 known player ids:', (pc<5).sum(), 'of', len(pc), f'({(pc<5).mean():.1%})')
# 6. team id continuity across seasons for tier-1 teams: new teamid but mostly same players
p=p.assign(year=p.date.dt.year)
lu=p.dropna(subset=['playerid']).groupby(['teamid','year']).agg(players=('playerid',lambda s:set(s.value_counts().index[:5])),name=('teamname','last'),lg=('league',lambda s:s.value_counts().index[0]))
lu=lu.reset_index()
moves=[]
for y in (2024,2025,2026):
    a=lu[lu.year==y-1]; b=lu[(lu.year==y)&lu.lg.isin(TIER1)]
    old_ids=set(a.teamid)
    for r in b.itertuples():
        if r.teamid in old_ids: continue
        best=max(((len(r.players & q.players),q.name) for q in a.itertuples()),default=(0,None))
        if best[0]>=3: moves.append((y,r.name,best[1],best[0]))
print('6. tier-1 teams with a NEW team id but >=3 starters from a previous-year team:', len(moves)); [print('   ',m) for m in moves[:15]]
# 7. timezone sanity vs side info done in market test. 8. calibration by league on 2026 predictions
G=load_games(files); _,pr=run(G); G['p']=pr
y26=G[G.date.dt.year==2026]
print('8. calibration slope by league, 2026 (1.0 = perfect; <1 overconfident, >1 underconfident):')
for lg in ['LCK','LPL','LEC','LCS','LCP','CBLOL','MSI','EWC','FST','WLDs','LCKC','EM','LFL','PRM']:
    s=y26[y26.league==lg]
    if len(s)<60: continue
    x=np.log(s.p/(1-s.p)).values; yy=s.y.values; b=1.0
    for _ in range(50):  # Newton for slope only (no intercept)
        q=1/(1+np.exp(-b*x)); g_=np.sum((yy-q)*x); h=np.sum(q*(1-q)*x*x); b+=g_/h
    print(f'   {lg:6} n={len(s):4d} slope {b:.2f}')
