"""Residual tests: does a feature improve on the rating model's own prediction?
Fit  logit(P) = a*logit(p_model) + b*feature  on 2023-25, score log loss on 2026."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from ratings import data_file, data_files
from ratings import load_games, run, Params, INTERNATIONAL, TIER1
files=data_files()
G=load_games(files); _,p=run(G); G['p']=p; G['z']=np.log(p/(1-p))
raw=pd.concat([pd.read_csv(f,low_memory=False,usecols=['gameid','date','position','teamid','side','gamelength','golddiffat20','patch','result']) for f in files])
raw=raw[raw.position=='team']; raw['date']=pd.to_datetime(raw.date)
# --- rest days & games in last 7 days, per team
raw=raw.sort_values('date')
raw['rest']=raw.groupby('teamid').date.diff().dt.total_seconds()/86400
raw['rest']=raw.rest.clip(upper=30)
raw['busy7']=raw.groupby('teamid').date.transform(lambda s: s.apply(lambda d: ((s>=d-pd.Timedelta('7D'))&(s<d)).sum()) if len(s)<5000 else 0)
pr=raw.pivot_table(index='gameid',columns='side',values=['rest','busy7'],aggfunc='first')
G['d_rest']=G.gameid.map((np.log1p(pr['rest']['Blue'].fillna(7))-np.log1p(pr['rest']['Red'].fillna(7))))
G['d_busy']=G.gameid.map(pr['busy7']['Blue']-pr['busy7']['Red'])
G['same_day']=G.gameid.map(((pr['rest']['Blue']<0.5)|(pr['rest']['Red']<0.5)).astype(float))  # 2nd+ game of a series
# --- within-series momentum: did blue team win the previous game of this series?
G['pair']=[tuple(sorted(x)) for x in zip(G.b_teamid,G.r_teamid)]
prev={}; mom=[]
for r in G.itertuples():
    if r.game_no==1: prev.pop(r.pair,None)
    w=prev.get(r.pair)
    mom.append(0 if w is None else (1 if w==r.b_teamid else -1))
    prev[r.pair]= r.b_teamid if r.y==1 else r.r_teamid
G['momentum']=mom
# --- side advantage: rolling blue win rate in this league over previous 300 games (pre-game)
G['blue_rate']=G.groupby('league').y.transform(lambda s: s.shift(1).rolling(300,min_periods=50).mean()).fillna(0.534)
G['side_dev']=np.log(G.blue_rate/(1-G.blue_rate))-0.135
# --- patch freshness: games since current patch started (league-wide), small = new patch
G['patch_new']=(G.groupby(['patch']).cumcount()<150).astype(float)
G['z_patchnew']=G.z*G.patch_new
# --- score tests
tr=(G.date.dt.year<2026).values & (G.date.dt.year>2023).values   # skip 2023 warm-up year
te=(G.date.dt.year==2026).values
maj=G.league.isin(TIER1|INTERNATIONAL).values
y=G.y.values
def fit(X):
    X=np.column_stack([np.ones(len(X)),X]); b=np.zeros(X.shape[1]); b[1]=1
    for _ in range(30):
        q=1/(1+np.exp(-X[tr]@b)); g=X[tr].T@(y[tr]-q); H=(X[tr]*(q*(1-q))[:,None]).T@X[tr]+1e-6*np.eye(len(b)); b+=np.linalg.solve(H,g)
    return b, 1/(1+np.exp(-X@b))
ll=lambda q,m: -(y[m]*np.log(q[m])+(1-y[m])*np.log(1-q[m]))
b0,q0=fit(G[['z']].values)
print(f'baseline (recalibrated) 2026 all {ll(q0,te).mean():.4f}  tier1+intl {ll(q0,te&maj).mean():.4f}  coefs {np.round(b0,3)}')
tests={'Rest days difference (log)':['d_rest'],'Games in last 7 days difference':['d_busy'],
       'Within-series momentum (won last game)':['momentum'],'League-specific side advantage':['side_dev'],
       'New patch: shrink confidence':['z_patchnew']}
for name,cols in tests.items():
    X=G[['z']+cols].fillna(0).values; b,q=fit(X)
    for m,lab in [(te,'all'),(te&maj,'tier1+intl')]:
        d=ll(q0,m)-ll(q,m)
        print(f'{name:42} {lab:10} gain {d.mean():+.4f} ± {d.std()/np.sqrt(m.sum()):.4f}  coef {np.round(b[2:],3)}')
# --- per-league calibration scale (fit 2024-25, test 2026)
lg=G.league.where(G.league.isin(['LPL','LCK','LEC','LCS','LCP','CBLOL','LCKC','EM','LFL','PRM']),'other')
Xl=np.column_stack([G.z*(lg==L) for L in sorted(lg.unique())]); b,q=fit(Xl)
for m,lab in [(te,'all'),(te&maj,'tier1+intl')]:
    d=ll(q0,m)-ll(q,m); print(f'{"Per-league confidence scaling":42} {lab:10} gain {d.mean():+.4f} ± {d.std()/np.sqrt(m.sum()):.4f}')
print('   per-league scales:', dict(zip(sorted(lg.unique()), np.round(b[1:],2))))
# --- series dependence check: actual vs predicted for deciding games
dec=G[(G.momentum!=0)]
print('\nGames after game 1: blue won previous ->', round(dec[dec.momentum==1].y.mean(),3),'vs model',round(dec[dec.momentum==1].p.mean(),3),
      '| red won previous ->', round(dec[dec.momentum==-1].y.mean(),3),'vs model',round(dec[dec.momentum==-1].p.mean(),3))
G[['gameid','momentum','d_rest','side_dev']].to_csv('idea_features.csv',index=False)
