"""Which inputs predict PROP outcomes (first blood / drake / tower / herald, soul)?
Same set-up as feature_test.py, but the target is the prop, not the game result."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import math, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_test.py")).read().split("d20, d80")[0])   # reuse feature building
tt = raw[raw.position == "team"]
tb = tt[tt.side == "Blue"].set_index("gameid")
d20, d80 = g.date.quantile(0.2), g.date.quantile(0.8)
train = ((g.date >= d20) & (g.date < d80)).values; test = (g.date >= d80).values
rows = []
for prop, col, form in [("First blood", "firstblood", "d_first_blood"), ("First drake", "firstdragon", "d_first_drake"),
                        ("First tower", "firsttower", "d_first_tower"), ("First herald", "firstherald", "d_first_herald")]:
    yy = pd.to_numeric(g.gameid.map(tb[col]), errors="coerce").values
    ok = ~np.isnan(yy)
    def score(cols):
        X = g[cols].fillna(0).values; tr, te = train & ok, test & ok
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), yy[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]; y_ = yy[te]
        return -(y_ * np.log(p) + (1 - y_) * np.log(1 - p))
    base = np.full(int((test & ok).sum()), np.nan)
    rate = yy[train & ok].mean(); y_ = yy[test & ok]
    ll_const = -(y_ * np.log(rate) + (1 - y_) * np.log(1 - rate))
    ll_rating = score(["rating_logit"]); ll_form = score([form, "d_player_gd15", "d_gd10"])
    ll_both = score(["rating_logit", form, "d_player_gd15", "d_gd10"])
    for nm, ll in [("blue-side base rate only", ll_const), ("rating only", ll_rating),
                   ("prop form + early stats", ll_form), ("rating + prop form", ll_both)]:
        rows.append(dict(prop=prop, inputs=nm, logloss=ll.mean(), gain_vs_base_rate=(ll_const - ll).mean(),
                         se=(ll_const - ll).std() / math.sqrt(len(ll))))
print(pd.DataFrame(rows).round(4).to_string(index=False))
