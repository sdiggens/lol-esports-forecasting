"""Champions, player comfort picks and melee mids, measured against the model.
Each player-game gets the model's pre-game win chance for that team; 'over' = actual wins - expected wins.
That strips out team strength: a champion picked mostly by strong teams doesn't look good just for that."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ratings.py
import numpy as np, pandas as pd
from ratings import data_file, data_files
G = pd.read_pickle("G_pre.pkl")[["gameid", "p", "y"]]
cols = ["gameid", "date", "league", "patch", "side", "position", "playername", "teamname", "champion", "result",
        "ban1", "ban2", "ban3", "ban4", "ban5", "participantid"]
raw = pd.concat([pd.read_csv(data_file(y), usecols=cols, low_memory=False) for y in (2023, 2024, 2025, 2026)])
raw["date"] = pd.to_datetime(raw.date)
raw = raw.drop_duplicates(["gameid", "participantid"]).merge(G, on="gameid")
raw["pt"] = np.where(raw.side == "Blue", raw.p, 1 - raw.p)           # model's chance for this row's team
pl = raw[raw.position != "team"].copy(); tm = raw[raw.position == "team"].copy()

def summarise(df, key, min_n):
    s = df.groupby(key).agg(games=("result", "size"), wins=("result", "sum"), exp=("pt", "sum"),
                            var=("pt", lambda p: (p * (1 - p)).sum()))
    s = s[s.games >= min_n]
    s["win%"] = 100 * s.wins / s.games; s["model%"] = 100 * s.exp / s.games
    s["over"] = s.wins - s.exp; s["z"] = s.over / np.sqrt(s["var"])
    return s.drop(columns="var")

RECENT = pl[pl.date >= "2026-08-15"]          # patches 16.16-16.20
print(f"Recent window: {RECENT.gameid.nunique()} games since 15 Aug (patches {sorted(RECENT.patch.dropna().unique())})")
# presence: picks + bans per game
nrec = RECENT.gameid.nunique()
bans = tm[tm.date >= "2026-08-15"][["ban1", "ban2", "ban3", "ban4", "ban5"]].stack().value_counts()
picks = RECENT.champion.value_counts()
pres = ((picks.add(bans, fill_value=0)) / nrec * 100).sort_values(ascending=False)
c = summarise(RECENT, "champion", 40)
c["presence%"] = pres.reindex(c.index).round(0)
print("\nMost contested champions (presence = picked or banned), with record vs model:")
print(c.sort_values("presence%", ascending=False).head(15).round(1).to_string())
print("\nBiggest over/under-performers vs model (40+ games, sorted by z):")
cz = c.sort_values("z")
print(pd.concat([cz.tail(8)[::-1], cz.head(8)]).round(2).to_string())
# how many 'significant' champions would luck alone produce?
print(f"\nChampions tested: {len(c)}; |z| > 2: {(c.z.abs() > 2).sum()} (luck alone would give about {0.046*len(c):.0f})")

# persistence check: does a champion's over/under-performance on one patch window predict the next?
def window(a, b): return summarise(pl[(pl.date >= a) & (pl.date < b)], "champion", 30)
pairs = []
edges = pd.date_range("2025-01-01", "2026-10-14", freq="21D")
for i in range(len(edges) - 2):
    w1, w2 = window(edges[i], edges[i + 1]), window(edges[i + 1], edges[i + 2])
    j = w1.join(w2, lsuffix="1", rsuffix="2", how="inner")
    pairs.append(j[["over1", "games1", "over2", "games2"]])
P = pd.concat(pairs); r1 = P.over1 / P.games1; r2 = P.over2 / P.games2
print(f"Champion over-performance, one 3-week window to the next: correlation {np.corrcoef(r1, r2)[0,1]:+.3f} (n={len(P)})")

# melee vs ranged mids
MELEE = set("""Aatrox Akali Akshan_no Alistar Ambessa Amumu Belveth Bel'Veth Briar Camille ChoGath Cho'Gath Darius Diana DrMundo Dr._Mundo Ekko Fiora Fizz Galio Garen Gragas Gwen Hecarim Illaoi Irelia JarvanIV Jarvan_IV Jax Kassadin Katarina Kayn Kha'Zix Kled LeeSin Lee_Sin Leona Lillia_no Malphite Maokai MasterYi Master_Yi Mordekaiser Nasus Nautilus Naafiri Nocturne Nunu_&_Willump Olaf Pantheon Poppy Qiyana Rammus RekSai Rek'Sai Rell Renekton Rengar Riven Rumble Sejuani Sett Shaco Shen Shyvana Sion Skarner Sylas Tahm_Kench Talon Trundle Tryndamere Udyr Urgot_no Vi Viego Volibear Warwick MonkeyKing Wukong XinZhao Xin_Zhao Yasuo Yone Yorick Zac Zed Yunara_no""".replace("_", " ").split())
MELEE |= {"Bel'Veth", "Cho'Gath", "Dr. Mundo", "Jarvan IV", "Kha'Zix", "Lee Sin", "Master Yi", "Nunu & Willump", "Rek'Sai", "Tahm Kench", "Xin Zhao", "K'Sante"}
mid = pl[pl.position == "mid"].copy(); mid["type"] = np.where(mid.champion.isin(MELEE), "melee", "ranged")
mid["window"] = np.where(mid.date >= "2026-08-15", "since 15 Aug", np.where(mid.date.dt.year == 2026, "2026 before", "2023-25"))
print("\nMelee v ranged mids (record vs model):")
print(summarise(mid, ["window", "type"], 1)[["games", "win%", "model%", "over", "z"]].round(2).to_string())
print("Most-played melee mids since 15 Aug:", mid[(mid.type == "melee") & (mid.window == "since 15 Aug")].champion.value_counts().head(8).to_dict())

# player comfort: prior games on that champion (any league, any year) at the time of the game
pl = pl.sort_values("date")
pl["prior"] = pl.groupby(["playername", "champion"]).cumcount()
pl["comfort"] = pd.cut(pl.prior, [-1, 0, 4, 14, 1000], labels=["first time", "1-4 before", "5-14 before", "15+ before"])
print("\nPlayer comfort on the champion (2025-26 games, record vs model):")
print(summarise(pl[pl.date.dt.year >= 2025], "comfort", 1)[["games", "win%", "model%", "over", "z"]].round(2).to_string())

# standout player-champion pairs recently (for colour, with the luck caveat)
r = pl[pl.date >= "2026-06-01"]
pc = summarise(r, ["playername", "champion"], 8).sort_values("z", ascending=False)
pc = pc.join(r.groupby(["playername", "champion"]).teamname.last())
print("\nStandout player-champion records since 1 June (8+ games):")
print(pc.head(12).round(2).to_string())

# robustness: only established players (100+ prior pro games in the data), and tier-1 leagues only
pl["career"] = pl.groupby("playername").cumcount()
est = pl[(pl.date.dt.year >= 2025) & (pl.career >= 100)]
print("\nComfort, established players only (100+ prior games):")
print(summarise(est, "comfort", 1)[["games", "win%", "model%", "over", "z"]].round(2).to_string())
from ratings import TIER1
t1 = est[est.league.isin(TIER1 | {"MSI", "WLDs", "EWC", "FST", "DCGI"})]
print("\n...and top leagues + international only:")
print(summarise(t1, "comfort", 1)[["games", "win%", "model%", "over", "z"]].round(2).to_string())
# team level: number of first-time picks in a team's draft
pl["first"] = pl.prior == 0
team_first = pl[pl.date.dt.year >= 2025].groupby(["gameid", "side"]).agg(n_first=("first", "sum"), y=("result", "first"), pt=("pt", "first")).reset_index()
team_first["n_first"] = team_first.n_first.clip(upper=3)
print("\nTeam drafts by number of first-time picks (2025-26):")
print(summarise(team_first.rename(columns={"y": "result"}), "n_first", 1)[["games", "win%", "model%", "over", "z"]].round(2).to_string())
