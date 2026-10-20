"""
Bayesian dynamic rating engine for pro League of Legends (v1).

Model
-----
Each game is a paired comparison on the logit scale:

    logit P(blue wins) = (mu_L(b) + u_b) - (mu_L(r) + u_r) + h

  u_i     team i's strength relative to its league        ~ N(m_i, s_i^2)
  mu_L    league L's strength level                         ~ N(M_L, S_L^2)
  h       blue-side advantage (fixed, estimated from data)

Beliefs are Gaussian and updated after every game by assumed-density
filtering (the Bayesian analogue of Elo/Glicko for a logistic likelihood):

  x   = indicator vector (+1 for blue components, -1 for red)
  m   = x . theta                (predicted logit, mean)
  v   = sum_j x_j^2 Var_j        (predicted logit, variance)
  p   = sigmoid(m / sqrt(1 + pi v / 8))     <- uncertainty shrinks p to 0.5
  k   = p (1 - p)
  theta_j += Var_j x_j (y - p) / (1 + k v)
  Var_j   -= Var_j^2 x_j^2 k / (1 + k v)

Between games, strengths drift as a random walk:   Var_j += tau^2 * days.
A lineup change inflates the team's variance by `roster_var` per new player,
so the model learns quickly about rebuilt rosters and slowly about stable ones.

Because mu_L cancels in domestic games, league levels are learned only from
cross-league games (MSI, Worlds, EWC, FST...), and an upset at an international
event moves the whole league, not just the team that played.
Tuned 2026-10-01 by walk-forward validation on 2026 Oracle's Elixir data.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd


@dataclass
class Params:
    team_var0: float = 0.5       # prior variance for a team never seen before
    league_var0: float = 1.5     # prior variance for a league's level
    tau_team: float = 0.001      # team drift variance per day (tuned on 2025)
    tau_league: float = 0.0005   # league drift variance per day
    roster_var: float = 0.03     # extra variance per changed starter
    side_adv: float = 0.135      # blue-side logit advantage (~53.4% blue WR)
    var_cap: float = 1.0
    gd_c: float = 1.5            # gold-diff@15 (thousands) per unit of logit strength gap
    gd_sigma: float = 3.5        # tuned with gd_col2; noise s.d. of gold-diff@15 (thousands) given strengths
    gd_col2: str = "gspd"        # tuned 2025 (+0.0034 on 2026): end-of-game gold margin as 2nd observation; optional second margin observation (e.g. gspd), with its own scale/noise
    gd_c2: float = 1.5
    gd_sigma2: float = 3.5
    gd_col: str = "gd15"         # which margin to use: gd15 | gd20 | gspd (end-of-game gold spent %)
    # Tier-2 leagues (academies, ERLs) almost never meet tier-1 teams, so their level
    # is not identified by the data. Give them an informative prior instead.
    tier2_mean: float = -1.2     # ~23% per-game vs an average tier-1 team
    tier2_var: float = 0.15
    tau_tier2: float = 0.0       # tier-2 league levels do NOT drift: almost no games link them to tier 1
    # Big-stage offset: each team has delta_i ~ N(0, intl_var0) that applies ONLY in
    # international games. Domestic games never touch it; international games update it.
    intl_var0: float = 0.0       # OFF: hurt intl predictions over 1,088 games 2023-26 (tested 0.05-0.3)
    tau_intl: float = 0.0002     # drift per day of the big-stage offset
    # Off-season: at a team's first game of a new year, pull its rating towards 0 and widen it.
    season_keep: float = 0.5     # tuned on 2025; share of last season's rating kept (theta and delta)
    season_var: float = 0.25     # tuned on 2025; extra variance added at the new season
    inherit_var: float = 0.05    # extra variance when a new team id inherits a rebranded roster
    cup_w: float = 0.5           # tested 2025-26: small consistent gain on LCK/LPL games (weak evidence); evidence weight of off-season cups (KeSPA Cup, Demacia Cup, ...)
    cross_w: float = 0.25        # games between a top-league team and a lower-tier team (promotion events etc.) count as 1/4 of a game.
                                 # Fit on 2024-25 top-v-top international games, confirmed on 2026: -0.0052 +/- 0.0027 log loss there,
                                 # +0.0003 on all games (lower tiers). See league_tilt.py / crossw test.
    playoff_w: float = 1.0       # tested 1.5-3: helped early-season, hurt Aug-Sep; off. Evidence weight of a playoff game relative to regular season


@dataclass
class State:
    mean: dict = field(default_factory=dict)
    var: dict = field(default_factory=dict)
    last_seen: dict = field(default_factory=dict)
    lineup: dict = field(default_factory=dict)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class RatingEngine:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.s = State()

    # ---- state helpers -------------------------------------------------
    def inherit(self, team, lineup, date):
        """A team id seen for the first time whose 5-man lineup shares >=3 players with another team's last
        lineup (not seen for 14+ days) inherits that team's rating: treats rebrands/sales as the same roster."""
        key = ("T", team)
        if key in self.s.mean or lineup is None or len(lineup) != 5:
            return
        best, overlap = None, 0
        for k, lu in self.s.lineup.items():
            if k[0] != "T" or k == key:
                continue
            ov = len(set(lu) & set(lineup))
            if ov > overlap and (date - self.s.last_seen[k]).days >= 14:
                best, overlap = k, ov
        if best is not None and overlap >= 3:
            self.s.mean[key] = self.s.mean[best]
            self.s.var[key] = min(self.s.var[best] + self.p.inherit_var + self.p.roster_var * (5 - overlap),
                                  self.p.var_cap)
            self.s.last_seen[key] = self.s.last_seen[best]
            self.s.lineup[key] = self.s.lineup[best]

    def _ensure(self, key, kind, date):
        if key not in self.s.mean:
            self.s.mean[key] = 0.0
            self.s.var[key] = {"team": self.p.team_var0, "league": self.p.league_var0,
                               "intl": self.p.intl_var0}[kind]
            if kind == "league" and key[1] not in TIER1:
                self.s.mean[key] = self.p.tier2_mean
                self.s.var[key] = self.p.tier2_var
            self.s.last_seen[key] = date
            return
        if kind in ("team", "intl") and date.year != self.s.last_seen[key].year:
            self.s.mean[key] *= self.p.season_keep
            self.s.var[key] = min(self.s.var[key] + self.p.season_var, self.p.var_cap)
        days = max((date - self.s.last_seen[key]).total_seconds() / 86400, 0)
        tau = {"team": self.p.tau_team, "league": self.p.tau_league, "intl": self.p.tau_intl}[kind]
        if kind == "league" and key[1] not in TIER1:
            tau = self.p.tau_tier2
        self.s.var[key] = min(self.s.var[key] + tau * days, self.p.var_cap)
        self.s.last_seen[key] = date

    def _roster(self, team_key, lineup):
        if lineup is None or len(lineup) != 5:      # incomplete lineup data: don't treat as a roster change
            return
        prev = self.s.lineup.get(team_key)
        if prev is not None:
            changed = len(set(lineup) - set(prev))
            if changed:
                self.s.var[team_key] = min(self.s.var[team_key] + self.p.roster_var * changed,
                                           self.p.var_cap)
        self.s.lineup[team_key] = lineup

    def _components(self, blue, red, lb, lr, intl=False):
        comps = [(("T", blue), 1.0), (("T", red), -1.0)]
        if lb != lr:
            comps += [(("L", lb), 1.0), (("L", lr), -1.0)]
        if intl and self.p.intl_var0 > 0:
            comps += [(("I", blue), 1.0), (("I", red), -1.0)]
        return comps

    def _default_var(self, k):
        return {"T": self.p.team_var0, "L": self.p.league_var0, "I": self.p.intl_var0}[k[0]]

    # ---- public API ----------------------------------------------------
    def _var_at(self, k, as_of):
        v = self.s.var.get(k, self._default_var(k))
        if as_of is None or k not in self.s.last_seen:
            return v
        days = max((as_of - self.s.last_seen[k]).total_seconds() / 86400, 0)
        tau = {"T": self.p.tau_team, "L": self.p.tau_league, "I": self.p.tau_intl}[k[0]]
        if k[0] == "L" and k[1] not in TIER1:
            tau = self.p.tau_tier2
        if k[0] in "TI" and as_of.year != self.s.last_seen[k].year:
            v += self.p.season_var
        return min(v + tau * days, self.p.var_cap)

    def _mean_at(self, k, as_of):
        m = self.s.mean.get(k, self._prior_mean(k))
        if as_of is not None and k in self.s.last_seen and k[0] in "TI" and as_of.year != self.s.last_seen[k].year:
            m *= self.p.season_keep
        return m

    def predict(self, blue, red, league_b, league_r, neutral=False, intl=False, as_of=None):
        """P(blue wins one game). Read-only. intl=True adds the big-stage offsets.
        as_of: date of the match, so uncertainty grows with time since each team last played."""
        comps = self._components(blue, red, league_b, league_r, intl)
        m = sum(x * self._mean_at(k, as_of) for k, x in comps)
        v = sum(x * x * self._var_at(k, as_of) for k, x in comps)
        # league offsets also contribute to absolute strength even in domestic games
        m += 0 if neutral else self.p.side_adv
        return sigmoid(m / math.sqrt(1 + math.pi * v / 8)), m, v

    def _prior_mean(self, k):
        return self.p.tier2_mean if (k[0] == "L" and k[1] not in TIER1) else 0.0

    def strength(self, team, league):
        return self.s.mean.get(("T", team), 0.0) + self.s.mean.get(("L", league), self._prior_mean(("L", league)))

    def _gd_update(self, comps, z, w=1.0, c=None, sigma=None):
        """Linear-Gaussian (Kalman) update from a margin observation (blue's gold lead), in thousands."""
        c = self.p.gd_c if c is None else c
        s2 = (self.p.gd_sigma if sigma is None else sigma) ** 2 / w
        m = sum(x * self.s.mean[k] for k, x in comps)
        S = c * c * sum(x * x * self.s.var[k] for k, x in comps) + s2
        resid = z - c * m
        for key, x in comps:
            var = self.s.var[key]
            self.s.mean[key] += var * c * x * resid / S
            self.s.var[key] = max(var - (var * c * x) ** 2 / S, 1e-4)

    def update(self, date, blue, red, league_b, league_r, y, lineup_b=None, lineup_r=None,
               gd15=None, intl=False, playoff=False, margin2=None, weight=1.0):
        """Process one completed game. Returns the PRE-game probability."""
        if self.p.inherit_var >= 0:
            self.inherit(blue, lineup_b, date)
            self.inherit(red, lineup_r, date)
        keys = [(("T", blue), "team"), (("T", red), "team"),
                (("L", league_b), "league"), (("L", league_r), "league")]
        if intl and self.p.intl_var0 > 0:
            keys += [(("I", blue), "intl"), (("I", red), "intl")]
        for key, kind in keys:
            self._ensure(key, kind, date)
        self._roster(("T", blue), lineup_b)
        self._roster(("T", red), lineup_r)

        p, m, v = self.predict(blue, red, league_b, league_r, intl=intl)
        w = (self.p.playoff_w if playoff else 1.0) * weight   # tempered likelihood: game counts as w games
        if (league_b in TIER1) != (league_r in TIER1):
            w *= self.p.cross_w                                # top-league v lower-tier: thin evidence about the top
        k = w * p * (1 - p)
        denom = 1 + k * v
        comps = self._components(blue, red, league_b, league_r, intl)
        for key, x in comps:
            var = self.s.var[key]
            self.s.mean[key] += w * var * x * (y - p) / denom
            self.s.var[key] = max(var - var * var * x * x * k / denom, 1e-4)
        if self.p.gd_c and gd15 is not None and not math.isnan(gd15):
            self._gd_update(comps, gd15 / 1000, w)
        if self.p.gd_col2 and margin2 is not None and not math.isnan(margin2):
            self._gd_update(comps, margin2 / 1000, w, self.p.gd_c2, self.p.gd_sigma2)
        return p


# ---- series maths ------------------------------------------------------
# 'Form on the day': every game in a series shares one shock e ~ N(0, s^2) on the logit scale, with the
# single-game probability held fixed (see series_math.py). Fitted on 2023-24 scorelines (log-likelihood),
# tested on 2025-26: fixes Bo5 over-confidence (+0.0135 ± 0.0033 log loss) and sweep frequency.
SERIES_SPREAD_TOP = 0.7     # both teams from tier-1 leagues (or an international event)
SERIES_SPREAD_LOW = 1.0     # anything involving a lower-tier team


def series_spread(league_a: str, league_b: str) -> float:
    return SERIES_SPREAD_TOP if (league_a in TIER1 and league_b in TIER1) else SERIES_SPREAD_LOW


def series_prob(p_game: float, best_of: int, spread: float = SERIES_SPREAD_TOP) -> float:
    """P(win series). spread=0 gives the old independent-games answer."""
    from series_math import series_prob as _sp
    return _sp(p_game, best_of, spread)


def series_scores(p_game: float, best_of: int, spread: float = SERIES_SPREAD_TOP) -> dict:
    """{(a_wins, b_wins): probability} for every possible scoreline."""
    from series_math import scores as _sc
    return _sc(p_game, best_of, spread)


# ---- data loading ------------------------------------------------------
def data_file(year: int) -> str:
    """The Oracle's Elixir CSV for one season: the newest '<year>_..._OraclesElixir....csv' download
    in this folder, else 'oe_<year>.csv'."""
    import glob, os
    found = glob.glob(f"{year}_*OraclesElixir*.csv")
    return max(found, key=os.path.getmtime) if found else f"oe_{year}.csv"


def data_files(years=None) -> list:
    """One CSV per season found in this folder (or for the given years), oldest season first."""
    import glob, os, re
    if years is None:
        pat = re.compile(r"^(?:oe_)?(20\d\d)(?:_.*OraclesElixir.*)?\.csv$")
        years = sorted({int(m.group(1)) for f in glob.glob("*.csv") for m in [pat.match(os.path.basename(f))] if m})
    return [data_file(y) for y in years if os.path.exists(data_file(y))]


def load_games(path_or_paths):
    paths = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
    # newest files first, so if the same season was downloaded twice the fresher copy wins
    paths = sorted(paths, key=lambda p: __import__("os").path.getmtime(p), reverse=True)
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    # drop duplicate rows (same game, same team/player slot) from overlapping downloads
    df = df.drop_duplicates(subset=["gameid", "participantid"], keep="first")
    df["date"] = pd.to_datetime(df["date"])
    df["teamid"] = df["teamid"].fillna("name:" + df["teamname"].astype(str))
    df["year"] = df["date"].dt.year
    players = df[df.position != "team"]
    lineups = (players.groupby(["gameid", "teamid"])["playerid"]
               .apply(lambda s: tuple(sorted(s.dropna().astype(str)))).rename("lineup"))
    t = df[df.position == "team"][["gameid", "date", "league", "teamid", "teamname",
                                   "side", "result", "patch", "game", "golddiffat15", "golddiffat20", "gspd", "playoffs", "year"]].copy()
    t = t.merge(lineups, left_on=["gameid", "teamid"], right_index=True, how="left")
    # a team's "home" league = the league it plays most of its games in
    # home league per team per year (teams get promoted/relegated between seasons)
    dom = t[~t.league.isin(INTERNATIONAL)]
    home_y = dom.groupby(["teamid", "year"])["league"].agg(lambda s: s.value_counts().index[0])
    home_any = dom.groupby("teamid")["league"].agg(lambda s: s.value_counts().index[0])
    t["home_league"] = [home_y.get((tid, yr), home_any.get(tid, lg))
                        for tid, yr, lg in zip(t.teamid, t.year, t.league)]
    b = t[t.side == "Blue"].add_prefix("b_").rename(columns={"b_gameid": "gameid"})
    r = t[t.side == "Red"].add_prefix("r_").rename(columns={"r_gameid": "gameid"})
    g = b.merge(r, on="gameid")
    g = g.rename(columns={"b_date": "date", "b_league": "league", "b_result": "y",
                          "b_patch": "patch", "b_game": "game_no", "b_golddiffat15": "gd15", "b_golddiffat20": "gd20", "b_gspd": "gspd",
                          "b_playoffs": "playoffs"})
    g = g.dropna(subset=["y"]).sort_values(["date", "gameid"]).reset_index(drop=True)
    g["y"] = g["y"].astype(int)
    return g


TIER1 = {"LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL"}
CUPS = {"KeSPA Cup", "KeSPA", "DCup", "DC", "DCGI", "ASI", "WSCI"}   # off-season / exhibition-style events
INTERNATIONAL = {"MSI", "WLDs", "EWC", "FST", "ASI", "KeSPA", "DC", "DCGI"}


def _margin(g, col):
    v = getattr(g, col)
    if pd.isna(v):
        return None
    return float(v) * (23000 if col == "gspd" else 1)   # gspd is a fraction; x23000 gives the same spread as gd15


def run(games: pd.DataFrame, params: Params | None = None) -> tuple[RatingEngine, np.ndarray]:
    eng = RatingEngine(params)
    preds = np.empty(len(games))
    for i, g in enumerate(games.itertuples(index=False)):
        lb = g.b_home_league
        lr = g.r_home_league
        lub = g.b_lineup if isinstance(g.b_lineup, tuple) else None
        lur = g.r_lineup if isinstance(g.r_lineup, tuple) else None
        preds[i] = eng.update(g.date, g.b_teamid, g.r_teamid, lb, lr, g.y, lub, lur,
                              _margin(g, eng.p.gd_col),
                              intl=g.league in INTERNATIONAL, playoff=g.playoffs == 1,
                              margin2=_margin(g, eng.p.gd_col2) if eng.p.gd_col2 else None,
                              weight=eng.p.cup_w if g.league in CUPS else 1.0)
    return eng, preds
