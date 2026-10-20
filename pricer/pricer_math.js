// Rift Pricer maths: a line-for-line port of ratings.py predict() + series_prob + power de-vig.
const RP = (() => {
  const DAY = 86400000;
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const parseDate = s => new Date(s.slice(0, 10) + "T00:00:00Z");

  function make(D) {
    const P = D.params, T1 = new Set(D.tier1);
    const byId = Object.fromEntries(D.teams.map(t => [t.id, t]));
    const days = (asOf, last) => Math.max((asOf - parseDate(last)) / DAY, 0);

    function team(t, asOf) {
      const newYear = asOf.getUTCFullYear() !== parseDate(t.last).getUTCFullYear();
      const mean = t.mean * (newYear ? P.season_keep : 1);
      const v = Math.min(t.var + (newYear ? P.season_var : 0) + P.tau_team * days(asOf, t.last), P.var_cap);
      return { mean, var: v };
    }
    function league(code, asOf) {
      const L = D.leagues[code], tier1 = T1.has(code);
      if (!L) return { mean: tier1 ? 0 : P.tier2_mean, var: tier1 ? P.league_var0 : P.tier2_var };
      const tau = tier1 ? P.tau_league : P.tau_tier2;
      return { mean: L.mean, var: Math.min(L.var + tau * days(asOf, L.last), P.var_cap) };
    }
    // P(team a wins one game when a is on blue side)
    function onBlue(a, b, asOf) {
      const ta = team(a, asOf), tb = team(b, asOf);
      let m = ta.mean - tb.mean, v = ta.var + tb.var;
      let la = null, lb = null;
      if (a.league !== b.league) {
        la = league(a.league, asOf); lb = league(b.league, asOf);
        m += la.mean - lb.mean; v += la.var + lb.var;
      }
      const z = (m + P.side_adv) / Math.sqrt(1 + Math.PI * v / 8);
      return { p: sigmoid(z), m, v, ta, tb, la, lb, z };
    }
    function game(aId, bId, asOf) {
      const a = byId[aId], b = byId[bId];
      const blue = onBlue(a, b, asOf);
      const redSide = onBlue(b, a, asOf);           // b on blue
      const pRed = 1 - redSide.p;
      return { p: (blue.p + pRed) / 2, pBlue: blue.p, pRed, detail: blue, detailRed: redSide, a, b };
    }
    if (D.series) setSeries(D.series);
    const spread = (a, b) => (T1.has(a.league) && T1.has(b.league)) ? D.series.spread_top : D.series.spread_low;
    return { game, byId, team, league, spread };
  }

  function comb(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; }
  // Independent-games scorelines: [{a, b, p}] e.g. 3-1
  function indepScores(p, bo) {
    const w = (bo + 1) / 2, q = 1 - p, out = [];
    for (let j = 0; j < w; j++) out.push({ a: w, b: j, p: comb(w - 1 + j, j) * Math.pow(p, w) * Math.pow(q, j) });
    for (let j = 0; j < w; j++) out.push({ a: j, b: w, p: comb(w - 1 + j, j) * Math.pow(q, w) * Math.pow(p, j) });
    return out;
  }
  // 'Form on the day' (see series_math.py): one shared logit shock e ~ N(0, s^2) per series,
  // with z chosen so that E[sigmoid(z + s e)] = p, i.e. the single-game probability is unchanged.
  let GH = null;                                   // {x: [...], w: [...]} Gauss-Hermite nodes from ratings.json
  function setSeries(cfg) { GH = { x: cfg.gh_x, w: cfg.gh_w }; }
  function latentZ(p, s) {
    p = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
    if (!(s > 0)) return Math.log(p / (1 - p));
    let lo = -40, hi = 40;
    for (let n = 0; n < 80; n++) {
      const mid = (lo + hi) / 2; let m = 0;
      for (let i = 0; i < GH.x.length; i++) m += GH.w[i] / (1 + Math.exp(-(mid + s * GH.x[i])));
      if (m < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }
  function scores(p, bo, s = 0) {
    if (bo <= 1) return [{ a: 1, b: 0, p }, { a: 0, b: 1, p: 1 - p }];
    if (!(s > 0) || !GH) return indepScores(p, bo);
    const z = latentZ(p, s); let out = null;
    for (let i = 0; i < GH.x.length; i++) {
      const pi = 1 / (1 + Math.exp(-(z + s * GH.x[i])));
      const sc = indepScores(pi, bo);
      if (!out) out = sc.map(r => ({ a: r.a, b: r.b, p: 0 }));
      sc.forEach((r, k) => { out[k].p += GH.w[i] * r.p; });
    }
    return out;
  }
  function series(p, bo, s = 0) {
    if (bo <= 1) return p;
    return scores(p, bo, s).filter(r => r.a > r.b).reduce((t, r) => t + r.p, 0);
  }
  function parseOdds(s) {
    s = String(s || "").trim();
    if (!s) return null;
    if (s.includes("/")) { const [n, d] = s.split("/").map(Number); return d > 0 && n > 0 ? n / d + 1 : null; }
    const v = Number(s);
    if (!isFinite(v)) return null;
    if (s.startsWith("+") || v >= 100) return v / 100 + 1;
    if (v <= -100) return 100 / Math.abs(v) + 1;
    return v > 1 ? v : null;
  }
  function devig(o1, o2) {
    const i1 = 1 / o1, i2 = 1 / o2; let lo = 0.5, hi = 3, k = 1;
    for (let n = 0; n < 100; n++) { k = (lo + hi) / 2; if (Math.pow(i1, k) + Math.pow(i2, k) > 1) lo = k; else hi = k; }
    return { p1: Math.pow(i1, k), p2: Math.pow(i2, k), k, over: i1 + i2 - 1 };
  }
  return { make, series, scores, setSeries, latentZ, parseOdds, devig, sigmoid };
})();
if (typeof module !== "undefined") module.exports = RP;
