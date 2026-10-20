"""
Series maths with 'form on the day'.

Each series gets one shared shock  e ~ N(0, s^2)  on the logit scale, applied to every game in it:
    P(team A wins game | e) = sigmoid(z + e)
z is chosen so that the AVERAGE single-game probability is still exactly p (the rating model's number):
    E_e[ sigmoid(z + e) ] = p
So Bo1 prices are untouched; only the dependence between games in a series changes.
Integrals are done with 40-point Gauss-Hermite quadrature.
"""
import math
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

_X, _W = hermegauss(40)
_W = _W / _W.sum()


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def latent_z(p, s):
    """z such that E[sigmoid(z + s*e)] = p."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    if s <= 0:
        return math.log(p / (1 - p))
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(np.dot(_W, _sig(mid + s * _X))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _indep_scores(p, bo):
    """dict {(a_wins, b_wins): prob} for independent games."""
    w = bo // 2 + 1
    q = 1 - p
    out = {}
    for j in range(w):
        c = math.comb(w - 1 + j, j)
        out[(w, j)] = c * p ** w * q ** j
        out[(j, w)] = c * q ** w * p ** j
    return out


def scores(p, bo, s):
    """Scoreline distribution with form-on-the-day spread s (s=0 -> independent games)."""
    if bo == 1:
        return {(1, 0): p, (0, 1): 1 - p}
    if s <= 0:
        return _indep_scores(p, bo)
    z = latent_z(p, s)
    ps = _sig(z + s * _X)
    out = {}
    for wi, pi in zip(_W, ps):
        for k, v in _indep_scores(float(pi), bo).items():
            out[k] = out.get(k, 0.0) + wi * v
    return out


def series_prob(p, bo, s):
    if bo == 1:
        return p
    return sum(v for (a, b), v in scores(p, bo, s).items() if a > b)
