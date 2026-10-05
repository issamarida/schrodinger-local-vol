"""Volatility regimes, defined from the option data itself.

For every quote date we compute a 30-day at-the-money implied volatility (a VIX-like level):
the ATM vol of each expiry is interpolated at k = ln(K/F) = 0 from the two out-of-the-money
quotes that bracket the forward, converted to total variance, and interpolated linearly in T to
30 calendar days. Days are then split into terciles of that level: low / medium / high.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qpricing import _qpcore

REGIMES = ("low", "medium", "high")


def _slice_atm_vol(g: pd.DataFrame) -> float:
    g = g.sort_values("k")
    below, above = g[g.k < 0], g[g.k >= 0]
    if below.empty or above.empty:
        return np.nan
    lo, hi = below.iloc[-1], above.iloc[0]
    vols = [
        _qpcore.black76_implied_vol(bool(r.is_call), float(r.F), float(r.strike), float(r["T"]),
                                    float(r.mid), float(r["df"]))
        for r in (lo, hi)
    ]
    if not all(np.isfinite(vols)):
        return np.nan
    w = -lo.k / (hi.k - lo.k)
    return float((1 - w) * vols[0] + w * vols[1])


def atm_vol_30d(data: pd.DataFrame) -> pd.Series:
    """30-day ATM implied vol per quote date."""
    out = {}
    for day, d in data.groupby("quote_date"):
        rows = []
        for (_, _), g in d.groupby(["root", "expiration"]):
            v = _slice_atm_vol(g)
            if np.isfinite(v):
                rows.append((float(g["T"].iloc[0]), v * v * float(g["T"].iloc[0])))
        if len(rows) < 2:
            continue
        T, w = np.array(sorted(rows)).T
        t30 = 30.0 / 365.0
        out[day] = float(np.sqrt(np.interp(t30, T, w) / t30))
    return pd.Series(out, name="atm_vol_30d").sort_index()


def classify(levels: pd.Series) -> pd.DataFrame:
    """Tercile regimes; returns the level, the regime label and the cut-points used."""
    q1, q2 = levels.quantile([1 / 3, 2 / 3])
    regime = pd.cut(levels, [-np.inf, q1, q2, np.inf], labels=list(REGIMES))
    out = pd.DataFrame({"atm_vol_30d": levels, "regime": regime.astype(str)})
    out.attrs["cuts"] = (float(q1), float(q2))
    return out
