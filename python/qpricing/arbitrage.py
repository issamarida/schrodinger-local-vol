"""Static-arbitrage checks of calibrated smiles: butterflies (negative densities) and calendars.

A model's smile is free of butterfly arbitrage when undiscounted call prices are convex in the
strike, i.e. the implied risk-neutral density is non-negative. The check is model-agnostic: each
calibrated slice is priced on a 1-index-point strike grid, OTM puts are turned into calls with
put-call parity, and every butterfly C(K - h) - 2 C(K) + C(K + h) must be >= 0. Two ranges:

* ``quoted``: between the lowest and highest strike the model was calibrated on;
* ``wings``: the quoted range extended by half its width on each side (in log-moneyness), where a
  smile is extrapolated to price tails, variance swaps or VIX-style integrals.

The Schrödinger model is a diffusion with strictly positive local variance, so its density is
positive by construction (and the discrete pricer integrates payoffs against a non-negative
density, so the property survives discretisation). The checks confirm that instead of assuming it.

Calendar arbitrage: with deterministic carry, undiscounted OTM prices normalised by the forward,
E[(e^k - e^X)^+] and E[(e^X - e^k)^+], must not decrease with maturity at fixed k = ln(K/F).
Per-slice models are checked expiry against next expiry on the overlap of their quoted ranges.

Usage::

    uv run qp-arbitrage                    # 2022 holdout, from the saved backtest parameters
    uv run qp-arbitrage --period 2019-08
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from qpricing.data import PERIODS, load_dataset, processed_path
from qpricing.models import (
    SVI,
    AdHocBlackScholes,
    BlackScholes,
    Heston,
    Schrodinger,
    Slice,
    SVIPrice,
)

CHECKED = (AdHocBlackScholes, SVI, SVIPrice, Heston, Schrodinger, BlackScholes)
FLY_TOL = 1e-10     # butterfly tolerance, fraction of the forward (Black-76 noise is ~1e-16)
TOL = 1e-7          # calendar tolerance, fraction of the forward (~0.0004 index points)
STEP = 1.0          # strike spacing of the butterfly grid, index points
MIN_QUOTES = 10


def strike_grid(s: Slice, wings: bool) -> np.ndarray:
    lo, hi = np.log(s.strikes.min() / s.F), np.log(s.strikes.max() / s.F)
    if wings:
        half = 0.5 * (hi - lo)
        lo, hi = lo - half, hi + half
    return np.arange(np.ceil(s.F * np.exp(lo)), np.floor(s.F * np.exp(hi)) + STEP, STEP)


def on_strikes(s: Slice, strikes: np.ndarray) -> Slice:
    """The slice's market inputs with an arbitrary strike grid (OTM option at every strike)."""
    return Slice(s.F, s.S, s.T, s.days, s.r, s.df, strikes, strikes >= s.F,
                 np.zeros_like(strikes))


def undiscounted_calls(s: Slice, otm_prices: np.ndarray) -> np.ndarray:
    fwd = otm_prices / s.df
    return np.where(s.is_call, fwd, fwd + s.F - s.strikes)


def butterflies(s: Slice, otm_prices: np.ndarray) -> np.ndarray:
    """C(K - h) - 2 C(K) + C(K + h) on a uniform strike grid (h**2 times the density)."""
    c = undiscounted_calls(s, otm_prices)
    return c[:-2] - 2.0 * c[1:-1] + c[2:]


def butterfly_stats(model, params: np.ndarray, s: Slice, wings: bool) -> dict:
    grid = on_strikes(s, strike_grid(s, wings))
    fly = butterflies(grid, model.price(params, grid))
    bad = fly < -FLY_TOL * s.F
    return {"violation": bool(bad.any()),
            "neg_mass": float(-np.where(bad, fly, 0.0).sum() / STEP),  # negative probability
            "worst_fly": float(fly.min())}


def market_butterflies(g: pd.DataFrame) -> dict:
    """Butterflies on adjacent listed strikes, priced from the quotes themselves.

    OTM puts are turned into calls with put-call parity (the same constant shifts bid, ask and
    mid). Weights (K3 - K2)/(K3 - K1), -1, (K2 - K1)/(K3 - K1). ``violation``: negative at mids;
    ``executable``: still negative after buying the wings at the ask and selling the body at the
    bid.
    """
    g = g.sort_values("strike").drop_duplicates("strike")
    K = g.strike.to_numpy(float)
    parity = np.where(g.is_call, 0.0, g["df"] * (g.F - g.strike))
    bid, ask, mid = (g[c].to_numpy(float) + parity for c in ("bid", "ask", "mid"))
    w1 = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
    w3 = 1.0 - w1
    at_mid = w1 * mid[:-2] - mid[1:-1] + w3 * mid[2:]
    at_ask = w1 * ask[:-2] - bid[1:-1] + w3 * ask[2:]
    return {"violation": bool((at_mid < 0).any()), "executable": bool((at_ask < 0).any())}


def normalised_otm(model, params: np.ndarray, s: Slice, k: np.ndarray) -> np.ndarray:
    grid = on_strikes(s, s.F * np.exp(k))
    return model.price(params, grid) / (s.df * s.F)


def _day(args) -> tuple[list[dict], list[dict]]:
    day, quotes, params = args
    models = {M.name: M() for M in CHECKED}
    frames = {key: g for key, g in quotes.groupby(["root", "expiration"]) if len(g) >= MIN_QUOTES}
    slices = {key: Slice.from_frame(g) for key, g in frames.items()}
    flies, cals = [], []
    for key, s in slices.items():
        flies.append({"quote_date": day, "root": key[0], "expiration": key[1], "days": s.days,
                      "model": "market", "range": "quoted", **market_butterflies(frames[key])})
        for name, model in models.items():
            theta = params.get((key[0], key[1], name))
            if theta is None or not np.all(np.isfinite(theta)):
                continue
            for wings in (False, True):
                try:
                    st = butterfly_stats(model, theta, s, wings)
                except (ValueError, RuntimeError):
                    continue
                flies.append({"quote_date": day, "root": key[0], "expiration": key[1],
                              "days": s.days, "model": name,
                              "range": "wings" if wings else "quoted", **st})

    # Calendar: adjacent expiries in time to expiry, on the overlap of their quoted k-ranges.
    ordered = sorted(slices.items(), key=lambda kv: kv[1].T)
    for (k1, s1), (k2, s2) in zip(ordered[:-1], ordered[1:], strict=True):
        if not s2.T > s1.T:
            continue
        lo = max(s1.k.min(), s2.k.min())
        hi = min(s1.k.max(), s2.k.max())
        if not hi > lo:
            continue
        k = np.linspace(lo, hi, 201)
        for name, model in models.items():
            t1, t2 = params.get((*k1, name)), params.get((*k2, name))
            if t1 is None or t2 is None or not (np.all(np.isfinite(t1))
                                                and np.all(np.isfinite(t2))):
                continue
            gap = normalised_otm(model, t2, s2, k) - normalised_otm(model, t1, s1, k)
            cals.append({"quote_date": day, "days_1": s1.days, "days_2": s2.days,
                         "model": name, "violation": bool((gap < -TOL).any()),
                         "worst_gap_pts": float(gap.min() * s1.F)})
    return flies, cals


def run(period: str = "2022H2", start: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_dataset(period=period)
    params = pd.read_parquet(processed_path("backtest_params", period))
    if start is not None:
        data = data[data.quote_date >= pd.Timestamp(start)]
    jobs = []
    for day, quotes in data.groupby("quote_date"):
        p = params[params.calib_date == day]
        lookup = {(r.root, r.expiration, r.model): np.asarray(r.params, float)
                  for r in p.itertuples()}
        jobs.append((day, quotes, lookup))
    flies, cals = [], []
    with ProcessPoolExecutor() as pool:
        for f, c in pool.map(_day, jobs):
            flies += f
            cals += c
    return pd.DataFrame(flies), pd.DataFrame(cals)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--period", choices=PERIODS, default="2022H2")
    ap.add_argument("--start", default=None, help="first quote date (default: whole period)")
    args = ap.parse_args(argv)
    flies, cals = run(args.period, args.start)
    flies.to_parquet(processed_path("butterflies", args.period), index=False)
    cals.to_parquet(processed_path("calendars", args.period), index=False)
    print(flies.groupby(["range", "model"]).violation.mean().unstack(0).round(4))
    print(cals.groupby("model").violation.mean().round(4))


if __name__ == "__main__":
    main()
