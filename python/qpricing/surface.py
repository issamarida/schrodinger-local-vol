"""One model for the whole surface: every expiry of a day from a single solve.

The per-slice Schrödinger model fits five parameters per expiry and says nothing about how
expiries relate. Here every expiry of a quote date comes from one diffusion, in three variants:

* ``surface_homogeneous``: one time-independent well sigma(x). Five parameters for the whole
  surface; every expiry is the same wavefunction read at a different imaginary time T. The
  strictest test of whether the well carries structure across maturities.
* ``surface_clock``: sigma(x, t) = lambda(t) sigma(x). With tau(t) = int_0^t lambda^2 ds this is
  the homogeneous process on the clock tau, so expiry i is priced at tau_i from the same solve.
  tau_i = sum_{j <= i} (T_j - T_{j-1}) exp(u_j) is increasing by construction. 5 + N parameters.
* ``surface_piecewise``: a time-dependent well. Each interval between consecutive expiries has
  its own five-parameter local variance, bootstrapped expiry by expiry (the classical way to
  calibrate a local-vol term structure). 5 N parameters, like per-slice fitting.

All three are a single positive diffusion, so their prices carry neither butterfly nor calendar
arbitrage for any parameter values; the fitted surfaces are checked numerically anyway.

Tests, per quote date: the in-sample fit, the calendar check, and leave-one-expiry-out: refit
without an interior expiry, then price it (the clock is interpolated linearly in T; the piecewise
well simply runs through the gap). The practitioner benchmark interpolates per-slice SVI or
ad-hoc smiles linearly in total implied variance at fixed k.

Usage::

    uv run qp-surface                  # 2022, every day
    uv run qp-surface --period 2019-08
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from qpricing import _qpcore
from qpricing.arbitrage import TOL, on_strikes
from qpricing.data import PERIODS, load_dataset, processed_path
from qpricing.models import SVI, AdHocBlackScholes, Schrodinger, Slice, SVIPrice, black76

MIN_QUOTES = 10
CALIB_GRID = _qpcore.GridSpec(1600, 400)
EVAL_GRID = _qpcore.GridSpec(3200, 1200)
U_BOUND = 4.0


def _inputs(slices: list[Slice]):
    return ([s.F for s in slices], [s.df for s in slices], [s.strikes for s in slices],
            [s.is_call for s in slices])


def _atm_variance(theta: np.ndarray) -> float:
    p = Schrodinger.to_params(theta)
    return p.a + p.b * (-p.rho * p.m + np.sqrt(p.m ** 2 + p.s ** 2))


class SharedWell:
    """One time-independent well, optionally on a deterministic clock."""

    def __init__(self, clock: bool):
        self.clock = clock
        self.name = "surface_clock" if clock else "surface_homogeneous"

    def taus(self, u: np.ndarray, T: np.ndarray) -> np.ndarray:
        if not self.clock:
            return T.copy()
        return np.cumsum(np.diff(np.r_[0.0, T]) * np.exp(u))

    def _price_at_taus(self, theta, taus, slices, grid):
        return _qpcore.schrodinger_price_surface(Schrodinger.to_params(theta), list(taus),
                                                 *_inputs(slices), grid)

    def fit(self, slices: list[Slice]) -> dict:
        T = np.array([s.T for s in slices])
        mid = np.concatenate([s.mid for s in slices])
        n = len(slices) if self.clock else 0
        # Start from a per-slice fit of the expiry nearest 14 days; set each clock so that
        # sigma(0)^2 tau_i matches that expiry's ATM total variance.
        ref = slices[int(np.argmin(np.abs(T - 14 / 365)))]
        theta0 = Schrodinger().calibrate(ref)
        x0 = theta0
        if self.clock:
            w = np.maximum.accumulate([s.atm_vol() ** 2 * s.T for s in slices]) / _atm_variance(
                theta0)
            dT = np.diff(np.r_[0.0, T])
            dtau = np.maximum(np.diff(np.r_[0.0, w]), 1e-3 * dT)
            x0 = np.r_[theta0, np.clip(np.log(dtau / dT), -U_BOUND + 0.1, U_BOUND - 0.1)]
        lower = np.r_[Schrodinger.lower, np.full(n, -U_BOUND)]
        upper = np.r_[Schrodinger.upper, np.full(n, U_BOUND)]
        scale = np.r_[ref.atm_vol() ** 2, 1.0, 0.5, 0.05, 0.05, np.full(n, 0.3)]

        def resid(x):
            p = self._price_at_taus(x[:5], self.taus(x[5:], T), slices, CALIB_GRID)
            return np.concatenate(p) - mid

        res = least_squares(resid, np.clip(x0, lower + 1e-9, upper - 1e-9),
                            bounds=(lower, upper), method="trf", x_scale=scale, diff_step=1e-4,
                            max_nfev=150)
        return {"theta": res.x[:5], "u": res.x[5:], "T": T, "n_params": 5 + n}

    def price(self, state: dict, slices: list[Slice], grid=EVAL_GRID) -> list[np.ndarray]:
        """Prices slices at any maturities inside the fitted range (clock interpolated in T)."""
        taus = self.taus(state["u"], state["T"])
        tau = np.interp([s.T for s in slices], np.r_[0.0, state["T"]], np.r_[0.0, taus])
        return self._price_at_taus(state["theta"], tau, slices, grid)

    def loo_price(self, state: dict, slices: list[Slice], i: int) -> np.ndarray:
        return self.price(self.fit(slices[:i] + slices[i + 1:]), [slices[i]])[0]


class PiecewiseWell:
    """Time-dependent well, piecewise constant between expiries, bootstrapped."""

    name = "surface_piecewise"

    @staticmethod
    def _price(thetas, ends, slices, grid):
        return _qpcore.schrodinger_price_piecewise(
            [Schrodinger.to_params(t) for t in thetas], list(ends), [s.T for s in slices],
            *_inputs(slices), grid)

    def _fit_segment(self, thetas: list, ends: list, s: Slice, w_prev: float, t_prev: float,
                     start: np.ndarray | None) -> np.ndarray:
        """Fits the well on (ends[-1], s.T] to slice s, with the earlier segments fixed."""
        if start is None:
            return Schrodinger().calibrate(s)
        # Rescale the previous well to the forward variance implied by the ATM term structure.
        w = s.atm_vol() ** 2 * s.T
        fwd = max((w - w_prev) / (s.T - t_prev), 0.05 * _atm_variance(start))
        x0 = Schrodinger.with_level(start, np.sqrt(fwd / _atm_variance(start)))
        x0 = np.clip(x0, Schrodinger.lower + 1e-9, Schrodinger.upper - 1e-9)
        scale = np.array([max(fwd, 1e-4), 1.0, 0.5, 0.05, 0.05])

        def resid(th):
            return self._price([*thetas, th], [*ends, s.T], [s], CALIB_GRID)[0] - s.mid

        res = least_squares(resid, x0, bounds=(Schrodinger.lower, Schrodinger.upper),
                            method="trf", x_scale=scale, diff_step=1e-4, max_nfev=200)
        return res.x

    def fit(self, slices: list[Slice]) -> dict:
        thetas, ends = [], []
        for s in slices:
            prev = ends[-1] if ends else 0.0
            w_prev = slices[len(ends) - 1].atm_vol() ** 2 * prev if ends else 0.0
            th = self._fit_segment(thetas, ends, s, w_prev, prev, thetas[-1] if thetas else None)
            thetas.append(th)
            ends.append(s.T)
        return {"thetas": thetas, "ends": ends, "n_params": 5 * len(thetas)}

    def price(self, state: dict, slices: list[Slice], grid=EVAL_GRID) -> list[np.ndarray]:
        return self._price(state["thetas"], state["ends"], slices, grid)

    def loo_price(self, state: dict, slices: list[Slice], i: int) -> np.ndarray:
        """Without expiry i, segment i+1 spans (T_{i-1}, T_{i+1}]; earlier ones are unchanged."""
        thetas, ends = state["thetas"][:i], state["ends"][:i]
        prev = ends[-1] if ends else 0.0
        w_prev = slices[i - 1].atm_vol() ** 2 * prev if ends else 0.0
        th = self._fit_segment(thetas, ends, slices[i + 1], w_prev, prev,
                               thetas[-1] if thetas else None)
        return self._price([*thetas, th], [*ends, slices[i + 1].T], [slices[i]], EVAL_GRID)[0]


SURFACE_MODELS = (SharedWell(False), SharedWell(True), PiecewiseWell())


def calendar_violations(model, state: dict, slices: list[Slice]) -> int:
    """Adjacent expiry pairs whose normalised OTM prices fall with T somewhere on the overlap."""
    bad = 0
    for s1, s2 in zip(slices[:-1], slices[1:], strict=True):
        lo, hi = max(s1.k.min(), s2.k.min()), min(s1.k.max(), s2.k.max())
        if hi > lo:
            k = np.linspace(lo, hi, 201)
            g1, g2 = (on_strikes(s, s.F * np.exp(k)) for s in (s1, s2))
            n1, n2 = (p / (s.df * s.F) for p, s in zip(model.price(state, [g1, g2]), (s1, s2),
                                                       strict=True))
            bad += int(((n2 - n1) < -TOL).any())
    return bad


def variance_interpolation(model, p1, p2, s1: Slice, s2: Slice, s: Slice) -> np.ndarray:
    """Practitioner interpolation: total implied variance linear in T at fixed k = ln(K/F)."""
    def w(p, sl):
        if isinstance(model, SVI):
            return np.maximum(SVI.variance(p, s.k), 1e-12) * sl.T
        return AdHocBlackScholes._vols(p, s.k) ** 2 * sl.T

    a = (s.T - s1.T) / (s2.T - s1.T)
    tot = (1 - a) * w(p1, s1) + a * w(p2, s2)
    return black76(s.is_call, s.F, s.strikes, s.T, np.sqrt(tot / s.T), s.df)


def _day(args) -> tuple[list[dict], list[dict], list[dict]]:
    day, quotes, params = args
    frames = [(k, g) for k, g in quotes.groupby(["root", "expiration"]) if len(g) >= MIN_QUOTES]
    ordered = sorted([(k, Slice.from_frame(g)) for k, g in frames], key=lambda kv: kv[1].T)
    keys, sl, seen = [], [], set()
    for k, s in ordered:  # one slice per maturity
        if s.T not in seen:
            keys.append(k)
            sl.append(s)
            seen.add(s.T)
    per_slice = Schrodinger()
    fits, days, loo = [], [], []

    for model in SURFACE_MODELS:
        t0 = time.perf_counter()
        state = model.fit(sl)
        secs = time.perf_counter() - t0
        for key, s, p in zip(keys, sl, model.price(state, sl), strict=True):
            theta = params.get((*key, "schrodinger"))
            fits.append({"quote_date": day, "root": key[0], "expiration": key[1],
                         "days": s.days, "model": model.name, "n": len(s.mid),
                         "sae": float(np.abs(p - s.mid).sum()),
                         "per_slice_sae": float(np.abs(per_slice.price(theta, s) - s.mid).sum())
                         if theta is not None else np.nan})
        days.append({"quote_date": day, "model": model.name, "n_expiries": len(sl),
                     "n_params": state["n_params"], "seconds": secs,
                     "calendar_pairs": len(sl) - 1,
                     "calendar_violations": calendar_violations(model, state, sl)})
        for i in range(1, len(sl) - 1):
            p = model.loo_price(state, sl, i)
            loo.append({"quote_date": day, "root": keys[i][0], "expiration": keys[i][1],
                        "days": sl[i].days, "model": model.name, "n": len(sl[i].mid),
                        "sae": float(np.abs(p - sl[i].mid).sum())})

    for i in range(1, len(sl) - 1):
        for M in (SVI, SVIPrice, AdHocBlackScholes):
            p1, p2 = params.get((*keys[i - 1], M.name)), params.get((*keys[i + 1], M.name))
            if p1 is None or p2 is None:
                continue
            p = variance_interpolation(M(), p1, p2, sl[i - 1], sl[i + 1], sl[i])
            loo.append({"quote_date": day, "root": keys[i][0], "expiration": keys[i][1],
                        "days": sl[i].days, "model": M.name, "n": len(sl[i].mid),
                        "sae": float(np.abs(p - sl[i].mid).sum())})
    return fits, days, loo


def run(period: str, workers: int | None = None, start: str | None = None):
    data = load_dataset(period=period)
    if start is not None:
        data = data[data.quote_date >= pd.Timestamp(start)]
    params = pd.read_parquet(processed_path("backtest_params", period))
    jobs = []
    for day, quotes in data.groupby("quote_date"):
        p = params[params.calib_date == day]
        lookup = {(r.root, r.expiration, r.model): np.asarray(r.params, float)
                  for r in p.itertuples() if np.all(np.isfinite(r.params))}
        jobs.append((day, quotes, lookup))
    fits, days, loo = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_day, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            f, d, lo = fut.result()
            fits += f
            days += d
            loo += lo
            print(f"[{n}/{len(jobs)}]", flush=True)
    return pd.DataFrame(fits), pd.DataFrame(days), pd.DataFrame(loo)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--period", choices=PERIODS, default="2022H2")
    ap.add_argument("--start", default=None, help="first quote date (default: whole period)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    fits, days, loo = run(args.period, args.workers, args.start)
    fits.to_parquet(processed_path("surface_fits", args.period), index=False)
    days.to_parquet(processed_path("surface_days", args.period), index=False)
    loo.to_parquet(processed_path("surface_loo", args.period), index=False)
    print(fits.groupby("model")[["sae", "per_slice_sae", "n"]].sum().eval("mae = sae / n")
          .eval("per_slice_mae = per_slice_sae / n"))
    print(days.groupby("model")[["calendar_violations", "calendar_pairs", "n_params",
                                 "seconds"]].mean())
    print(loo.groupby("model")[["sae", "n"]].sum().eval("mae = sae / n"))


if __name__ == "__main__":
    main()
