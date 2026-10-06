"""Daily delta-hedging backtest: whose delta leaves the smallest hedging error?

For every out-of-the-money option in the calibration universe on day t whose contract is still
quoted on the next trading day, a position of one option is hedged with ``delta`` futures on the
same expiry's forward and held to the next close:

    error = (V_{t+1} - V_t) - delta * (F_{t+1} - F_t)

V is the bid-ask mid and F the put-call-parity forward of that expiry on each day. Every model's
delta is dV/dF from its own day-t calibration (central bump of F by 0.1%), so it encodes how that
model expects the smile to move when the index moves:

* ``black_scholes``: Black-76 delta at the option's own implied vol. Sticky strike.
* ``minimum_variance``: Hull & White (2017). The BS delta plus a correction for the average
  co-movement of implied vol with the index,
  delta_MV = delta_BS + vega / (F sqrt(T)) * (a + b delta_BS + c delta_BS^2), with (a, b, c)
  fitted by least squares on the development window (Jul-Aug 2022) separately for calls and
  puts, and applied unchanged to the holdout and to 2019.
* ``adhoc_bs``, ``svi``, ``svi_price``, ``heston``: parameters frozen, so the smile moves with
  the forward in log-moneyness (sticky moneyness).
* ``schrodinger``: the local-vol delta. The local variance is a fixed function of the absolute
  forward level, so a bump of F shifts the calibrated SVI centre m by -ln(1 + bump).
* ``schrodinger_floating``: the same calibration with the smile moving with the forward instead
  (params frozen), which isolates the effect of the dynamics from the effect of the fit.

The comparison metric is Hull & White's gain, 1 - SSE(model) / SSE(Black-Scholes), with day-block
bootstrap intervals.

Usage::

    uv run qp-hedge                 # 2022 (MV coefficients from Jul-Aug, scored on Sep-Dec)
    uv run qp-hedge --period 2019-08
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import ndtr

from qpricing.data import PERIODS, load_all_quotes, load_dataset, processed_path
from qpricing.models import (
    SVI,
    AdHocBlackScholes,
    Heston,
    Schrodinger,
    Slice,
    SVIPrice,
    implied_vols,
)

BUMP = 1e-3
MIN_QUOTES = 10
DEV_END = pd.Timestamp("2022-08-31")
FROZEN_SMILE_MODELS = (AdHocBlackScholes, SVI, SVIPrice, Heston)
DELTAS = ["black_scholes", "minimum_variance", "adhoc_bs", "svi", "svi_price", "heston",
          "schrodinger", "schrodinger_floating"]
_KEY = ["root", "expiration", "strike", "is_call"]


def bumped(s: Slice, rel: float) -> Slice:
    return Slice(s.F * (1 + rel), s.S * (1 + rel), s.T, s.days, s.r, s.df, s.strikes, s.is_call,
                 s.mid)


def frozen_delta(model, params: np.ndarray, s: Slice) -> np.ndarray:
    """dV/dF with the model's parameters frozen (the smile moves with the forward)."""
    up, dn = model.price(params, bumped(s, BUMP)), model.price(params, bumped(s, -BUMP))
    return (up - dn) / (2 * BUMP * s.F)


def local_vol_delta(params: np.ndarray, s: Slice, model: Schrodinger) -> np.ndarray:
    """dV/dF with the local variance fixed in absolute index level (shift the SVI centre m)."""
    def shifted(rel):
        p = np.array(params, float)
        p[3] -= np.log1p(rel)
        return model.price(p, bumped(s, rel))

    return (shifted(BUMP) - shifted(-BUMP)) / (2 * BUMP * s.F)


def black_greeks(s: Slice, iv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Black-76 delta dV/dF and vega dV/dsigma at each option's own implied vol."""
    sd = iv * np.sqrt(s.T)
    d1 = (np.log(s.F / s.strikes) + 0.5 * sd * sd) / sd
    delta = s.df * np.where(s.is_call, ndtr(d1), ndtr(d1) - 1.0)
    vega = s.df * s.F * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(s.T)
    return delta, vega


def _day(args) -> pd.DataFrame:
    today, tomorrow, params = args
    schrod = Schrodinger()
    models = {M.name: M() for M in FROZEN_SMILE_MODELS}
    nxt = tomorrow.set_index(_KEY).mid.rename("V1")
    fwd_next = tomorrow.groupby(["root", "expiration"]).F.first().rename("F1")
    out = []
    for key, g in today.groupby(["root", "expiration"]):
        if len(g) < MIN_QUOTES or key not in fwd_next.index:
            continue
        s = Slice.from_frame(g)
        iv = implied_vols(s)
        rows = g[["quote_date", *_KEY, "days", "T", "F", "mid"]].rename(columns={"mid": "V0"})
        rows = rows.reset_index(drop=True)
        rows["iv"] = iv
        rows["delta_black_scholes"], rows["vega"] = black_greeks(s, iv)
        for name, model in models.items():
            theta = params.get((*key, name))
            if theta is not None and np.all(np.isfinite(theta)):
                rows[f"delta_{name}"] = frozen_delta(model, theta, s)
        theta = params.get((*key, "schrodinger"))
        if theta is not None and np.all(np.isfinite(theta)):
            rows["delta_schrodinger"] = local_vol_delta(theta, s, schrod)
            rows["delta_schrodinger_floating"] = frozen_delta(schrod, theta, s)
        rows["F1"] = fwd_next.loc[key]
        out.append(rows.join(nxt, on=_KEY))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    return df[np.isfinite(df.V1) & np.isfinite(df.iv)]


def build_panel(period: str) -> pd.DataFrame:
    """One row per (option, day) with V_t, V_{t+1}, F_t, F_{t+1} and every model's delta."""
    data = load_dataset(period=period)
    allq = load_all_quotes(period)
    params = pd.read_parquet(processed_path("backtest_params", period))
    days = sorted(data.quote_date.unique())
    by_day = dict(list(allq.groupby("quote_date")))
    jobs = []
    for d, d1 in zip(days[:-1], days[1:], strict=True):
        p = params[params.calib_date == d]
        lookup = {(r.root, r.expiration, r.model): np.asarray(r.params, float)
                  for r in p.itertuples()}
        jobs.append((data[data.quote_date == d], by_day[d1], lookup))
    with ProcessPoolExecutor() as pool:
        frames = list(pool.map(_day, jobs))
    panel = pd.concat(frames, ignore_index=True)
    panel["dV"] = panel.V1 - panel.V0
    panel["dF"] = panel.F1 - panel.F
    return panel


def fit_minimum_variance(dev: pd.DataFrame) -> dict[bool, np.ndarray]:
    """Hull-White (2017) coefficients (a, b, c) per option type, by ordinary least squares."""
    coefs = {}
    for is_call, g in dev.groupby("is_call"):
        d = g.delta_black_scholes.to_numpy()
        scale = (g.vega / np.sqrt(g["T"]) * g.dF / g.F).to_numpy()
        X = scale[:, None] * np.column_stack([np.ones_like(d), d, d * d])
        y = (g.dV - g.delta_black_scholes * g.dF).to_numpy()
        coefs[bool(is_call)] = np.linalg.lstsq(X, y, rcond=None)[0]
    return coefs


def add_minimum_variance(panel: pd.DataFrame, coefs: dict[bool, np.ndarray]) -> pd.DataFrame:
    d = panel.delta_black_scholes
    abc = np.vstack([coefs[bool(c)] for c in panel.is_call])
    corr = abc[:, 0] + abc[:, 1] * d + abc[:, 2] * d * d
    panel["delta_minimum_variance"] = d + panel.vega / (panel.F * np.sqrt(panel["T"])) * corr
    return panel


def hedging_errors(panel: pd.DataFrame) -> pd.DataFrame:
    err = pd.DataFrame({m: panel.dV - panel[f"delta_{m}"] * panel.dF for m in DELTAS})
    err["unhedged"] = panel.dV
    return err


def bootstrap_gain(panel: pd.DataFrame, reps: int = 2000, seed: int = 7) -> dict:
    """Hull-White gain of every delta vs Black-Scholes, with 95% day-block bootstrap CIs."""
    cols = [*DELTAS, "unhedged"]
    sq = hedging_errors(panel) ** 2
    sq["quote_date"] = panel.quote_date.to_numpy()
    daily = sq.groupby("quote_date")[cols].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(reps, len(daily)))
    boot = daily.to_numpy()[idx].sum(axis=1)
    point = daily.sum().to_numpy()
    bs = cols.index("black_scholes")
    out = {"n": len(panel), "n_days": len(daily)}
    for j, m in enumerate(cols):
        out[f"rmse_{m}"] = float(np.sqrt(point[j] / len(panel)))
        out[f"gain_{m}"] = float(1 - point[j] / point[bs])
        lo, hi = np.percentile(1 - boot[:, j] / boot[:, bs], [2.5, 97.5])
        out[f"gain_{m}_lo"], out[f"gain_{m}_hi"] = float(lo), float(hi)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--period", choices=PERIODS, default="2022H2")
    args = ap.parse_args(argv)
    panel = build_panel(args.period)
    if args.period == "2022H2":
        dev = panel[panel.quote_date <= DEV_END]
    else:
        dev = build_panel("2022H2").query("quote_date <= @DEV_END")
    coefs = fit_minimum_variance(dev.dropna(subset=["delta_black_scholes"]))
    panel = add_minimum_variance(panel, coefs)
    panel.to_parquet(processed_path("hedging", args.period), index=False)
    print({k: np.round(v, 4).tolist() for k, v in coefs.items()})
    scored = panel if args.period != "2022H2" else panel[panel.quote_date > DEV_END]
    scored = scored.dropna(subset=[f"delta_{m}" for m in DELTAS])
    g = bootstrap_gain(scored)
    for m in [*DELTAS, "unhedged"]:
        print(f"{m:22s} rmse {g[f'rmse_{m}']:.3f}  gain {100 * g[f'gain_{m}']:+.1f}% "
              f"[{100 * g[f'gain_{m}_lo']:+.1f}, {100 * g[f'gain_{m}_hi']:+.1f}]")


if __name__ == "__main__":
    main()
