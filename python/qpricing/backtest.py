"""Next-day out-of-sample backtest on SPX/SPXW options.

Protocol, for every quote date t and every expiry slice with at least ``MIN_QUOTES`` quotes:

1. Calibrate each model to the slice's out-of-the-money mid prices on day t.
2. In-sample: price the same quotes on day t.
3. Out-of-sample ("out"): keep the parameters frozen and price the *next trading day's* quotes
   for the same expiry, using only that day's observable inputs (forward, discount factor, time
   to expiry). Extra parameters cannot help a model here unless they capture structure that
   persists.
4. Re-anchored out-of-sample ("anchored"): as 3, but first refit one volatility-level parameter
   per model to the two next-day quotes nearest the forward (the ATM vol a desk re-marks every
   morning). Those two anchor quotes are then excluded from scoring, so every scored price is
   still a genuine prediction of the smile's shape.

Usage::

    uv run qp-backtest                         # full sample, all cores
    uv run qp-backtest --start 2022-07-01 --end 2022-08-31   # development window only
    uv run qp-backtest --period 2019-08        # the second, out-of-period sample
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
    reanchor,
)

MIN_QUOTES = 10
MODEL_TYPES = (BlackScholes, AdHocBlackScholes, SVI, SVIPrice, Heston, Schrodinger)
_QUOTE_COLS = ["root", "expiration", "days", "k", "strike", "is_call", "F", "bid", "ask", "mid"]


def _price_rows(frame: pd.DataFrame, prices: dict[str, np.ndarray], sample: str,
                calib_date: pd.Timestamp) -> pd.DataFrame:
    out = frame[["quote_date", *_QUOTE_COLS]].copy()
    out["calib_date"] = calib_date
    out["sample"] = sample
    for name, p in prices.items():
        out[name] = p
    return out


def run_day(today: pd.DataFrame,
            tomorrow: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrates every slice of `today`; prices it in-sample and on `tomorrow`."""
    models = [M() for M in MODEL_TYPES]
    calib_date = today.quote_date.iloc[0]
    next_slices = {} if tomorrow is None else dict(list(tomorrow.groupby(["root", "expiration"])))
    rows, params = [], []

    for key, g in today.groupby(["root", "expiration"]):
        if len(g) < MIN_QUOTES:
            continue
        s = Slice.from_frame(g)
        nxt = next_slices.get(key)
        s_next = Slice.from_frame(nxt) if nxt is not None and len(nxt) >= MIN_QUOTES else None
        anchor = None
        if s_next is not None:
            anchor = np.zeros(len(nxt), dtype=bool)
            anchor[np.argsort(np.abs(s_next.k))[:2]] = True
        p_in, p_out, p_anc = {}, {}, {}
        for m in models:
            t0 = time.perf_counter()
            try:
                theta = m.calibrate(s)
                p_in[m.name] = m.price(theta, s)
                if s_next is not None:
                    p_out[m.name] = m.price(theta, s_next)
                    p_anc[m.name] = m.price(reanchor(m, theta, s_next, anchor), s_next)[~anchor]
            except Exception as exc:  # a failed fit is recorded, never silently dropped
                theta = np.full(m.n_params, np.nan)
                p_in[m.name] = np.full(len(g), np.nan)
                if s_next is not None:
                    p_out[m.name] = np.full(len(nxt), np.nan)
                    p_anc[m.name] = np.full(int((~anchor).sum()), np.nan)
                print(f"[warn] {m.name} {calib_date.date()} {key}: {exc}", flush=True)
            params.append({"calib_date": calib_date, "root": key[0], "expiration": key[1],
                           "days": s.days, "model": m.name, "params": theta.tolist(),
                           "seconds": time.perf_counter() - t0})
        rows.append(_price_rows(g, p_in, "in", calib_date))
        if s_next is not None:
            rows.append(_price_rows(nxt, p_out, "out", calib_date))
            rows.append(_price_rows(nxt[~anchor], p_anc, "anchored", calib_date))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(params)


def _worker(args):
    return run_day(*args)


def run(data: pd.DataFrame, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = sorted(data.quote_date.unique())
    by_day = {d: g for d, g in data.groupby("quote_date")}
    jobs = [(by_day[d], by_day[days[i + 1]] if i + 1 < len(days) else None)
            for i, d in enumerate(days)]
    results, params = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, job): job[0].quote_date.iloc[0] for job in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            r, p = fut.result()
            results.append(r)
            params.append(p)
            print(f"[{n}/{len(jobs)}] {futures[fut].date()} done", flush=True)
    res = pd.concat(results, ignore_index=True).sort_values(["calib_date", "sample", "root"])
    return res.reset_index(drop=True), pd.concat(params, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--start", default=None, help="first calibration date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last calibration date (YYYY-MM-DD)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--period", choices=PERIODS, default="2022H2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    args.out = args.out or processed_path("backtest", args.period)

    data = load_dataset(period=args.period)
    days = data.quote_date
    lo = pd.Timestamp(args.start) if args.start else days.min()
    hi = pd.Timestamp(args.end) if args.end else days.max()
    # Keep one extra trading day after `end` so its out-of-sample evaluation exists.
    all_days = np.sort(days.unique())
    after = all_days[all_days > np.datetime64(hi)]
    hi_eval = after[0] if len(after) else hi
    data = data[(days >= lo) & (days <= hi_eval)]

    t0 = time.time()
    res, params = run(data, args.workers)
    res = res[res.calib_date <= hi]
    params = params[params.calib_date <= hi]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(args.out, index=False)
    params.to_parquet(args.out.with_name(args.out.stem + "_params.parquet"), index=False)
    print(f"wrote {len(res):,} priced quotes to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
