"""Loading and cleaning SPX/SPXW end-of-day option quotes.

Source: the free 2022-H2 sample of HistoricalData.net (https://historicaldata.net), fetched by
``scripts/fetch_data.sh``. The raw files are licensed for use but not redistribution, so they are
never committed; this module turns them into a compact per-quote table cached under
``data/processed/``.

Conventions (following the vendor's methodology, README sections 5-6):

* ``T`` is calendar days / 365, minus one day for AM-settled contracts (they settle on the
  opening print, so the last trading session is the day before expiry).
* ``r`` is the Treasury par curve stored in ``manifest.json`` for that quote date, linearly
  interpolated in ``T`` and flat outside the first/last tenor.
* The forward is implied by put-call parity, ``F = K + e^{rT} (C - P)``, as the median over the
  five strikes nearest the index close with two-sided quotes on both legs.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "day_by_date"
PROCESSED_DIR = ROOT / "data" / "processed"

RATE_TENORS = np.array([1 / 12, 0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
_COLUMNS = [
    "contract", "underlying", "expiration", "type", "strike", "quote_date", "bid", "ask",
    "volume", "open_interest", "underlying_close", "settlement_time", "iv",
]


@dataclass(frozen=True)
class QuoteFilter:
    """Which quotes enter calibration and evaluation.

    Chosen on the July-August development window only (see README, "Protocol").
    """

    min_days: float = 2.0         # drop contracts in their final two sessions
    max_days: float = 45.0        # short-dated universe
    min_bid: float = 0.05         # a real, positive bid (the SPX minimum tick)
    max_rel_spread: float = 0.5   # (ask - bid) / mid
    otm_only: bool = True         # calls with K >= F, puts with K < F


def load_rate_curves(raw_dir: Path = RAW_DIR) -> dict[str, np.ndarray]:
    """Treasury par curves (decimal) by quote date, as recorded in the vendor manifest."""
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    return {
        day: np.array([np.nan if v is None else v / 100.0 for v in curve])
        for day, curve in manifest["rates"].items()
    }


def interpolate_rate(curve: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Linear interpolation in time, flat extrapolation, missing tenors dropped."""
    ok = np.isfinite(curve)
    return np.interp(T, RATE_TENORS[ok], curve[ok])


def implied_forwards(day: pd.DataFrame, n_strikes: int = 5) -> pd.Series:
    """Put-call-parity forward per (root, expiration)."""
    two_sided = day[(day.bid > 0) & (day.ask > 0)]
    calls = two_sided[two_sided.is_call].set_index(["root", "expiration", "strike"]).mid
    puts = two_sided[~two_sided.is_call].set_index(["root", "expiration", "strike"]).mid
    pairs = pd.concat({"c": calls, "p": puts}, axis=1, join="inner").reset_index()
    meta = day.groupby(["root", "expiration"])[["T", "r", "S"]].first()
    pairs = pairs.join(meta, on=["root", "expiration"])
    pairs["F"] = pairs.strike + np.exp(pairs.r * pairs["T"]) * (pairs.c - pairs.p)
    pairs["dist"] = (pairs.strike - pairs.S).abs()
    nearest = pairs.sort_values("dist").groupby(["root", "expiration"]).head(n_strikes)
    return nearest.groupby(["root", "expiration"]).F.median()


def load_day(path: Path, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    """All SPX/SPXW quotes of one file with T, r, discount factor and implied forward attached."""
    df = pd.read_csv(path, usecols=_COLUMNS)
    df = df[df.underlying.isin(["SPX", "SPXW"])].copy()
    df["root"] = df.contract.str[:-15]
    df["quote_date"] = pd.to_datetime(df.quote_date)
    df["expiration"] = pd.to_datetime(df.expiration)
    days = (df.expiration - df.quote_date).dt.days - (df.settlement_time == "AM").astype(int)
    df["days"] = days.astype(float)
    df = df[df.days > 0]
    df["T"] = df.days / 365.0
    curve = curves[df.quote_date.iloc[0].strftime("%Y-%m-%d")]
    df["r"] = interpolate_rate(curve, df["T"].to_numpy())
    df["df"] = np.exp(-df.r * df["T"])
    df["S"] = df.underlying_close
    df["is_call"] = df.type == "call"
    df["mid"] = 0.5 * (df.bid + df.ask)

    forwards = implied_forwards(df)
    df = df.join(forwards.rename("F"), on=["root", "expiration"])
    df = df[np.isfinite(df.F) & (df.F > 0)]
    df["k"] = np.log(df.strike / df.F)
    keep = [
        "quote_date", "root", "expiration", "days", "T", "r", "df", "S", "F", "strike", "k",
        "is_call", "bid", "ask", "mid", "volume", "open_interest", "iv",
    ]
    return df[keep].reset_index(drop=True)


def apply_filter(df: pd.DataFrame, f: QuoteFilter = QuoteFilter()) -> pd.DataFrame:
    spread = df.ask - df.bid
    mask = (
        (df.days >= f.min_days)
        & (df.days <= f.max_days)
        & (df.bid >= f.min_bid)
        & (df.ask > df.bid)
        & (spread / df.mid <= f.max_rel_spread)
    )
    if f.otm_only:
        mask &= (df.is_call & (df.strike >= df.F)) | (~df.is_call & (df.strike < df.F))
    return df[mask].reset_index(drop=True)


def _load_filtered(path: Path, curves: dict[str, np.ndarray],
                   quote_filter: QuoteFilter) -> pd.DataFrame:
    return apply_filter(load_day(path, curves), quote_filter)


def build_dataset(raw_dir: Path = RAW_DIR, out: Path | None = None,
                  quote_filter: QuoteFilter = QuoteFilter()) -> pd.DataFrame:
    """Processes every daily file into one filtered table (cached as parquet)."""
    out = out or PROCESSED_DIR / "spx_quotes.parquet"
    files = sorted(raw_dir.glob("*_options.csv"))
    if not files:
        raise FileNotFoundError(f"no daily files in {raw_dir}; run scripts/fetch_data.sh first")
    curves = load_rate_curves(raw_dir)
    with ProcessPoolExecutor() as pool:
        frames = list(pool.map(partial(_load_filtered, curves=curves, quote_filter=quote_filter),
                               files))
    data = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out, index=False)
    return data


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    path = path or PROCESSED_DIR / "spx_quotes.parquet"
    if not path.exists():
        return build_dataset(out=path)
    return pd.read_parquet(path)
