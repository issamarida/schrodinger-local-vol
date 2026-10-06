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
RAW_DIR_2019 = ROOT / "data" / "raw_2019_08"
PROCESSED_DIR = ROOT / "data" / "processed"

# Test periods. "2022H2" is the main sample (development Jul-Aug, holdout Sep-Dec); "2019-08" is a
# second, later-added out-of-period test from another vendor (scripts/fetch_data_2019.sh).
PERIODS = ("2022H2", "2019-08")

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


def load_rate_curves_2019(raw_dir: Path = RAW_DIR_2019) -> dict[str, np.ndarray]:
    """US Treasury par curves (decimal) for 2019, on the same tenors as the 2022 manifest."""
    t = pd.read_csv(raw_dir / "treasury_par_curve_2019.csv", parse_dates=["Date"])
    cols = ["1 Mo", "3 Mo", "1 Yr", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]
    return {d.strftime("%Y-%m-%d"): row[cols].to_numpy(float) / 100.0
            for d, row in t.set_index("Date").iterrows()}


def _read_hod_2019(path: Path) -> pd.DataFrame:
    """One day of the historicaloptiondata.com sample, renamed to the 2022 vendor's columns.

    SPX (monthly) contracts are AM-settled, SPXW contracts PM-settled.
    """
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "underlying": raw.UnderlyingSymbol, "root": raw.UnderlyingSymbol,
        "expiration": raw.Expiration, "type": raw.Type, "strike": raw.Strike.astype(float),
        "quote_date": raw.DataDate, "bid": raw.Bid, "ask": raw.Ask, "volume": raw.Volume,
        "open_interest": raw.OpenInterest, "underlying_close": raw.UnderlyingPrice,
        "settlement_time": np.where(raw.UnderlyingSymbol == "SPX", "AM", "PM"), "iv": raw.IV,
    })


def _read_hd_2022(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=_COLUMNS)
    df = df[df.underlying.isin(["SPX", "SPXW"])].copy()
    df["root"] = df.contract.str[:-15]
    return df


def load_day(path: Path, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    """All SPX/SPXW quotes of one file with T, r, discount factor and implied forward attached."""
    reader = _read_hod_2019 if path.name.startswith("L2_options_") else _read_hd_2022
    df = reader(path)
    df = df[df.underlying.isin(["SPX", "SPXW"])].copy()
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


def processed_path(name: str, period: str = "2022H2") -> Path:
    """data/processed/<name>.parquet for 2022H2, data/processed/<period>/<name>.parquet else."""
    base = PROCESSED_DIR if period == "2022H2" else PROCESSED_DIR / period
    return base / f"{name}.parquet"


def _daily_files(period: str) -> tuple[list[Path], dict[str, np.ndarray]]:
    if period == "2022H2":
        files, script = sorted(RAW_DIR.glob("*_options.csv")), "scripts/fetch_data.sh"
        curves = load_rate_curves(RAW_DIR) if files else {}
    elif period == "2019-08":
        files, script = sorted(RAW_DIR_2019.glob("L2_options_*.csv")), "scripts/fetch_data_2019.sh"
        curves = load_rate_curves_2019() if files else {}
    else:
        raise ValueError(f"unknown period {period!r}; expected one of {PERIODS}")
    if not files:
        raise FileNotFoundError(f"no daily files for {period}; run {script} first")
    return files, curves


def build_dataset(period: str = "2022H2", out: Path | None = None,
                  quote_filter: QuoteFilter = QuoteFilter()) -> pd.DataFrame:
    """Processes every daily file of a period into one filtered table (cached as parquet)."""
    out = out or processed_path("spx_quotes", period)
    files, curves = _daily_files(period)
    with ProcessPoolExecutor() as pool:
        frames = list(pool.map(partial(_load_filtered, curves=curves, quote_filter=quote_filter),
                               files))
    data = pd.concat(frames, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out, index=False)
    return data


def load_dataset(path: Path | None = None, period: str = "2022H2") -> pd.DataFrame:
    path = path or processed_path("spx_quotes", period)
    if not path.exists():
        return build_dataset(period, out=path)
    return pd.read_parquet(path)


_ALL_QUOTE_COLS = ["quote_date", "root", "expiration", "strike", "is_call", "F", "mid"]


def _load_two_sided(path: Path, curves: dict[str, np.ndarray], max_days: float) -> pd.DataFrame:
    df = load_day(path, curves)
    ok = (df.bid > 0) & (df.ask > df.bid) & (df.days <= max_days)
    return df.loc[ok, _ALL_QUOTE_COLS].reset_index(drop=True)


def load_all_quotes(period: str = "2022H2", max_days: float = 45.0) -> pd.DataFrame:
    """Every two-sided quote (calls and puts, any moneyness) up to `max_days`, unfiltered.

    The hedging test needs each option's mid on the next day even after it moved into the money.
    """
    path = processed_path("spx_all_quotes", period)
    if path.exists():
        return pd.read_parquet(path)
    files, curves = _daily_files(period)
    with ProcessPoolExecutor() as pool:
        frames = list(pool.map(partial(_load_two_sided, curves=curves, max_days=max_days), files))
    data = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)
    return data
