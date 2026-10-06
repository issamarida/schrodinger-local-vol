"""Tables and figures for the three follow-up studies: arbitrage, hedging and the surface fit.

Called by ``qp-report`` after the pricing tables. Each section reads the parquet written by its
study (``qp-arbitrage``, ``qp-hedge``, ``qp-surface``) and is skipped if that file is missing.
For 2022 every section reports the holdout (Sep-Dec); for other periods the whole period.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qpricing.arbitrage import STEP, butterflies, on_strikes, strike_grid
from qpricing.data import load_dataset, processed_path
from qpricing.hedging import DELTAS, hedging_errors
from qpricing.models import Schrodinger, Slice, SVIPrice

HOLDOUT_START = pd.Timestamp("2022-09-01")
# Hedges opened the session before a CPI release (13 Sep, 13 Oct, 10 Nov, 13 Dec 2022).
CPI_EVE_2022 = pd.to_datetime(["2022-09-12", "2022-10-12", "2022-11-09", "2022-12-12"])

ARB_MODELS = ["schrodinger", "heston", "adhoc_bs", "svi", "svi_price", "black_scholes"]
ARB_LABELS = {"schrodinger": "Schrödinger (per slice)", "heston": "Heston", "adhoc_bs":
              "Ad-hoc BS (quadratic smile)", "svi": "SVI, fit to implied vols",
              "svi_price": "SVI, fit to prices", "black_scholes": "Black-Scholes"}
DELTA_LABELS = {
    "black_scholes": "Black-Scholes, own implied vol (sticky strike)",
    "minimum_variance": "Minimum variance (Hull-White 2017)",
    "adhoc_bs": "Ad-hoc BS (smile floats with F)",
    "svi": "SVI, iv fit (smile floats with F)",
    "svi_price": "SVI, price fit (smile floats with F)",
    "heston": "Heston",
    "schrodinger": "Schrödinger, local-vol delta",
    "schrodinger_floating": "Schrödinger, smile floats with F",
}
SURFACE_LABELS = {
    "surface_homogeneous": "One well, time-homogeneous",
    "surface_clock": "One well on a fitted clock",
    "surface_piecewise": "Time-dependent well (bootstrapped)",
    "svi": "SVI (iv), total-variance interpolation",
    "svi_price": "SVI (price), total-variance interpolation",
    "adhoc_bs": "Ad-hoc BS, total-variance interpolation",
}
INK, INK_2, GRID, SURFACE, ACCENT, NEG = ("#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6",
                                          "#4a3aa7")


def _scope(df: pd.DataFrame, period: str, col: str = "quote_date") -> pd.DataFrame:
    return df[df[col] >= HOLDOUT_START] if period == "2022H2" else df


def _where(period: str) -> str:
    return "2022 holdout, Sep-Dec" if period == "2022H2" else period


def _boot_ratio(daily: pd.DataFrame, num: str, den: str, reps: int = 2000,
                seed: int = 7) -> tuple[float, float, float]:
    """1 - sum(num) / sum(den) with a 95% day-block bootstrap interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(reps, len(daily)))
    a, b = daily[num].to_numpy(), daily[den].to_numpy()
    boot = 1 - a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(1 - a.sum() / b.sum()), float(lo), float(hi)


def _fmt(point: float, lo: float, hi: float) -> str:
    return f"**{100 * point:+.1f}%** [{100 * lo:+.1f}, {100 * hi:+.1f}]"


# ---------------------------------------------------------------------------
# Arbitrage
# ---------------------------------------------------------------------------
def arbitrage_section(period: str) -> list[str]:
    fpath, cpath = processed_path("butterflies", period), processed_path("calendars", period)
    if not fpath.exists():
        return []
    flies = _scope(pd.read_parquet(fpath), period)
    cals = _scope(pd.read_parquet(cpath), period)
    m = flies[flies.model == "market"]
    lines = [
        f"## Static arbitrage: {_where(period)}", "",
        "Each calibrated slice is priced on a 1-point strike grid. A butterfly "
        "C(K-1) - 2C(K) + C(K+1) below -1e-10 F is a negative density. *Quoted*: between the "
        "lowest and highest calibrated strike. *Wings*: that range widened by half its width on "
        "each side in log-moneyness. Negative mass is the probability the smile assigns below "
        "zero, median over the violating slices.", "",
        f"Market check, same slices: {100 * m.executable.mean():.2f}% of {len(m):,} slices "
        "contain a butterfly that is still negative after buying the wings at the ask and "
        "selling the body at the bid (mids alone violate convexity on "
        f"{100 * m.violation.mean():.0f}% of slices, which is quote noise).", "",
        "| model | slices | butterfly arbitrage, quoted | in the wings | negative mass (wings) "
        "| calendar arbitrage, adjacent expiries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ARB_MODELS:
        f = flies[flies.model == model]
        q, w = f[f["range"] == "quoted"], f[f["range"] == "wings"]
        c = cals[cals.model == model]
        mass = w.loc[w.violation, "neg_mass"].median() if w.violation.any() else 0.0
        lines.append(f"| {ARB_LABELS[model]} | {len(q):,} | {100 * q.violation.mean():.1f}% | "
                     f"{100 * w.violation.mean():.1f}% | {mass:.2g} | "
                     f"{100 * c.violation.mean():.1f}% of {len(c):,} |")
    lines += ["", "Calendar arbitrage: undiscounted OTM prices normalised by the forward must "
              "not fall with maturity at fixed k = ln(K/F); checked on the overlap of the quoted "
              "ranges of each pair of adjacent expiries, tolerance 1e-7 F (0.0004 index points). "
              "The surface models below are checked the same way.", ""]
    return lines


def fig_density(period: str, path: Path) -> None:
    """Implied density of a holdout slice where price-fitted SVI goes negative in a wing."""
    flies = _scope(pd.read_parquet(processed_path("butterflies", period)), period)
    params = pd.read_parquet(processed_path("backtest_params", period))
    data = load_dataset(period=period)
    bad = flies[(flies.model == "svi_price") & (flies["range"] == "wings") & flies.violation
                & flies.days.between(5, 12)].sort_values("neg_mass")
    row = bad.iloc[int(0.8 * len(bad))]  # a clearly visible but not extreme case
    g = data[(data.quote_date == row.quote_date) & (data.root == row.root)
             & (data.expiration == row.expiration)]
    s = Slice.from_frame(g)
    grid = on_strikes(s, strike_grid(s, wings=True))
    K = grid.strikes[1:-1]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    sel = params[(params.calib_date == row.quote_date) & (params.root == row.root)
                 & (params.expiration == row.expiration)].set_index("model")["params"]
    for model, color, style, label in (
            (SVIPrice(), NEG, "--", "SVI fitted to prices"),
            (Schrodinger(), ACCENT, "-", "Schrödinger (positive local variance)")):
        theta = np.asarray(sel[model.name], float)
        dens = butterflies(grid, model.price(theta, grid)) / STEP ** 2
        ax.plot(K, 1e3 * dens, color=color, linewidth=2, linestyle=style, label=label)
        if model.name == "svi_price":
            ax.fill_between(K, 1e3 * dens, 0, where=dens < 0, color=color, alpha=0.25,
                            linewidth=0)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    # zoom on the wing where the SVI density is most negative
    svi_dens = butterflies(grid, SVIPrice().price(np.asarray(sel["svi_price"], float), grid))
    i = int(np.argmin(svi_dens))
    win = slice(max(i - 120, 0), min(i + 120, len(K)))
    ins = ax.inset_axes([0.62, 0.45, 0.35, 0.4])
    ins.set_facecolor(SURFACE)
    for model, color, style in ((SVIPrice(), NEG, "--"), (Schrodinger(), ACCENT, "-")):
        dens = butterflies(grid, model.price(np.asarray(sel[model.name], float), grid))
        ins.plot(K[win], 1e3 * dens[win] / STEP ** 2, color=color, linewidth=1.6,
                 linestyle=style)
    ins.axhline(0, color=INK_2, linewidth=0.8)
    ins.fill_between(K[win], 1e3 * svi_dens[win], 0, where=svi_dens[win] < 0, color=NEG,
                     alpha=0.25, linewidth=0)
    ins.tick_params(colors=INK_2, labelsize=7)
    ins.set_title("wing, zoomed", color=INK_2, fontsize=8)
    for side in ins.spines.values():
        side.set_color(GRID)
    lo, hi = s.strikes.min(), s.strikes.max()
    ax.axvspan(lo, hi, color=GRID, alpha=0.5, linewidth=0)
    ax.text(0.5 * (lo + hi), ax.get_ylim()[1] * 0.92, "quoted strikes", color=INK_2, fontsize=8,
            ha="center")
    ax.set_xlabel("strike", color=INK_2, fontsize=9)
    ax.set_ylabel("risk-neutral density (per 1,000 index points)", color=INK_2, fontsize=9)
    ax.set_title(f"Implied density, {row.root} {row.expiration.date()} ({int(row.days)}d) on "
                 f"{row.quote_date.date()}: shaded area is negative probability",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hedging
# ---------------------------------------------------------------------------
def hedging_table(panel: pd.DataFrame) -> pd.DataFrame:
    err = hedging_errors(panel)
    sq = err ** 2
    sq["quote_date"] = panel.quote_date.to_numpy()
    daily = sq.groupby("quote_date").sum()
    log_ratio = np.log(daily[DELTAS].div(daily.black_scholes, axis=0))
    rows = []
    for m in DELTAS:
        point, lo, hi = _boot_ratio(daily, m, "black_scholes")
        rows.append({"delta": m, "rmse": float(np.sqrt(sq[m].mean())), "gain": point,
                     "lo": lo, "hi": hi, "mean_day_gain": float(1 - np.exp(log_ratio[m].mean())),
                     "share_days_better": float((log_ratio[m] < 0).mean())})
    out = pd.DataFrame(rows)
    out.attrs["unhedged_rmse"] = float(np.sqrt(sq.unhedged.mean()))
    out.attrs["n"], out.attrs["days"] = len(panel), len(daily)
    return out


def _load_hedging(period: str) -> pd.DataFrame | None:
    path = processed_path("hedging", period)
    if not path.exists():
        return None
    panel = _scope(pd.read_parquet(path), period)
    return panel.dropna(subset=[f"delta_{m}" for m in DELTAS])


def hedging_section(period: str) -> list[str]:
    panel = _load_hedging(period)
    if panel is None:
        return []
    t = hedging_table(panel)
    ex = (hedging_table(panel[~panel.quote_date.isin(CPI_EVE_2022)]).set_index("delta")
          if period == "2022H2" else None)
    lines = [
        f"## Delta hedging: {_where(period)}", "",
        "Every OTM option in the calibration universe that is still quoted the next day is "
        "hedged once with futures on its expiry's forward: error = dV - delta dF, mids and "
        f"parity forwards, {t.attrs['n']:,} option-days over {t.attrs['days']} days. "
        "Gain = 1 - SSE(delta) / SSE(Black-Scholes), Hull & White (2017)'s measure, with 95% "
        "day-block bootstrap intervals. *Mean day* weighs every day equally (geometric mean of "
        "the daily SSE ratio), so a few large moves cannot dominate. Minimum-variance "
        "coefficients are fitted on Jul-Aug 2022 only."
        + (" *Ex-CPI* drops the four hedges held over a CPI release (13 Sep, 13 Oct, 10 Nov, "
           "13 Dec), two of which carry half of the squared error." if ex is not None else ""),
        "",
        "| delta | RMSE (pts) | gain vs BS | mean-day gain | days better than BS |"
        + (" gain ex-CPI |" if ex is not None else ""),
        "|---|---:|---:|---:|---:|" + ("---:|" if ex is not None else ""),
    ]
    for r in t.itertuples():
        row = (f"| {DELTA_LABELS[r.delta]} | {r.rmse:.2f} | {_fmt(r.gain, r.lo, r.hi)} | "
               f"{100 * r.mean_day_gain:+.1f}% | {100 * r.share_days_better:.0f}% |")
        if ex is not None:
            e = ex.loc[r.delta]
            row += f" {_fmt(e.gain, e.lo, e.hi)} |"
        lines.append(row)
    lines += ["", f"Unhedged RMSE: {t.attrs['unhedged_rmse']:.2f} index points.", ""]

    # Why: each delta's adjustment to BS against the adjustment that would have been optimal.
    buckets = pd.cut(panel.delta_black_scholes.abs(), [0, 0.1, 0.25, 0.5],
                     labels=["5-10", "10-25", "25-50"])
    lines += [
        "What the market paid for. For each |delta| bucket: the median adjustment each model "
        "makes to the BS delta, and the constant adjustment that would have minimised the "
        "squared hedging error ex post (least squares of the BS hedging error on dF).", "",
        "| option | abs(delta) | optimal ex post | local vol | smile floats with F | "
        "minimum variance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (is_call, b), g in panel.groupby(["is_call", buckets], observed=True):
        y = g.dV - g.delta_black_scholes * g.dF
        best = float((g.dF * y).sum() / (g.dF ** 2).sum())
        med = {m: float((g[f"delta_{m}"] - g.delta_black_scholes).median())
               for m in ("schrodinger", "schrodinger_floating", "minimum_variance")}
        lines.append(f"| {'call' if is_call else 'put'} | {b}% | {best:+.3f} | "
                     f"{med['schrodinger']:+.3f} | {med['schrodinger_floating']:+.3f} | "
                     f"{med['minimum_variance']:+.3f} |")
    return lines + [""]


def fig_hedging(period: str, path: Path) -> None:
    panel = _load_hedging(period)
    if panel is None:
        return
    t = hedging_table(panel).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, left=False)
    y = np.arange(len(t))
    lo = np.clip(100 * t.lo.to_numpy(), -100, None)
    for yi, r, low in zip(y, t.itertuples(), lo, strict=True):
        color = ACCENT if r.delta == "schrodinger" else INK_2
        ax.plot([low, 100 * r.hi], [yi, yi], color=color, linewidth=2, alpha=0.5,
                solid_capstyle="round")
        clipped = r.gain < -1
        ax.plot(-100 if clipped else 100 * r.gain, yi, "<" if clipped else "o", color=color,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(max(100 * r.hi, 100 * r.gain, -100) + 3, yi, f"{100 * r.gain:+.0f}%",
                color=INK, fontsize=8, va="center")
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, [DELTA_LABELS[d] for d in t.delta], color=INK, fontsize=8)
    ax.set_xlim(-100, 100)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("hedging-error variance reduction vs Black-Scholes delta (%, clipped at -100)",
                  color=INK_2, fontsize=8)
    ax.set_title(f"Daily delta hedging, {_where(period)}: gain with 95% CI", color=INK,
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------
def surface_section(period: str) -> list[str]:
    fpath = processed_path("surface_fits", period)
    if not fpath.exists():
        return []
    fits = _scope(pd.read_parquet(fpath), period)
    days = _scope(pd.read_parquet(processed_path("surface_days", period)), period)
    loo = _scope(pd.read_parquet(processed_path("surface_loo", period)), period)
    ref = fits[fits.model == "surface_piecewise"]  # every variant is scored on the same quotes
    per_slice_mae = ref.per_slice_sae.sum() / ref.n.sum()
    lines = [
        f"## One model for the whole surface: {_where(period)}", "",
        "Every expiry of a day (2-45 days, the per-slice universe) from one diffusion, "
        "calibrated per day on prices. Per-slice Schrödinger, 5 parameters per expiry: in-sample "
        f"MAE {per_slice_mae:.3f} on the same quotes.", "",
        "| model | parameters per day (median) | in-sample MAE | calendar arbitrage "
        "| seconds per day (median) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("surface_homogeneous", "surface_clock", "surface_piecewise"):
        f, d = fits[fits.model == model], days[days.model == model]
        lines.append(f"| {SURFACE_LABELS[model]} | {d.n_params.median():.0f} | "
                     f"{f.sae.sum() / f.n.sum():.3f} | {int(d.calendar_violations.sum())} of "
                     f"{int(d.calendar_pairs.sum()):,} pairs | {d.seconds.median():.0f} |")
    lines += [
        "", "Leave one expiry out: refit without an interior expiry, then price it. The "
        "Schrödinger surfaces price it from the diffusion itself; the per-slice benchmarks "
        "interpolate their neighbours' smiles linearly in total implied variance at fixed k. "
        "Improvement of the time-dependent well vs each, with 95% day-block intervals.", "",
        "| model | quotes | MAE | time-dependent well vs this |", "|---|---:|---:|---:|",
    ]
    daily = loo.pivot_table(index="quote_date", columns="model", values="sae", aggfunc="sum")
    n = loo[loo.model == "surface_piecewise"].groupby("quote_date").n.sum()
    common = daily.dropna().index
    for model in ("surface_piecewise", "surface_clock", "surface_homogeneous", "svi",
                  "svi_price", "adhoc_bs"):
        if model not in daily:
            continue
        mae = daily.loc[common, model].sum() / n.loc[common].sum()
        cmp = ("" if model == "surface_piecewise" else
               _fmt(*_boot_ratio(daily.loc[common], "surface_piecewise", model)))
        lines.append(f"| {SURFACE_LABELS[model]} | {n.loc[common].sum():,} | {mae:.3f} | {cmp} |")
    return lines + [""]
